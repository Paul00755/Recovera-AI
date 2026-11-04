import uuid
import pytest
import datetime as dt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.user import User
from app.models.patient import Patient
from app.models.appointment import Appointment
from app.models.notification import Notification
from app.models.sms_job import SMSJob
from app.models.audit_log import AuditLog
from app.core.constants import Role, AppointmentStatus, SMSJobStatus
from app.core.security import create_access_token, get_password_hash
from app.services.sms_gateway_service import process_due_sms_jobs, MockDeviceDriver


def test_followup_appointment_three_notification_schedules(client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch):
    # Ensure immediate mock SMS dispatch without network calls
    monkeypatch.setattr("app.services.sms_gateway_service.get_driver", lambda: MockDeviceDriver())
    monkeypatch.setattr("app.core.config.settings.ONEPLUS_GATEWAY_ENABLED", False)

    uid = uuid.uuid4().hex[:6]

    # Create doctor
    doc_user = User(
        name=f"Dr. Test {uid}",
        email=f"doc_{uid}@recovera.ai",
        password_hash=get_password_hash("DocPass123!"),
        role=Role.DOCTOR.value,
        phone=f"+9198{uid}001",
        active=True
    )
    # Create nurse
    nurse_user = User(
        name=f"Nurse Test {uid}",
        email=f"nurse_{uid}@recovera.ai",
        password_hash=get_password_hash("NursePass123!"),
        role=Role.NURSE.value,
        phone=f"+9198{uid}002",
        active=True
    )
    # Create patient user & patient
    patient_user = User(
        name=f"Patient Test {uid}",
        email=f"patient_{uid}@recovera.ai",
        password_hash=get_password_hash("PatientPass123!"),
        role=Role.PATIENT.value,
        phone=f"+9198{uid}003",
        active=True
    )
    db_session.add_all([doc_user, nurse_user, patient_user])
    db_session.commit()
    db_session.refresh(doc_user)
    db_session.refresh(nurse_user)
    db_session.refresh(patient_user)

    patient = Patient(
        user_id=patient_user.id,
        patient_code=f"PT-TEST-{uid}",
        assigned_doctor_id=doc_user.id,
        phone=patient_user.phone,
        sms_opt_in=True,
        active=True,
        care_status="active"
    )
    db_session.add(patient)
    db_session.commit()
    db_session.refresh(patient)

    created_appt_id = None

    try:
        token = create_access_token(data={"sub": nurse_user.id, "role": Role.NURSE.value})
        headers = {"Authorization": f"Bearer {token}"}

        # Schedule appointment 3 days in the future
        future_start = (dt.datetime.utcnow() + dt.timedelta(days=3)).replace(minute=0, second=0, microsecond=0)
        payload = {
            "patient_id": patient.id,
            "doctor_id": doc_user.id,
            "start_time": future_start.isoformat(),
            "notes": "Follow-up check on recovery progress",
            "appointment_type": "follow_up"
        }

        response = client.post("/api/appointments", json=payload, headers=headers)
        assert response.status_code == 201, response.text
        data = response.json()
        created_appt_id = data["id"]
        assert data["appointment_type"] == "follow_up"
        assert data["doctor_name"] == doc_user.name
        assert data["patient_name"] == patient_user.name

        # Ensure session sees committed transactions
        db_session.expire_all()

        # 1. Verify in-app notifications
        # Patient in-app notification
        patient_notif = db_session.scalar(
            select(Notification).where(
                Notification.user_id == patient_user.id,
                Notification.category == "appointment"
            )
        )
        assert patient_notif is not None
        assert "Follow-up Appointment Scheduled" in patient_notif.title
        assert doc_user.name in patient_notif.message

        # Doctor in-app notification
        doc_notif = db_session.scalar(
            select(Notification).where(
                Notification.user_id == doc_user.id,
                Notification.category == "appointment"
            )
        )
        assert doc_notif is not None

        # 2. Verify all THREE SMS schedules exist
        # Schedule 1: Immediate confirmation
        key_confirm = f"appt_confirm_{created_appt_id}"
        job_confirm = db_session.scalar(select(SMSJob).where(SMSJob.idempotency_key == key_confirm))
        assert job_confirm is not None
        assert job_confirm.state == SMSJobStatus.SENT.value
        assert "has been confirmed for" in job_confirm.message_text

        # Schedule 2: 24h prior reminder (1 day before)
        key_24h = f"appt_remind_24h_{created_appt_id}"
        job_24h = db_session.scalar(select(SMSJob).where(SMSJob.idempotency_key == key_24h))
        assert job_24h is not None
        assert job_24h.state == SMSJobStatus.QUEUED.value
        expected_24h = future_start - dt.timedelta(days=1)
        assert abs((job_24h.scheduled_for - expected_24h).total_seconds()) < 2
        assert "tomorrow" in job_24h.message_text

        # Schedule 3: 2h prior reminder (2 hours before)
        key_2h = f"appt_remind_2h_{created_appt_id}"
        job_2h = db_session.scalar(select(SMSJob).where(SMSJob.idempotency_key == key_2h))
        assert job_2h is not None
        assert job_2h.state == SMSJobStatus.QUEUED.value
        expected_2h = future_start - dt.timedelta(hours=2)
        assert abs((job_2h.scheduled_for - expected_2h).total_seconds()) < 2
        assert "2 hours" in job_2h.message_text

        # 3. Test background dispatch of due SMS jobs
        # Fast-forward simulated time to 24h reminder time
        due_count = process_due_sms_jobs(db_session, now=expected_24h + dt.timedelta(seconds=5))
        assert due_count >= 1
        db_session.refresh(job_24h)
        assert job_24h.state == SMSJobStatus.SENT.value

    finally:
        # Full test cleanup
        db_session.rollback()
        # Delete all SMS jobs referencing patient or test user phones
        sms_jobs = db_session.scalars(
            select(SMSJob).where(
                (SMSJob.patient_id == patient.id) |
                (SMSJob.recipient_phone.in_([patient_user.phone, doc_user.phone, nurse_user.phone]))
            )
        ).all()
        for j in sms_jobs:
            db_session.delete(j)

        if created_appt_id:
            appt = db_session.get(Appointment, created_appt_id)
            if appt:
                db_session.delete(appt)
            audits = db_session.scalars(select(AuditLog).where(AuditLog.target_id == created_appt_id)).all()
            for a in audits:
                db_session.delete(a)

        # Delete notifications
        notifs = db_session.scalars(
            select(Notification).where(Notification.user_id.in_([patient_user.id, doc_user.id, nurse_user.id]))
        ).all()
        for n in notifs:
            db_session.delete(n)

        # Delete patient and users
        db_session.delete(patient)
        db_session.delete(patient_user)
        db_session.delete(nurse_user)
        db_session.delete(doc_user)
        db_session.commit()
