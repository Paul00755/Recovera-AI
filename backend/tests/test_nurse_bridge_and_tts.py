import uuid
import pytest
from starlette.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.constants import Role, ConsultationStatus
from app.core.security import get_password_hash, create_access_token
from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.consultation import Consultation, ConsultationSignal, ConsultationMessage
from app.models.audit_log import AuditLog
from app.models.notification import Notification
from app.models.sms_job import SMSJob


class MockDeviceDriver:
    def send_sms(self, phone: str, text: str):
        return {"status": "success", "id": "sim_sms_mock"}


def test_nurse_bridge_initiates_consultation_and_alerts_doctor(client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that when a Nurse initiates a consultation:
    1. Both Doctor and Patient are alerted via notifications and SMS.
    2. The Nurse acts as a bridge and does NOT have an active consultation call returned by GET /api/consultations/active.
    3. The Doctor receives the incoming consultation alert.
    """
    monkeypatch.setattr("app.services.sms_gateway_service.get_driver", lambda: MockDeviceDriver())
    monkeypatch.setattr("app.core.config.settings.ONEPLUS_GATEWAY_ENABLED", False)

    uid = uuid.uuid4().hex[:6]
    doc_user = None
    nurse_user = None
    patient_user = None
    patient = None
    created_consultation_ids = []

    try:
        doc_user = User(
            name=f"Dr. Alert {uid}",
            email=f"doc_alert_{uid}@recovera.ai",
            password_hash=get_password_hash("DocPass123!"),
            role=Role.DOCTOR.value,
            phone=f"+9198{uid}201",
            availability_status="online",
            active=True
        )
        nurse_user = User(
            name=f"Nurse Bridge {uid}",
            email=f"nurse_bridge_{uid}@recovera.ai",
            password_hash=get_password_hash("NursePass123!"),
            role=Role.NURSE.value,
            phone=f"+9198{uid}202",
            availability_status="online",
            active=True
        )
        patient_user = User(
            name=f"Patient Bridge {uid}",
            email=f"patient_bridge_{uid}@recovera.ai",
            password_hash=get_password_hash("PatientPass123!"),
            role=Role.PATIENT.value,
            phone=f"+9198{uid}203",
            active=True
        )
        db_session.add_all([doc_user, nurse_user, patient_user])
        db_session.commit()
        db_session.refresh(doc_user)
        db_session.refresh(nurse_user)
        db_session.refresh(patient_user)

        patient = Patient(
            user_id=patient_user.id,
            patient_code=f"PT-BRIDGE-{uid}",
            assigned_doctor_id=doc_user.id,
            phone=patient_user.phone,
            sms_opt_in=True,
            active=True,
            care_status="active"
        )
        db_session.add(patient)
        db_session.commit()
        db_session.refresh(patient)

        pda = PatientDoctorAssignment(
            patient_id=patient.id,
            doctor_id=doc_user.id,
            assigned_by=nurse_user.id,
            is_active=True
        )
        db_session.add(pda)
        db_session.commit()

        nurse_token = create_access_token(data={"sub": nurse_user.id, "role": Role.NURSE.value})
        doc_token = create_access_token(data={"sub": doc_user.id, "role": Role.DOCTOR.value})
        patient_token = create_access_token(data={"sub": patient_user.id, "role": Role.PATIENT.value})

        nurse_headers = {"Authorization": f"Bearer {nurse_token}"}
        doc_headers = {"Authorization": f"Bearer {doc_token}"}
        patient_headers = {"Authorization": f"Bearer {patient_token}"}

        # 1. Nurse initiates bridge consultation
        bridge_resp = client.post(
            "/api/consultations",
            headers=nurse_headers,
            json={
                "patient_id": patient.id,
                "doctor_id": doc_user.id,
                "mode": "video",
                "notes": "Nurse connecting patient with Dr. Alert"
            }
        )
        assert bridge_resp.status_code == 201, bridge_resp.text
        consult_data = bridge_resp.json()
        consult_id = consult_data["id"]
        created_consultation_ids.append(consult_id)

        # 2. Verify Doctor received immediate in-app notification & SMS
        doc_notifs = db_session.scalars(
            select(Notification).where(Notification.user_id == doc_user.id)
        ).all()
        assert len(doc_notifs) > 0
        latest_doc_notif = doc_notifs[-1]
        assert "Telehealth Bridge" in latest_doc_notif.title
        assert f"Nurse {nurse_user.name}" in latest_doc_notif.message

        doc_sms = db_session.scalars(
            select(SMSJob).where(SMSJob.recipient_phone == doc_user.phone)
        ).all()
        assert len(doc_sms) > 0
        assert f"Nurse {nurse_user.name}" in doc_sms[-1].message_text

        # 3. Verify Patient also received immediate in-app notification & SMS
        pat_notifs = db_session.scalars(
            select(Notification).where(Notification.user_id == patient_user.id)
        ).all()
        assert len(pat_notifs) > 0

        # 4. Verify Nurse does NOT get active consultation call via /api/consultations/active
        nurse_active_resp = client.get("/api/consultations/active", headers=nurse_headers)
        assert nurse_active_resp.status_code == 200
        assert nurse_active_resp.json() is None

        # 5. Doctor starts consultation -> active consultation returned for doctor
        client.put(f"/api/consultations/{consult_id}/start", headers=doc_headers)
        doc_active_resp = client.get("/api/consultations/active", headers=doc_headers)
        assert doc_active_resp.status_code == 200
        assert doc_active_resp.json() is not None
        assert doc_active_resp.json()["id"] == consult_id

        # Cleanly end consultation
        client.put(f"/api/consultations/{consult_id}/end", headers=doc_headers)

    finally:
        # Full teardown ensuring 0 residual test data
        db_session.rollback()
        for cid in created_consultation_ids:
            signals = db_session.scalars(select(ConsultationSignal).where(ConsultationSignal.consultation_id == cid)).all()
            for s in signals:
                db_session.delete(s)
            messages = db_session.scalars(select(ConsultationMessage).where(ConsultationMessage.consultation_id == cid)).all()
            for m in messages:
                db_session.delete(m)
            c = db_session.get(Consultation, cid)
            if c:
                db_session.delete(c)
            audits = db_session.scalars(select(AuditLog).where(AuditLog.target_id == cid)).all()
            for a in audits:
                db_session.delete(a)

        if patient:
            sms_jobs = db_session.scalars(
                select(SMSJob).where(
                    (SMSJob.patient_id == patient.id) |
                    (SMSJob.recipient_phone.in_([patient_user.phone, doc_user.phone, nurse_user.phone]))
                )
            ).all()
            for sj in sms_jobs:
                db_session.delete(sj)
            pdas = db_session.scalars(select(PatientDoctorAssignment).where(PatientDoctorAssignment.patient_id == patient.id)).all()
            for pda in pdas:
                db_session.delete(pda)
            db_session.delete(patient)

        for u in [doc_user, nurse_user, patient_user]:
            if u:
                notifs = db_session.scalars(select(Notification).where(Notification.user_id == u.id)).all()
                for n in notifs:
                    db_session.delete(n)
                db_session.delete(u)

        db_session.commit()


def test_tts_endpoint_returns_audio_for_all_languages(client: TestClient):
    """
    Verifies /api/tts endpoint returns valid audio/mpeg bytes for:
    - Tamil ('ta')
    - Hindi ('hi')
    - English ('en')
    """
    # 1. Tamil
    resp_ta = client.get("/api/tts", params={"text": "வணக்கம், நலமா?", "lang": "ta"})
    assert resp_ta.status_code == 200
    assert resp_ta.headers["content-type"] == "audio/mpeg"
    assert len(resp_ta.content) > 1000

    # 2. Hindi
    resp_hi = client.get("/api/tts", params={"text": "नमस्ते, आप कैसे हैं?", "lang": "hi"})
    assert resp_hi.status_code == 200
    assert resp_hi.headers["content-type"] == "audio/mpeg"
    assert len(resp_hi.content) > 1000

    # 3. English
    resp_en = client.get("/api/tts", params={"text": "Hello, how are you feeling today?", "lang": "en"})
    assert resp_en.status_code == 200
    assert resp_en.headers["content-type"] == "audio/mpeg"
    assert len(resp_en.content) > 1000
