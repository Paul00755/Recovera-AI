import uuid
import pytest
import datetime as dt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.consultation import Consultation, ConsultationMessage, ConsultationSignal, ConsultationParticipant
from app.models.notification import Notification
from app.models.sms_job import SMSJob
from app.models.audit_log import AuditLog
from app.core.constants import Role, ConsultationStatus
from app.core.security import create_access_token, get_password_hash
from app.services.sms_gateway_service import MockDeviceDriver


def test_consultation_lifecycle_full(client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch):
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
            name=f"Dr. Consult {uid}",
            email=f"doc_c_{uid}@recovera.ai",
            password_hash=get_password_hash("DocPass123!"),
            role=Role.DOCTOR.value,
            phone=f"+9198{uid}101",
            availability_status="online",
            active=True
        )
        nurse_user = User(
            name=f"Nurse Consult {uid}",
            email=f"nurse_c_{uid}@recovera.ai",
            password_hash=get_password_hash("NursePass123!"),
            role=Role.NURSE.value,
            phone=f"+9198{uid}102",
            availability_status="online",
            active=True
        )
        patient_user = User(
            name=f"Patient Consult {uid}",
            email=f"patient_c_{uid}@recovera.ai",
            password_hash=get_password_hash("PatientPass123!"),
            role=Role.PATIENT.value,
            phone=f"+9198{uid}103",
            active=True
        )
        db_session.add_all([doc_user, nurse_user, patient_user])
        db_session.commit()
        db_session.refresh(doc_user)
        db_session.refresh(nurse_user)
        db_session.refresh(patient_user)

        patient = Patient(
            user_id=patient_user.id,
            patient_code=f"PT-CONSULT-{uid}",
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
            assigned_by=doc_user.id,
            is_active=True
        )
        db_session.add(pda)
        db_session.commit()
        doc_token = create_access_token(data={"sub": doc_user.id, "role": Role.DOCTOR.value})
        nurse_token = create_access_token(data={"sub": nurse_user.id, "role": Role.NURSE.value})
        doc_headers = {"Authorization": f"Bearer {doc_token}"}
        nurse_headers = {"Authorization": f"Bearer {nurse_token}"}

        # 1. Doctor creates consultation request
        resp = client.post(
            "/api/consultations",
            headers=doc_headers,
            json={
                "patient_id": patient.id,
                "doctor_id": doc_user.id,
                "mode": "video",
                "notes": "Post-discharge 15m review"
            }
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        consult_id = data["id"]
        created_consultation_ids.append(consult_id)

        assert data["status"] == ConsultationStatus.WAITING.value
        assert data["mode"] == "video"
        assert data["created_at"].endswith("Z")
        assert data["expires_at"].endswith("Z")

        # 2. Doctor starts consultation -> status becomes active, doctor availability becomes busy
        start_resp = client.put(f"/api/consultations/{consult_id}/start", headers=doc_headers)
        assert start_resp.status_code == 200, start_resp.text
        start_data = start_resp.json()
        assert start_data["status"] == ConsultationStatus.ACTIVE.value
        assert start_data["started_at"].endswith("Z")
        assert start_data["expires_at"].endswith("Z")

        # Verify doctor is busy
        db_session.refresh(doc_user)
        assert doc_user.availability_status == "busy"

        # Verify idempotency of /start
        idempotent_resp = client.put(f"/api/consultations/{consult_id}/start", headers=doc_headers)
        assert idempotent_resp.status_code == 200
        assert idempotent_resp.json()["status"] == ConsultationStatus.ACTIVE.value

        # 3. Doctor extends consultation by 5 minutes
        extend_resp = client.put(
            f"/api/consultations/{consult_id}/extend",
            headers=doc_headers,
            json={"minutes": 5}
        )
        assert extend_resp.status_code == 200, extend_resp.text
        ext_data = extend_resp.json()
        assert ext_data["expires_at"].endswith("Z")
        assert "extended by 5 minutes" in ext_data["message"]

        # 4. Nurse creates and starts consultation
        nurse_create = client.post(
            "/api/consultations",
            headers=nurse_headers,
            json={
                "patient_id": patient.id,
                "doctor_id": doc_user.id,
                "mode": "audio",
                "notes": "Nurse assisted connect"
            }
        )
        assert nurse_create.status_code == 201
        nurse_consult_id = nurse_create.json()["id"]
        created_consultation_ids.append(nurse_consult_id)

        # Nurse starts consultation
        nurse_start = client.put(f"/api/consultations/{nurse_consult_id}/start", headers=nurse_headers)
        assert nurse_start.status_code == 200, nurse_start.text
        assert nurse_start.json()["status"] == ConsultationStatus.ACTIVE.value

        # Nurse extends consultation by 10 minutes
        nurse_ext = client.put(
            f"/api/consultations/{nurse_consult_id}/extend",
            headers=nurse_headers,
            json={"minutes": 10}
        )
        assert nurse_ext.status_code == 200, nurse_ext.text
        assert "extended by 10 minutes" in nurse_ext.json()["message"]

        # 5. Messaging and Signaling UTC timestamp verification
        msg_resp = client.post(
            f"/api/consultations/{consult_id}/messages",
            headers=doc_headers,
            json={"message": "Hello from doctor"}
        )
        assert msg_resp.status_code == 201
        assert msg_resp.json()["created_at"].endswith("Z")

        get_msgs = client.get(f"/api/consultations/{consult_id}/messages", headers=doc_headers)
        assert get_msgs.status_code == 200
        for m in get_msgs.json():
            assert m["created_at"].endswith("Z")

        # Test translation endpoint
        trans_resp = client.post(
            f"/api/consultations/{consult_id}/translate-messages",
            headers=doc_headers,
            json={"target_lang": "en"}
        )
        assert trans_resp.status_code == 200
        trans_data = trans_resp.json()
        assert len(trans_data) >= 1
        assert "translated_message" in trans_data[0]

        sig_resp = client.post(
            f"/api/consultations/{consult_id}/signals",
            headers=doc_headers,
            json={"signal_type": "offer", "signal_data": "{}"}
        )
        assert sig_resp.status_code == 201
        assert sig_resp.json()["created_at"].endswith("Z")

        # 6. End consultation -> status completed, doctor availability online, idempotent
        end_resp = client.put(f"/api/consultations/{consult_id}/end", headers=doc_headers)
        assert end_resp.status_code == 200, end_resp.text
        assert end_resp.json()["status"] == ConsultationStatus.COMPLETED.value
        assert end_resp.json()["ended_at"].endswith("Z")

        db_session.refresh(doc_user)
        assert doc_user.availability_status == "online"

        # End nurse consultation as well
        client.put(f"/api/consultations/{nurse_consult_id}/end", headers=nurse_headers)

        # 7. Test Accept / Reject / Join Workflow (Patient requests -> Doctor accepts -> Both join -> Timer begins)
        pat_token = create_access_token(data={"sub": patient_user.id, "role": Role.PATIENT.value})
        pat_headers = {"Authorization": f"Bearer {pat_token}"}

        # 7a. Patient requests consultation
        pat_req = client.post(
            "/api/consultations",
            headers=pat_headers,
            json={
                "patient_id": patient.id,
                "doctor_id": doc_user.id,
                "mode": "video",
                "notes": "Patient requesting checkup"
            }
        )
        assert pat_req.status_code == 201
        pat_consult = pat_req.json()
        created_consultation_ids.append(pat_consult["id"])
        assert pat_consult["status"] == ConsultationStatus.WAITING.value
        assert pat_consult["initiated_by"] == patient_user.id
        assert pat_consult["both_joined"] is False

        # 7b. Doctor accepts consultation
        acc_resp = client.put(f"/api/consultations/{pat_consult['id']}/accept", headers=doc_headers)
        assert acc_resp.status_code == 200, acc_resp.text
        assert acc_resp.json()["status"] == ConsultationStatus.ACCEPTED.value

        # 7c. Doctor joins room
        doc_join = client.post(f"/api/consultations/{pat_consult['id']}/join", headers=doc_headers)
        assert doc_join.status_code == 200
        assert doc_join.json()["active_participants_count"] == 1
        assert doc_join.json()["both_joined"] is False
        # Status should NOT be active yet because only doctor has joined
        assert doc_join.json()["status"] == ConsultationStatus.ACCEPTED.value

        # 7d. Patient joins room -> Both now joined!
        pat_join = client.post(f"/api/consultations/{pat_consult['id']}/join", headers=pat_headers)
        assert pat_join.status_code == 200
        assert pat_join.json()["active_participants_count"] == 2
        assert pat_join.json()["both_joined"] is True
        # Status transitions to ACTIVE and started_at is now set!
        assert pat_join.json()["status"] == ConsultationStatus.ACTIVE.value
        assert pat_join.json()["started_at"] is not None

        # Clean up this consultation
        client.put(f"/api/consultations/{pat_consult['id']}/end", headers=doc_headers)

        # 7e. Test Reject Workflow
        pat_req2 = client.post(
            "/api/consultations",
            headers=pat_headers,
            json={
                "patient_id": patient.id,
                "doctor_id": doc_user.id,
                "mode": "audio",
                "notes": "Patient asking audio call"
            }
        )
        assert pat_req2.status_code == 201
        pat_consult2 = pat_req2.json()
        created_consultation_ids.append(pat_consult2["id"])

        rej_resp = client.put(f"/api/consultations/{pat_consult2['id']}/reject", headers=doc_headers)
        assert rej_resp.status_code == 200
        assert rej_resp.json()["status"] == ConsultationStatus.REJECTED.value

    finally:
        # Full teardown ensuring 0 residual test data
        db_session.rollback()
        for cid in created_consultation_ids:
            parts = db_session.scalars(select(ConsultationParticipant).where(ConsultationParticipant.consultation_id == cid)).all()
            for p in parts:
                db_session.delete(p)
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
            for j in sms_jobs:
                db_session.delete(j)

            notifs = db_session.scalars(
                select(Notification).where(Notification.user_id.in_([patient_user.id, doc_user.id, nurse_user.id]))
            ).all()
            for n in notifs:
                db_session.delete(n)

            pdas = db_session.scalars(select(PatientDoctorAssignment).where(PatientDoctorAssignment.patient_id == patient.id)).all()
            for a in pdas:
                db_session.delete(a)

            db_session.delete(patient)

        if patient_user:
            db_session.delete(patient_user)
        if nurse_user:
            db_session.delete(nurse_user)
        if doc_user:
            db_session.delete(doc_user)
        db_session.commit()
