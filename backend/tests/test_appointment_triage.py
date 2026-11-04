import datetime as dt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.main import app
from app.core.database import get_db
from app.models.user import User
from app.models.patient import Patient
from app.models.appointment import Appointment
from app.core.security import create_access_token

client = TestClient(app)

def _auth(email: str, db: Session):
    u = db.scalar(select(User).where(User.email == email))
    token = create_access_token({"sub": u.id, "email": u.email, "role": u.role})
    return {"Authorization": f"Bearer {token}"}

def test_patient_booking_nurse_triage_and_forwarding():
    db = next(get_db())
    try:
        nurse = db.scalar(select(User).where(User.role == "nurse", User.active == True))
        doctor = db.scalar(select(User).where(User.role == "doctor", User.active == True))
        patient = db.scalar(select(Patient).where(Patient.active == True))
        patient_user = db.get(User, patient.user_id)

        assert nurse and doctor and patient and patient_user

        pat_hdr = _auth(patient_user.email, db)
        nurse_hdr = _auth(nurse.email, db)
        doc_hdr = _auth(doctor.email, db)

        # 1. Patient requests/books an appointment
        slot_time = (dt.datetime.utcnow() + dt.timedelta(days=2)).replace(hour=10, minute=0, second=0, microsecond=0)
        book_resp = client.post(
            "/api/appointments",
            headers=pat_hdr,
            json={
                "patient_id": patient.id,
                "doctor_id": doctor.id,
                "start_time": slot_time.isoformat(),
                "notes": "Severe knee stiffness and pain while walking after surgery",
                "appointment_type": "follow_up"
            }
        )
        assert book_resp.status_code == 201, book_resp.text
        appt_data = book_resp.json()
        assert appt_data["decision_status"] == "nurse_review"
        appt_id = appt_data["id"]

        # 2. Nurse forwards to doctor (with optional slot adjustment & triage notes)
        forward_resp = client.put(
            f"/api/appointments/{appt_id}/nurse-forward",
            headers=nurse_hdr,
            json={
                "doctor_id": doctor.id,
                "start_time": slot_time.isoformat(),
                "triage_notes": "Assessed patient post-op symptoms: forward to Ortho for clinical examination."
            }
        )
        assert forward_resp.status_code == 200, forward_resp.text
        fwd_data = forward_resp.json()
        assert fwd_data["decision_status"] == "forwarded_to_doctor"
        assert fwd_data["booked_by"] == nurse.id
        assert "Assessed patient post-op symptoms" in fwd_data["notes"]

        # 3. Doctor accepts appointment
        dec_resp = client.put(
            f"/api/appointments/{appt_id}/doctor-decision",
            headers=doc_hdr,
            json={
                "decision": "accept",
                "notes": "Reviewed and confirmed."
            }
        )
        assert dec_resp.status_code == 200, dec_resp.text
        assert dec_resp.json()["decision_status"] == "accepted"

    finally:
        # Cleanup created appointment
        if 'appt_id' in locals():
            appt = db.get(Appointment, appt_id)
            if appt:
                db.delete(appt)
                db.commit()
        db.close()
