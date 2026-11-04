import os
import sys
import pytest
import datetime as dt
from fastapi.testclient import TestClient

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.main import app
from app.core.database import SessionLocal
from app.core.security import create_access_token, get_password_hash
from app.core.constants import Role, DoctorStatus, AppointmentStatus, MedicalSpecialty
from app.models.user import User
from app.models.patient import Patient
from app.models.appointment import Appointment
from app.models.consultation import Consultation

client = TestClient(app)

@pytest.fixture
def test_setup():
    db = SessionLocal()
    try:
        # 1. Doctor 1 (Neurology)
        doc1 = db.query(User).filter_by(email="neuro.doc@hospital.org").first()
        if not doc1:
            doc1 = User(
                email="neuro.doc@hospital.org",
                password_hash=get_password_hash("DocPass123!"),
                name="Dr. Aris Thorne",
                role=Role.DOCTOR.value,
                specialization="neurology",
                availability_status="online"
            )
            db.add(doc1)
            db.commit()
            db.refresh(doc1)

        # 2. Doctor 2 (ENT)
        doc2 = db.query(User).filter_by(email="ent.doc@hospital.org").first()
        if not doc2:
            doc2 = User(
                email="ent.doc@hospital.org",
                password_hash=get_password_hash("DocPass123!"),
                name="Dr. Elena Vance",
                role=Role.DOCTOR.value,
                specialization="ent",
                availability_status="online"
            )
            db.add(doc2)
            db.commit()
            db.refresh(doc2)

        # 3. Nurse
        nurse = db.query(User).filter_by(email="head.nurse@hospital.org").first()
        if not nurse:
            nurse = User(
                email="head.nurse@hospital.org",
                password_hash=get_password_hash("NursePass123!"),
                name="Nurse Clara",
                role=Role.NURSE.value
            )
            db.add(nurse)
            db.commit()
            db.refresh(nurse)

        # 4. Patient assigned to Doc 1
        p_user1 = db.query(User).filter_by(email="p1.phase6@hospital.org").first()
        if not p_user1:
            p_user1 = User(
                email="p1.phase6@hospital.org",
                password_hash=get_password_hash("PatientPass123!"),
                name="Patient John Neuro",
                role=Role.PATIENT.value
            )
            db.add(p_user1)
            db.commit()
            db.refresh(p_user1)

        pat1 = db.query(Patient).filter_by(user_id=p_user1.id).first()
        if not pat1:
            pat1 = Patient(
                user_id=p_user1.id,
                patient_code="P6NEURO01",
                active=True,
                assigned_doctor_id=doc1.id
            )
            db.add(pat1)
        else:
            pat1.assigned_doctor_id = doc1.id
        db.commit()
        db.refresh(pat1)

        # 5. Patient assigned to Doc 2
        p_user2 = db.query(User).filter_by(email="p2.phase6@hospital.org").first()
        if not p_user2:
            p_user2 = User(
                email="p2.phase6@hospital.org",
                password_hash=get_password_hash("PatientPass123!"),
                name="Patient Jane ENT",
                role=Role.PATIENT.value
            )
            db.add(p_user2)
            db.commit()
            db.refresh(p_user2)

        pat2 = db.query(Patient).filter_by(user_id=p_user2.id).first()
        if not pat2:
            pat2 = Patient(
                user_id=p_user2.id,
                patient_code="P6ENT002",
                active=True,
                assigned_doctor_id=doc2.id
            )
            db.add(pat2)
        else:
            pat2.assigned_doctor_id = doc2.id
        db.commit()
        db.refresh(pat2)

        tokens = {
            "doc1": create_access_token({"sub": doc1.id, "email": doc1.email, "role": doc1.role}),
            "doc2": create_access_token({"sub": doc2.id, "email": doc2.email, "role": doc2.role}),
            "nurse": create_access_token({"sub": nurse.id, "email": nurse.email, "role": nurse.role}),
            "p1": create_access_token({"sub": p_user1.id, "email": p_user1.email, "role": p_user1.role}),
        }

        return {
            "doc1": doc1,
            "doc2": doc2,
            "nurse": nurse,
            "pat1": pat1,
            "pat2": pat2,
            "tokens": tokens,
        }
    finally:
        db.close()

def test_doctor_registration_requires_specialization():
    """Verify that doctor registration mandates a valid medical specialization and phone."""
    # Attempt to register staff without phone -> 422
    resp_no_phone = client.post("/api/auth/register-staff", json={
        "email": "no.phone.doc@hospital.org",
        "password": "ValidPassword123!",
        "name": "Dr. No Phone",
        "role": "doctor",
        "staff_code": "RECOVERA-STAFF-2026",
        "specialization": "orthopaedics"
    })
    assert resp_no_phone.status_code == 422, "Expected 422 for missing phone number"

    # Attempt to register doctor without specialization -> 422
    resp = client.post("/api/auth/register-staff", json={
        "email": "invalid.doc@hospital.org",
        "password": "ValidPassword123!",
        "name": "Dr. No Specialty",
        "phone": "+919876543210",
        "role": "doctor",
        "staff_code": "RECOVERA-STAFF-2026"
    })
    assert resp.status_code == 422, f"Expected 422 for missing specialization, got {resp.status_code}"

    # Register doctor with valid specialization and phone -> 201
    resp_valid = client.post("/api/auth/register-staff", json={
        "email": f"doc.spec.{dt.datetime.now().timestamp()}@hospital.org",
        "password": "ValidPassword123!",
        "name": "Dr. Valid Spec",
        "phone": "+919876543210",
        "role": "doctor",
        "staff_code": "RECOVERA-STAFF-2026",
        "specialization": "orthopaedics"
    })
    assert resp_valid.status_code == 201

def test_scoped_patient_visibility(test_setup):
    """Doctors should ONLY see patients assigned to them. Nurses see all."""
    tokens = test_setup["tokens"]
    doc1_id = test_setup["doc1"].id
    doc2_id = test_setup["doc2"].id

    # Doc 1 queries patients -> only pat1
    r1 = client.get("/api/patients", headers={"Authorization": f"Bearer {tokens['doc1']}"})
    assert r1.status_code == 200
    p_ids1 = [p["id"] for p in r1.json()]
    assert test_setup["pat1"].id in p_ids1
    assert test_setup["pat2"].id not in p_ids1, "Doctor 1 saw Doctor 2's patient!"

    # Doc 2 queries patients -> only pat2
    r2 = client.get("/api/patients", headers={"Authorization": f"Bearer {tokens['doc2']}"})
    assert r2.status_code == 200
    p_ids2 = [p["id"] for p in r2.json()]
    assert test_setup["pat2"].id in p_ids2
    assert test_setup["pat1"].id not in p_ids2, "Doctor 2 saw Doctor 1's patient!"

    # Nurse queries patients -> sees both
    rn = client.get("/api/patients", headers={"Authorization": f"Bearer {tokens['nurse']}"})
    assert rn.status_code == 200
    p_ids_nurse = [p["id"] for p in rn.json()]
    assert test_setup["pat1"].id in p_ids_nurse
    assert test_setup["pat2"].id in p_ids_nurse

def test_nurse_assign_doctor_to_patient(test_setup):
    """Nurse can assign a patient to a different doctor."""
    nurse_tok = test_setup["tokens"]["nurse"]
    pat2_id = test_setup["pat2"].id
    doc1_id = test_setup["doc1"].id

    # Nurse reassigns pat2 to doc1
    resp = client.put(f"/api/patients/{pat2_id}/assign-doctor",
        headers={"Authorization": f"Bearer {nurse_tok}"},
        json={"doctor_id": doc1_id}
    )
    assert resp.status_code == 200

    # Doc 1 now sees pat2
    doc1_tok = test_setup["tokens"]["doc1"]
    r1 = client.get("/api/patients", headers={"Authorization": f"Bearer {doc1_tok}"})
    p_ids1 = [p["id"] for p in r1.json()]
    assert pat2_id in p_ids1

def test_doctor_availability_update(test_setup):
    """Doctor can set status to online or away with an away_until datetime."""
    doc1_tok = test_setup["tokens"]["doc1"]
    away_time = (dt.datetime.now() + dt.timedelta(hours=4)).isoformat()

    # Set away
    resp = client.put("/api/doctors/availability",
        headers={"Authorization": f"Bearer {doc1_tok}"},
        json={"status": "away", "away_until": away_time}
    )
    assert resp.status_code == 200
    assert resp.json()["availability_status"] == "away"

    # Set back to online
    resp2 = client.put("/api/doctors/availability",
        headers={"Authorization": f"Bearer {doc1_tok}"},
        json={"status": "online"}
    )
    assert resp2.status_code == 200
    assert resp2.json()["availability_status"] == "online"

def test_appointment_booking_and_overlap_prevention(test_setup):
    """Verify that booking 60m appointments prevents overlapping slots for the same doctor."""
    nurse_tok = test_setup["tokens"]["nurse"]
    doc1_id = test_setup["doc1"].id
    pat1_id = test_setup["pat1"].id

    tomorrow_10am = dt.datetime.combine(dt.date.today() + dt.timedelta(days=2), dt.time(10, 0))

    # Clean any prior appointment at this slot
    db = SessionLocal()
    try:
        db.query(Appointment).filter_by(doctor_id=doc1_id, start_time=tomorrow_10am).delete()
        db.commit()
    finally:
        db.close()

    # Book slot at 10:00 AM
    b1 = client.post("/api/appointments",
        headers={"Authorization": f"Bearer {nurse_tok}"},
        json={
            "patient_id": pat1_id,
            "doctor_id": doc1_id,
            "start_time": tomorrow_10am.isoformat(),
            "notes": "Post-discharge checkup"
        }
    )
    assert b1.status_code == 201
    appt_id = b1.json()["id"]

    # Attempt to book overlapping appointment (10:30 AM) with same doctor -> 409 Conflict
    overlap_time = tomorrow_10am + dt.timedelta(minutes=30)
    b2 = client.post("/api/appointments",
        headers={"Authorization": f"Bearer {nurse_tok}"},
        json={
            "patient_id": pat1_id,
            "doctor_id": doc1_id,
            "start_time": overlap_time.isoformat(),
            "notes": "Conflicting slot"
        }
    )
    assert b2.status_code == 409, f"Expected 409 Conflict, got {b2.status_code}"

    # Verify free slots for doctor on that date excludes 10:00 AM
    date_str = (dt.date.today() + dt.timedelta(days=2)).isoformat()
    slots_resp = client.get(f"/api/doctors/{doc1_id}/slots?date={date_str}",
        headers={"Authorization": f"Bearer {nurse_tok}"}
    )
    assert slots_resp.status_code == 200
    free_slots = slots_resp.json()
    assert tomorrow_10am.isoformat() not in free_slots

def test_consultation_lifecycle_and_doctor_extension(test_setup):
    """Verify 15-minute consultation: start, doctor extend (+5m, +10m), patient restriction, and end."""
    doc1_tok = test_setup["tokens"]["doc1"]
    p1_tok = test_setup["tokens"]["p1"]
    pat1_id = test_setup["pat1"].id
    doc1_id = test_setup["doc1"].id

    # 1. Create consultation
    c_resp = client.post("/api/consultations",
        headers={"Authorization": f"Bearer {p1_tok}"},
        json={"patient_id": pat1_id, "doctor_id": doc1_id, "mode": "video", "notes": "Urgent recovery question"}
    )
    assert c_resp.status_code == 201
    c_id = c_resp.json()["id"]
    assert c_resp.json()["status"] == "waiting"

    # 2. Doctor starts consultation -> status becomes active
    start_resp = client.put(f"/api/consultations/{c_id}/start", headers={"Authorization": f"Bearer {doc1_tok}"})
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "active"
    orig_expiry = dt.datetime.fromisoformat(start_resp.json()["expires_at"])

    # 3. Patient attempts to extend consultation -> 403 Forbidden (doctors only!)
    pat_ext = client.put(f"/api/consultations/{c_id}/extend",
        headers={"Authorization": f"Bearer {p1_tok}"},
        json={"minutes": 10}
    )
    assert pat_ext.status_code == 403, "Patients should not be able to extend consultation!"

    # 4. Doctor extends consultation by 10 minutes -> expires_at extended
    doc_ext = client.put(f"/api/consultations/{c_id}/extend",
        headers={"Authorization": f"Bearer {doc1_tok}"},
        json={"minutes": 10}
    )
    assert doc_ext.status_code == 200
    new_expiry = dt.datetime.fromisoformat(doc_ext.json()["expires_at"])
    assert new_expiry > orig_expiry

    # 5. Send message inside consultation
    msg_resp = client.post(f"/api/consultations/{c_id}/messages",
        headers={"Authorization": f"Bearer {doc1_tok}"},
        json={"message": "Please monitor your blood pressure carefully."}
    )
    assert msg_resp.status_code == 201

    # 6. End consultation
    end_resp = client.put(f"/api/consultations/{c_id}/end", headers={"Authorization": f"Bearer {doc1_tok}"})
    assert end_resp.status_code == 200
    assert end_resp.json()["status"] == "completed"

def test_nurse_dashboard_endpoint(test_setup):
    """Verify partitioned nurse dashboard returns today's appointments, missed, upcoming, and dept breakdowns."""
    nurse_tok = test_setup["tokens"]["nurse"]
    resp = client.get("/api/patients/nurse-dashboard", headers={"Authorization": f"Bearer {nurse_tok}"})
    assert resp.status_code == 200
    data = resp.json()
    assert "all_patients" in data
    assert "by_specialty" in data
    assert "needs_immediate_care" in data
    assert "todays_appointments" in data
    assert "missed_appointments" in data
    assert "upcoming_appointments" in data


def test_doctor_appointment_decisions_and_nurse_notification(test_setup):
    """Verify doctor can accept, deny, or request reschedule (with mandatory date and nurse alert)."""
    doc1_tok = test_setup["tokens"]["doc1"]
    nurse_tok = test_setup["tokens"]["nurse"]
    pat1_id = test_setup["pat1"].id
    doc1_id = test_setup["doc1"].id

    # 1. Nurse books appointment for unique future slot
    import random
    unique_day = random.randint(30, 300)
    appt_time = (dt.datetime.utcnow() + dt.timedelta(days=unique_day)).replace(hour=11, minute=0, second=0, microsecond=0)
    book_resp = client.post("/api/appointments",
        headers={"Authorization": f"Bearer {nurse_tok}"},
        json={
            "patient_id": pat1_id,
            "doctor_id": doc1_id,
            "start_time": appt_time.isoformat(),
            "notes": "Follow-up neurology consultation"
        }
    )
    assert book_resp.status_code == 201
    appt_id = book_resp.json()["id"]

    # Doctor should have received an appointment notification
    notifs_doc = client.get("/api/notifications", headers={"Authorization": f"Bearer {doc1_tok}"})
    assert notifs_doc.status_code == 200
    assert any(n["category"] == "appointment" for n in notifs_doc.json())

    # 2. Doctor requests reschedule without date -> 422
    err_resched = client.put(f"/api/appointments/{appt_id}/doctor-decision",
        headers={"Authorization": f"Bearer {doc1_tok}"},
        json={"decision": "reschedule", "notes": "Emergency surgery overlap"}
    )
    assert err_resched.status_code == 422, "reschedule_date must be mandatory"

    # 3. Doctor requests reschedule with date -> 200, status becomes rescheduled
    new_date = (dt.datetime.utcnow() + dt.timedelta(days=unique_day + 1)).replace(hour=14, minute=0, second=0, microsecond=0)
    resched_resp = client.put(f"/api/appointments/{appt_id}/doctor-decision",
        headers={"Authorization": f"Bearer {doc1_tok}"},
        json={
            "decision": "reschedule",
            "reschedule_date": new_date.isoformat(),
            "notes": "Attending neuro conference"
        }
    )
    assert resched_resp.status_code == 200
    assert resched_resp.json()["status"] == "rescheduled"
    assert resched_resp.json()["decision_status"] == "reschedule_requested"
    assert resched_resp.json()["reschedule_proposed_time"] is not None

    # 4. Nurse receives in-app alert for reschedule request
    nurse_notifs = client.get("/api/notifications", headers={"Authorization": f"Bearer {nurse_tok}"})
    assert nurse_notifs.status_code == 200
    resched_notifs = [n for n in nurse_notifs.json() if n["category"] == "reschedule_request"]
    assert len(resched_notifs) > 0
    assert "Doctor [" in resched_notifs[0]["message"]
    assert "asking for a reschedule" in resched_notifs[0]["message"]

    # 5. Doctor accepts another appointment
    appt_time2 = (dt.datetime.utcnow() + dt.timedelta(days=unique_day + 2)).replace(hour=9, minute=0, second=0, microsecond=0)
    book_resp2 = client.post("/api/appointments",
        headers={"Authorization": f"Bearer {nurse_tok}"},
        json={
            "patient_id": pat1_id,
            "doctor_id": doc1_id,
            "start_time": appt_time2.isoformat()
        }
    )
    appt_id2 = book_resp2.json()["id"]

    accept_resp = client.put(f"/api/appointments/{appt_id2}/doctor-decision",
        headers={"Authorization": f"Bearer {doc1_tok}"},
        json={"decision": "accept", "notes": "Confirmed, all clear"}
    )
    assert accept_resp.status_code == 200
    assert accept_resp.json()["decision_status"] == "accepted"


def test_in_app_notifications_lifecycle(test_setup):
    """Verify notifications unread count, listing, single read, and read-all."""
    nurse_tok = test_setup["tokens"]["nurse"]

    # 1. Unread count endpoint
    count_resp = client.get("/api/notifications/unread-count", headers={"Authorization": f"Bearer {nurse_tok}"})
    assert count_resp.status_code == 200
    assert "unread_count" in count_resp.json()

    # 2. List notifications
    list_resp = client.get("/api/notifications", headers={"Authorization": f"Bearer {nurse_tok}"})
    assert list_resp.status_code == 200
    notifs = list_resp.json()

    if notifs:
        first_id = notifs[0]["id"]
        # Mark single as read
        read_resp = client.put(f"/api/notifications/{first_id}/read", headers={"Authorization": f"Bearer {nurse_tok}"})
        assert read_resp.status_code == 200

    # 3. Mark all as read
    all_read = client.put("/api/notifications/read-all", headers={"Authorization": f"Bearer {nurse_tok}"})
    assert all_read.status_code == 200

    # Unread count should now be 0
    count_after = client.get("/api/notifications/unread-count", headers={"Authorization": f"Bearer {nurse_tok}"})
    assert count_after.json()["unread_count"] == 0

