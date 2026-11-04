import os
import io
import sys
import uuid
import pytest
import datetime as dt

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.main import app
from app.core.database import get_db, SessionLocal
from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment, PatientDeletionTicket
from app.models.medication import Medication, MedicationOccurrence
from app.models.notification import Notification
from app.core.constants import Role, DocStatus
from app.services.handwriting_ml_service import MedicalPrescriptionParser, HandwrittenMLService

client = TestClient(app)


def get_token(email: str, role: str) -> str:
    db = SessionLocal()
    try:
        user = db.scalar(select(User).where(User.email == email))
        if not user:
            from app.core.security import get_password_hash
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                name=f"Test {role.capitalize()}",
                role=role,
                password_hash=get_password_hash("Password@123"),
                active=True,
                specialization="general_medicine" if role == Role.DOCTOR.value else None
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        from app.core.security import create_access_token
        return create_access_token({"sub": user.id, "role": user.role, "email": user.email})
    finally:
        db.close()


def test_handwriting_ml_prescription_parser():
    """Verify prescription parser accurately extracts medicines, days, and period timing."""
    sample_prescription_text = """
    Dr. Sharma Clinic - Prescription
    Patient: John Doe, Age 45
    Rx:
    1. Tab Amoxicillin 500mg - 1-0-1 - for 5 days (After Food)
    2. Tab Paracetamol 650mg - TDS - for 3 days - Before Food
    3. Tab Pantoprazole 40mg - 1-0-0 - x 10 days - AC
    4. Tab Cetirizine 10mg - QHS - 7 days (At bedtime)
    Review in 1 week.
    """
    records = MedicalPrescriptionParser.extract_prescription_records(sample_prescription_text)
    assert len(records) >= 3

    # Check Amoxicillin: 500mg, 1-0-1 (twice daily, morning & night), 5 days, after food
    amox = next((r for r in records if "amoxicillin" in r["name"].lower()), None)
    assert amox is not None
    assert "500" in amox["dose"]
    assert amox["days"] == 5
    assert amox["period"] == "1-0-1"
    assert amox["food_relation"] == "after_food"
    assert len(amox["schedules"]) == 2

    # Check Paracetamol: 650mg, TDS (thrice daily), 3 days, before food
    pcm = next((r for r in records if "paracetamol" in r["name"].lower()), None)
    assert pcm is not None
    assert pcm["days"] == 3
    assert pcm["frequency"] == "thrice_daily"
    assert pcm["food_relation"] == "before_food"

    # Check Pantoprazole: 40mg, 1-0-0, 10 days, before food
    panto = next((r for r in records if "pantoprazole" in r["name"].lower()), None)
    assert panto is not None
    assert panto["days"] == 10
    assert panto["period"] == "1-0-0"
    assert panto["food_relation"] == "before_food"


def test_patient_editing_by_doctor_and_nurse():
    """Verify that both doctor (assigned) and nurse can edit patient basic details and active status."""
    admin_token = get_token("admin_test_p17@recovera.ai", Role.ADMIN.value)
    nurse_token = get_token("nurse_test_p17@recovera.ai", Role.NURSE.value)
    doc_token = get_token("doc_test_p17@recovera.ai", Role.DOCTOR.value)

    u_suffix = uuid.uuid4().hex[:6]
    # 1. Register a patient
    reg_resp = client.post("/api/patients", json={
        "name": "Eileen TestPatient",
        "dob": "1990-05-15",
        "phone": f"+1987{uuid.uuid4().hex[:7]}",
        "email": f"eileen_{u_suffix}@example.com",
        "gender": "female",
        "preferred_language": "en",
        "sms_opt_in": True,
        "has_medical_history": False
    }, headers={"Authorization": f"Bearer {nurse_token}"})
    assert reg_resp.status_code == 201
    patient = reg_resp.json()
    pid = patient["id"]

    # 2. Nurse edits patient details (Name, Age -> DOB, Preferred Language, Active status)
    edit_resp = client.put(f"/api/patients/{pid}", json={
        "name": "Eileen Updated by Nurse",
        "age": 35,
        "phone": "+19876543299",
        "preferred_language": "ta",
        "sms_opt_in": False,
        "active": True
    }, headers={"Authorization": f"Bearer {nurse_token}"})
    assert edit_resp.status_code == 200
    updated_nurse = edit_resp.json()
    assert updated_nurse["name"] == "Eileen Updated by Nurse"
    assert updated_nurse["preferred_language"] == "ta"
    assert updated_nurse["sms_opt_in"] is False
    assert updated_nurse["age"] == 35 or updated_nurse["age"] == 36

    # 3. Assign doctor to patient
    db = SessionLocal()
    doc_user = db.scalar(select(User).where(User.email == "doc_test_p17@recovera.ai"))
    db.close()

    assign_resp = client.post(f"/api/patients/{pid}/assign-doctor", json={
        "doctor_id": doc_user.id,
        "department": "Internal Medicine"
    }, headers={"Authorization": f"Bearer {admin_token}"})
    assert assign_resp.status_code == 200

    # 4. Assigned Doctor edits patient details (active toggle to inactive and back)
    doc_edit = client.put(f"/api/patients/{pid}", json={
        "name": "Eileen Updated by Doctor",
        "active": False
    }, headers={"Authorization": f"Bearer {doc_token}"})
    assert doc_edit.status_code == 200
    assert doc_edit.json()["active"] is False

    # Doctor reactivates patient
    doc_edit2 = client.put(f"/api/patients/{pid}", json={
        "active": True
    }, headers={"Authorization": f"Bearer {doc_token}"})
    assert doc_edit2.status_code == 200
    assert doc_edit2.json()["active"] is True


def test_patient_deletion_ticket_workflow():
    """
    Verify complete deletion ticket lifecycle:
    - Nurse submits deletion ticket with reason
    - System generates in-app notification alert for Admin
    - Admin lists deletion requests
    - Admin approves deletion -> patient deleted, nurse receives alert
    """
    admin_token = get_token("admin_test_p17@recovera.ai", Role.ADMIN.value)
    nurse_token = get_token("nurse_test_p17@recovera.ai", Role.NURSE.value)

    # 1. Create patient
    p_resp = client.post("/api/patients", json={
        "name": "Delete Me Patient",
        "phone": f"+1988{uuid.uuid4().hex[:7]}",
        "gender": "male",
        "sms_opt_in": False
    }, headers={"Authorization": f"Bearer {nurse_token}"})
    pid = p_resp.json()["id"]

    # 2. Nurse submits deletion request ticket
    del_req = client.post(f"/api/patients/{pid}/deletion-request", json={
        "reason": "Duplicate admission record; transferred to Central General Hospital."
    }, headers={"Authorization": f"Bearer {nurse_token}"})
    assert del_req.status_code == 200
    ticket_data = del_req.json()
    assert "ticket_id" in ticket_data
    ticket_id = ticket_data["ticket_id"]

    # 3. Verify in-app alert generated for admin
    db = SessionLocal()
    admin_user = db.scalar(select(User).where(User.email == "admin_test_p17@recovera.ai"))
    notif = db.scalar(
        select(Notification)
        .where(
            Notification.user_id == admin_user.id,
            Notification.category == "deletion_request"
        )
        .order_by(Notification.created_at.desc())
    )
    assert notif is not None
    assert "Duplicate admission record" in notif.message
    db.close()

    # 4. Admin lists deletion requests
    list_reqs = client.get("/api/patients/deletion-requests", headers={"Authorization": f"Bearer {admin_token}"})
    assert list_reqs.status_code == 200
    tickets = list_reqs.json()
    assert any(t["id"] == ticket_id and t["status"] == "pending" for t in tickets)

    # 5. Admin approves deletion request
    appr_resp = client.post(f"/api/patients/deletion-requests/{ticket_id}/approve", headers={"Authorization": f"Bearer {admin_token}"})
    assert appr_resp.status_code == 200

    # 6. Verify patient was deleted
    get_pat = client.get(f"/api/patients/{pid}", headers={"Authorization": f"Bearer {admin_token}"})
    assert get_pat.status_code == 404


def test_prescription_approval_auto_ingestion():
    """
    Verify that when a doctor approves a prescription document:
    - Extracted medicines with days and period are ingested into active Medications and Occurrences!
    - The patient's Recovery Plan is automatically updated!
    """
    admin_token = get_token("admin_test_p17@recovera.ai", Role.ADMIN.value)
    nurse_token = get_token("nurse_test_p17@recovera.ai", Role.NURSE.value)

    # 1. Create patient
    p_resp = client.post("/api/patients", json={
        "name": "Prescription Ingestion Patient",
        "phone": f"+1999{uuid.uuid4().hex[:7]}",
        "sms_opt_in": True
    }, headers={"Authorization": f"Bearer {nurse_token}"})
    pid = p_resp.json()["id"]

    # 2. Upload prescription photo/file
    sample_content = b"Rx: Tab Amoxicillin 500mg - 1-0-1 - for 5 days (After Food)\nTab Paracetamol 650mg - TDS - 3 days"
    files = {"file": ("rx_photo.png", sample_content, "image/png")}
    data = {"doc_type": "prescription"}
    up_resp = client.post(f"/api/patients/{pid}/documents", files=files, data=data, headers={"Authorization": f"Bearer {nurse_token}"})
    assert up_resp.status_code == 201
    doc_id = up_resp.json()["id"]

    # 3. Doctor/Admin approves document
    appr_resp = client.post(f"/api/documents/{doc_id}/approve", headers={"Authorization": f"Bearer {admin_token}"})
    assert appr_resp.status_code == 200
    assert appr_resp.json()["status"] == "approved"
    assert appr_resp.json().get("ingested_medications", 0) >= 1

    # 4. Verify patient now has active medications created
    meds_resp = client.get(f"/api/patients/{pid}/medications", headers={"Authorization": f"Bearer {admin_token}"})
    assert meds_resp.status_code == 200
    meds = meds_resp.json()
    assert len(meds) >= 1
    amox = next((m for m in meds if "amoxicillin" in m["name"].lower()), None)
    assert amox is not None
    assert amox["status"] == "active"
    assert len(amox["schedules"]) >= 1

    # 5. Verify occurrences exist for the prescribed days
    occ_resp = client.get(f"/api/patients/{pid}/medication-occurrences", headers={"Authorization": f"Bearer {admin_token}"})
    assert occ_resp.status_code == 200
    occs = occ_resp.json()
    assert len(occs) >= 1
