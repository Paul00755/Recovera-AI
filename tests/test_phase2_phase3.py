import os
import io
import sys
import pytest
from fastapi.testclient import TestClient

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.main import app
from app.core.database import SessionLocal
from app.core.security import create_access_token
from app.models.user import User
from app.models.patient import Patient
from app.models.clinical_document import ClinicalDocument
from app.core.constants import Role, DocStatus
from app.services.ocr_service import clean_ocr_text

client = TestClient(app)

@pytest.fixture
def auth_tokens():
    db = SessionLocal()
    try:
        doctor = db.query(User).filter_by(role=Role.DOCTOR.value).first()
        if not doctor:
            doctor = User(email="doctor.fixture.p2@hospital.org", password_hash="hash", name="Dr. Fixture", role=Role.DOCTOR.value, specialization="general_medicine")
            db.add(doctor)
            db.commit()

        nurse = db.query(User).filter_by(role=Role.NURSE.value).first()
        if not nurse:
            nurse = User(email="nurse.fixture.p2@hospital.org", password_hash="hash", name="Nurse Fixture", role=Role.NURSE.value)
            db.add(nurse)
            db.commit()

        patient_user = db.query(User).filter_by(role=Role.PATIENT.value).first()
        if not patient_user:
            patient_user = User(email="patient.fixture.p2@hospital.org", password_hash="hash", name="Patient Fixture", role=Role.PATIENT.value)
            db.add(patient_user)
            db.commit()

        patient = db.query(Patient).filter_by(user_id=patient_user.id).first()
        if not patient:
            patient = Patient(user_id=patient_user.id, patient_code="PTESTP2P3", active=True, assigned_doctor_id=doctor.id)
            db.add(patient)
            db.commit()
        
        doc_token = create_access_token({"sub": doctor.id, "role": doctor.role, "email": doctor.email})
        nurse_token = create_access_token({"sub": nurse.id, "role": nurse.role, "email": nurse.email})
        patient_token = create_access_token({"sub": patient_user.id, "role": patient_user.role, "email": patient_user.email})
        
        return {
            "doctor": doc_token,
            "nurse": nurse_token,
            "patient": patient_token,
            "patient_user": patient_user,
            "patient_record": patient
        }
    finally:
        db.close()

def test_ocr_noise_cleaning():
    """Verify clean_ocr_text cleans noise artifacts like (page 1) and repeated headers."""
    dirty_text = "Hospital Discharge (page 1)\n\nPatient Name: John Doe\nPage 1 of 2\nTake Amoxicillin 500mg twice daily\nscanned by HP-ScanJet\n"
    cleaned = clean_ocr_text(dirty_text)
    assert "(page 1)" not in cleaned
    assert "Page 1 of 2" not in cleaned
    assert "scanned by HP-ScanJet" not in cleaned
    assert "Take Amoxicillin 500mg twice daily" in cleaned

def test_patient_search_and_directory(auth_tokens):
    """Verify clinicians can list and search patients."""
    resp = client.get("/api/patients", headers={"Authorization": f"Bearer {auth_tokens['doctor']}"})
    assert resp.status_code == 200
    patients = resp.json()
    assert len(patients) >= 1

    # Search by code or name
    first_code = patients[0]["patient_code"]
    search_resp = client.get(f"/api/patients?q={first_code}", headers={"Authorization": f"Bearer {auth_tokens['doctor']}"})
    assert search_resp.status_code == 200
    search_results = search_resp.json()
    assert len(search_results) >= 1
    assert search_results[0]["patient_code"] == first_code

def test_patient_registration_with_sms_opt_in(auth_tokens):
    """Verify patient registration creates patient_code, normalizes phone, and records opt-in."""
    import uuid
    unique_email = f"kavitha.{uuid.uuid4().hex[:6]}@example.com"
    payload = {
        "name": "Kavitha Raj",
        "dob": "1988-03-22",
        "phone": "+91 98401 23456",
        "email": unique_email,
        "preferred_language": "ta",
        "sms_opt_in": True,
        "has_medical_history": True,
        "medical_history": {
            "has_history": True,
            "conditions": [{"name": "Type 2 Diabetes", "duration": "3 years"}],
            "past_medications": [{"name": "Metformin 500mg", "dose": "500mg"}],
            "past_treatments": [],
            "past_hospitalizations": [],
            "notes": "Under regular clinical follow-up."
        }
    }
    resp = client.post("/api/patients", headers={"Authorization": f"Bearer {auth_tokens['nurse']}"}, json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Kavitha Raj"
    assert data["phone"] == "+919840123456"
    assert data["sms_opt_in"] is True
    assert data["patient_code"].startswith("P")

def test_document_upload_ocr_and_doctor_approval(auth_tokens):
    """Verify document upload creates draft, OCR extracts content, and doctor can approve."""
    # 1. Get patient ID
    p_resp = client.get("/api/patients", headers={"Authorization": f"Bearer {auth_tokens['doctor']}"})
    pid = p_resp.json()[0]["id"]

    # 2. Upload sample prescription text file as mock doc
    fake_prescription = b"Prescription (page 1)\nTab Amoxicillin 500mg twice daily after food for 5 days\n"
    files = {"file": ("prescription_sample.pdf", io.BytesIO(fake_prescription), "application/pdf")}
    data = {"doc_type": "prescription"}

    upload_resp = client.post(
        f"/api/patients/{pid}/documents",
        headers={"Authorization": f"Bearer {auth_tokens['nurse']}"},
        data=data,
        files=files
    )
    assert upload_resp.status_code == 201
    doc_data = upload_resp.json()
    assert doc_data["status"] == "draft"
    doc_id = doc_data["id"]

    # 3. Doctor reviews and approves document
    approve_resp = client.post(
        f"/api/documents/{doc_id}/approve",
        headers={"Authorization": f"Bearer {auth_tokens['doctor']}"}
    )
    assert approve_resp.status_code == 200
    assert approve_resp.json()["status"] == "approved"

def test_rejected_document_hidden_from_patient(auth_tokens):
    """Verify rejected documents are hidden from patient portal."""
    db = SessionLocal()
    try:
        # Find patient corresponding to patient_user
        patient = db.query(Patient).filter_by(user_id=auth_tokens["patient_user"].id).first()
        if not patient:
            pytest.skip("No patient linked to test patient user")

        # Upload doc for this patient
        fake_doc = b"Clinical notes (page 1)\nConfidential internal triage note\n"
        files = {"file": ("triage_note.pdf", io.BytesIO(fake_doc), "application/pdf")}
        data = {"doc_type": "recovery_instructions"}

        upload_resp = client.post(
            f"/api/patients/{patient.id}/documents",
            headers={"Authorization": f"Bearer {auth_tokens['nurse']}"},
            data=data,
            files=files
        )
        doc_id = upload_resp.json()["id"]

        # Doctor rejects the document
        reject_resp = client.post(
            f"/api/documents/{doc_id}/reject",
            headers={"Authorization": f"Bearer {auth_tokens['doctor']}"}
        )
        assert reject_resp.status_code == 200

        # Patient listings must NOT contain the rejected document
        pat_docs_resp = client.get(
            f"/api/patients/{patient.id}/documents",
            headers={"Authorization": f"Bearer {auth_tokens['patient']}"}
        )
        assert pat_docs_resp.status_code == 200
        pat_doc_ids = [d["id"] for d in pat_docs_resp.json()]
        assert doc_id not in pat_doc_ids

        # Patient direct access to rejected document must return 403 Forbidden
        direct_resp = client.get(
            f"/api/documents/{doc_id}",
            headers={"Authorization": f"Bearer {auth_tokens['patient']}"}
        )
        assert direct_resp.status_code == 403
    finally:
        db.close()
