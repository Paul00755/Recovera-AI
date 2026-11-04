import os
import sys
import pytest
from fastapi.testclient import TestClient

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.main import app
from app.core.database import SessionLocal
from app.core.security import get_password_hash, create_access_token
from app.models.user import User
from app.models.patient import Patient
from app.core.constants import Role
from app.api.deps import enforce_patient_isolation
from fastapi import HTTPException

client = TestClient(app)

@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_login_success_and_jwt(db_session):
    """Verify login with valid credentials returns JWT token with sub and role."""
    # Ensure test doctor exists
    email = "test.doc.p1@hospital.org"
    password = "DocPassword123!"
    doc = db_session.query(User).filter_by(email=email).first()
    if not doc:
        doc = User(email=email, password_hash=get_password_hash(password), name="Test Doctor", role=Role.DOCTOR.value)
        db_session.add(doc)
        db_session.commit()

    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["user"]["email"] == email
    assert data["user"]["role"] == "doctor"

def test_login_invalid_credentials():
    """Verify login with invalid password returns 401."""
    resp = client.post("/api/auth/login", json={"email": "nonexistent@hospital.org", "password": "WrongPassword!"})
    assert resp.status_code == 401

def test_password_length_guard():
    """Verify passwords exceeding 72 bytes are rejected safely with 422."""
    long_pass = "X" * 80
    resp = client.post("/api/auth/login", json={"email": "test@hospital.org", "password": long_pass})
    assert resp.status_code == 422

def test_staff_registration_with_code(db_session):
    """Verify staff registration requires correct hospital code."""
    # Invalid code -> 403
    resp = client.post("/api/auth/register-staff", json={
        "email": "nurse.test.403@hospital.org",
        "password": "Password123!",
        "name": "Nurse Unauthorized",
        "phone": "+919876543210",
        "role": "nurse",
        "staff_code": "WRONG-CODE"
    })
    assert resp.status_code == 403

    # Valid code -> 201
    email = "nurse.test.201@hospital.org"
    # Delete if exists from previous run
    existing = db_session.query(User).filter_by(email=email).first()
    if existing:
        db_session.delete(existing)
        db_session.commit()

    resp = client.post("/api/auth/register-staff", json={
        "email": email,
        "password": "Password123!",
        "name": "Nurse Valid",
        "phone": "+919876543210",
        "role": "nurse",
        "staff_code": "RECOVERA-STAFF-2026"
    })
    assert resp.status_code == 201
    assert resp.json()["role"] == "nurse"

def test_rbac_access_control(db_session):
    """Verify non-admin roles cannot access admin endpoints."""
    # Create nurse token
    nurse = db_session.query(User).filter_by(role=Role.NURSE.value).first()
    nurse_token = create_access_token({"sub": nurse.id, "role": nurse.role, "email": nurse.email})

    # Nurse calling admin-only reset-tickets -> 403 Forbidden
    resp = client.get("/api/auth/reset-tickets", headers={"Authorization": f"Bearer {nurse_token}"})
    assert resp.status_code == 403

    # Admin calling reset-tickets -> 200 OK
    admin = db_session.query(User).filter_by(role=Role.ADMIN.value).first()
    admin_token = create_access_token({"sub": admin.id, "role": admin.role, "email": admin.email})
    resp = client.get("/api/auth/reset-tickets", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200

def test_patient_isolation_guard(db_session):
    """Verify patient users cannot access other patients' records."""
    # Find two patients
    pats = db_session.query(Patient).limit(2).all()
    if len(pats) >= 2:
        p1, p2 = pats[0], pats[1]
        user1 = db_session.get(User, p1.user_id)
        user2 = db_session.get(User, p2.user_id)

        # Patient 1 accessing Patient 2 -> 403
        with pytest.raises(HTTPException) as exc_info:
            enforce_patient_isolation(patient_id=p2.id, current_user=user1, db=db_session)
        assert exc_info.value.status_code == 403

        # Patient 1 accessing Patient 1 -> Success
        result = enforce_patient_isolation(patient_id=p1.id, current_user=user1, db=db_session)
        assert result.id == p1.id

        # Clinician accessing Patient 1 -> Success
        clinician = db_session.query(User).filter_by(role=Role.DOCTOR.value).first()
        result = enforce_patient_isolation(patient_id=p1.id, current_user=clinician, db=db_session)
        assert result.id == p1.id

def test_password_reset_ticket_lifecycle(db_session):
    """Verify full support-ticket workflow for password resets."""
    # 1. User submits forgot password
    email = "nurse.test.201@hospital.org"
    resp = client.post("/api/auth/forgot-password", json={"email": email, "reason": "Locked out of terminal"})
    assert resp.status_code == 200
    ticket_id = resp.json()["ticket_id"]

    # 2. Admin lists tickets and sees it
    admin = db_session.query(User).filter_by(role=Role.ADMIN.value).first()
    admin_token = create_access_token({"sub": admin.id, "role": admin.role, "email": admin.email})
    resp = client.get("/api/auth/reset-tickets", headers={"Authorization": f"Bearer {admin_token}"})
    assert resp.status_code == 200
    tickets = resp.json()
    assert any(t["id"] == ticket_id for t in tickets)

    # 3. Admin resolves ticket with new password
    new_pw = "NewSecurePassword2026!"
    resp = client.post(
        f"/api/auth/reset-tickets/{ticket_id}/resolve",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"action": "resolved", "new_password": new_pw, "notes": "Identity verified in person."}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "resolved"

    # 4. User can now login with new password
    resp = client.post("/api/auth/login", json={"email": email, "password": new_pw})
    assert resp.status_code == 200
    assert "token" in resp.json()
