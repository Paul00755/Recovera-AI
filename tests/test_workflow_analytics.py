import pytest
import datetime as dt
from fastapi.testclient import TestClient
from app.main import app
from app.core.database import SessionLocal
from app.core.security import create_access_token
from app.models.user import User
from app.models.patient import Patient

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_tokens():
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.role == "admin").first()
        doctor_user = db.query(User).filter(User.role == "doctor").first()
        nurse_user = db.query(User).filter(User.role == "nurse").first()
        patient_user = db.query(User).filter(User.role == "patient").first()

        tokens = {}
        if admin_user:
            tokens["admin"] = create_access_token({"sub": admin_user.id, "role": admin_user.role})
        if doctor_user:
            tokens["doctor"] = create_access_token({"sub": doctor_user.id, "role": doctor_user.role})
        if nurse_user:
            tokens["nurse"] = create_access_token({"sub": nurse_user.id, "role": nurse_user.role})
        if patient_user:
            tokens["patient"] = create_access_token({"sub": patient_user.id, "role": patient_user.role})
        return tokens
    finally:
        db.close()


def test_analytics_today_for_admin(setup_tokens):
    token = setup_tokens.get("admin")
    if not token:
        pytest.skip("Admin user not found")
    res = client.get("/api/analytics/workflow?timeline=today", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["timeline"] == "today"
    assert "kpis" in data
    assert "time_series" in data
    assert "charts" in data
    assert "role_metrics" in data
    assert data["role"] == "admin"


def test_analytics_timelines_for_doctor(setup_tokens):
    token = setup_tokens.get("doctor")
    if not token:
        pytest.skip("Doctor user not found")
    for tl in ["yesterday", "this_week", "this_month", "last_month"]:
        res = client.get(f"/api/analytics/workflow?timeline={tl}", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        data = res.json()
        assert data["timeline"] == tl
        assert "assigned_patients_count" in data["role_metrics"]


def test_analytics_custom_date(setup_tokens):
    token = setup_tokens.get("nurse") or setup_tokens.get("admin")
    today_str = dt.date.today().isoformat()
    res = client.get(
        f"/api/analytics/workflow?timeline=custom&start_date={today_str}&end_date={today_str}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["timeline"] == "custom"
    assert "time_series" in data
