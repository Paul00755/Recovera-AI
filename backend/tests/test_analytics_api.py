import uuid
import pytest
import datetime as dt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.appointment import Appointment
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence
from app.models.checkin import DailyCheckIn
from app.core.constants import Role, OccurrenceState, DocStatus, RiskLevel
from app.core.security import create_access_token, get_password_hash


def test_admin_dashboard_analytics(client: TestClient, db_session: Session):
    uid = uuid.uuid4().hex[:6]
    admin_user = User(
        name=f"Admin {uid}",
        email=f"admin_{uid}@recovera.ai",
        password_hash=get_password_hash("AdminPass123!"),
        role=Role.ADMIN.value,
        active=True
    )
    db_session.add(admin_user)
    db_session.commit()
    db_session.refresh(admin_user)

    try:
        token = create_access_token(data={"sub": admin_user.id, "role": Role.ADMIN.value})
        headers = {"Authorization": f"Bearer {token}"}

        response = client.get("/api/admin/dashboard", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "department_chart" in data
        assert "charts" in data
        assert "doctor_status_distribution" in data["charts"]
        assert "sms_health" in data
    finally:
        db_session.delete(admin_user)
        db_session.commit()


def test_doctor_workflow_analytics_isolation(client: TestClient, db_session: Session):
    uid = uuid.uuid4().hex[:6]
    # Create doctor 1 with 0 patients
    doc1 = User(
        name=f"Dr. Lonely {uid}",
        email=f"dr_lonely_{uid}@recovera.ai",
        password_hash=get_password_hash("Pass123!"),
        role=Role.DOCTOR.value,
        specialization="cardiology",
        active=True
    )
    # Create doctor 2 with 1 patient
    doc2 = User(
        name=f"Dr. Busy {uid}",
        email=f"dr_busy_{uid}@recovera.ai",
        password_hash=get_password_hash("Pass123!"),
        role=Role.DOCTOR.value,
        specialization="neurology",
        active=True
    )
    db_session.add_all([doc1, doc2])
    db_session.commit()
    db_session.refresh(doc1)
    db_session.refresh(doc2)

    pat_user = User(
        name=f"John Heart {uid}",
        email=f"john_heart_{uid}@recovera.ai",
        password_hash=get_password_hash("Pass123!"),
        role=Role.PATIENT.value,
        active=True
    )
    db_session.add(pat_user)
    db_session.commit()
    db_session.refresh(pat_user)

    pat = Patient(
        patient_code=f"PT-HEART-{uid}",
        user_id=pat_user.id,
        assigned_doctor_id=doc2.id,
        active=True
    )
    db_session.add(pat)
    db_session.commit()
    db_session.refresh(pat)

    try:
        # Doctor 1 queries analytics: should have 0 caseload and 0 occurrences
        token1 = create_access_token(data={"sub": doc1.id, "role": Role.DOCTOR.value})
        res1 = client.get("/api/analytics/workflow?timeline=today", headers={"Authorization": f"Bearer {token1}"})
        assert res1.status_code == 200
        d1 = res1.json()
        assert d1["role_metrics"]["caseload"] == 0
        assert d1["kpis"]["total_doses_scheduled"] == 0

        # Doctor 2 queries analytics: should have caseload 1
        token2 = create_access_token(data={"sub": doc2.id, "role": Role.DOCTOR.value})
        res2 = client.get("/api/analytics/workflow?timeline=today", headers={"Authorization": f"Bearer {token2}"})
        assert res2.status_code == 200
        d2 = res2.json()
        assert d2["role_metrics"]["caseload"] == 1
    finally:
        db_session.delete(pat)
        db_session.delete(pat_user)
        db_session.delete(doc1)
        db_session.delete(doc2)
        db_session.commit()


def test_patient_workflow_analytics(client: TestClient, db_session: Session):
    uid = uuid.uuid4().hex[:6]
    pat_user = User(
        name=f"Sarah Connor {uid}",
        email=f"sarah_connor_{uid}@recovera.ai",
        password_hash=get_password_hash("Pass123!"),
        role=Role.PATIENT.value,
        active=True
    )
    db_session.add(pat_user)
    db_session.commit()
    db_session.refresh(pat_user)

    pat = Patient(
        patient_code=f"PT-SARAH-{uid}",
        user_id=pat_user.id,
        active=True
    )
    db_session.add(pat)
    db_session.commit()
    db_session.refresh(pat)

    # Add a check-in for Sarah
    checkin = DailyCheckIn(
        patient_id=pat.id,
        checkin_date=dt.date.today(),
        pain_level=3,
        sleep_hours=8.0,
        fever=False,
        cough=False
    )
    db_session.add(checkin)
    db_session.commit()

    try:
        token = create_access_token(data={"sub": pat_user.id, "role": Role.PATIENT.value})
        res = client.get("/api/analytics/workflow?timeline=this_week", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        d = res.json()
        assert d["role"] == "patient"
        assert d["kpis"]["average_pain_level"] == 3.0
        assert d["kpis"]["checkin_count"] == 1
    finally:
        db_session.delete(checkin)
        db_session.delete(pat)
        db_session.delete(pat_user)
        db_session.commit()
