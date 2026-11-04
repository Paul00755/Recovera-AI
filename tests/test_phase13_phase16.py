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
from app.core.security import create_access_token
from app.models.user import User
from app.models.patient import Patient, MedicalHistory, PatientDoctorAssignment
from app.models.medication import Medication, MedicationOccurrence, MedicationSchedule, MedicationAdherence
from app.models.audit_log import AuditLog
from app.models.checkin import DailyCheckIn
from app.core.constants import Role
from app.services.interaction_service import check_drug_interactions
from app.core.audit import record_audit, verify_audit_log_chain

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_env():
    db = SessionLocal()
    try:
        # Admin
        admin = db.query(User).filter_by(role=Role.ADMIN.value).first()
        if not admin:
            admin = User(email="admin.phase13@hospital.org", password_hash="hash", name="Admin 13", role=Role.ADMIN.value)
            db.add(admin)
            db.commit()

        # Doctor 1 (Cardiology)
        doc1 = db.query(User).filter_by(email="doc1.phase13@hospital.org").first()
        if not doc1:
            doc1 = User(
                email="doc1.phase13@hospital.org",
                password_hash="hash",
                name="Dr. Cardio",
                role=Role.DOCTOR.value,
                specialization="cardiology",
                availability_status="online"
            )
            db.add(doc1)
            db.commit()

        # Doctor 2 (General Medicine)
        doc2 = db.query(User).filter_by(email="doc2.phase13@hospital.org").first()
        if not doc2:
            doc2 = User(
                email="doc2.phase13@hospital.org",
                password_hash="hash",
                name="Dr. General",
                role=Role.DOCTOR.value,
                specialization="general_medicine",
                availability_status="online"
            )
            db.add(doc2)
            db.commit()

        # Nurse
        nurse = db.query(User).filter_by(role=Role.NURSE.value).first()
        if not nurse:
            nurse = User(email="nurse.phase13@hospital.org", password_hash="hash", name="Nurse 13", role=Role.NURSE.value)
            db.add(nurse)
            db.commit()

        # Patient user & patient
        patient_user = db.query(User).filter_by(email="patient.phase13@hospital.org").first()
        if not patient_user:
            patient_user = User(email="patient.phase13@hospital.org", password_hash="hash", name="Patient 13", role=Role.PATIENT.value)
            db.add(patient_user)
            db.commit()

        patient = db.query(Patient).filter_by(user_id=patient_user.id).first()
        if not patient:
            patient = Patient(
                user_id=patient_user.id,
                patient_code="P13-TEST-001",
                active=True,
                assigned_doctor_id=doc1.id
            )
            db.add(patient)
            db.commit()

        # Medical history with CKD and Penicillin allergy
        history = db.query(MedicalHistory).filter_by(patient_id=patient.id).first()
        if not history:
            history = MedicalHistory(
                patient_id=patient.id,
                has_history=True,
                conditions=["Chronic Kidney Disease", "Hypertension"],
                notes="Severe anaphylactic allergy to Penicillin"
            )
            db.add(history)
            db.commit()

        admin_token = create_access_token(data={"sub": admin.id, "email": admin.email, "role": admin.role})
        doc1_token = create_access_token(data={"sub": doc1.id, "email": doc1.email, "role": doc1.role})
        nurse_token = create_access_token(data={"sub": nurse.id, "email": nurse.email, "role": nurse.role})

        return {
            "admin_token": admin_token,
            "doc1_token": doc1_token,
            "nurse_token": nurse_token,
            "patient_id": patient.id,
            "doc1_id": doc1.id,
            "doc2_id": doc2.id
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Phase 13: ADE & Cross-Reactivity Interceptor Tests
# ---------------------------------------------------------------------------

def test_p13_interaction_service_unit():
    """Direct clinical test of DDI, condition contraindication, and allergy rules."""
    # 1. Warfarin + Aspirin DDI
    res1 = check_drug_interactions("Aspirin", active_medications=["Warfarin 5mg"])
    assert res1["has_interactions"] is True
    assert res1["max_severity"] == "contraindicated"
    assert res1["override_required"] is True

    # 2. NSAID + Chronic Kidney Disease contraindication
    res2 = check_drug_interactions("Ibuprofen 400mg", conditions=["Stage 3 Chronic Kidney Disease"])
    assert res2["has_interactions"] is True
    assert res2["max_severity"] == "contraindicated"

    # 3. Penicillin allergy cross-reactivity
    res3 = check_drug_interactions("Amoxicillin 500mg", allergies=["Penicillin"])
    assert res3["has_interactions"] is True
    assert res3["max_severity"] == "contraindicated"

    # 4. Safe drug - Acetaminophen with no interactions
    res4 = check_drug_interactions("Acetaminophen 500mg", active_medications=["Metformin"], conditions=["Hypertension"])
    assert res4["has_interactions"] is False
    assert res4["max_severity"] == "none"
    assert res4["override_required"] is False


def test_p13_check_interactions_endpoint(setup_test_env):
    """Test the POST /api/medications/check-interactions endpoint."""
    env = setup_test_env
    resp = client.post(
        "/api/medications/check-interactions",
        json={"patient_id": env["patient_id"], "medication_name": "Ibuprofen 400mg"},
        headers={"Authorization": f"Bearer {env['doc1_token']}"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_interactions"] is True
    assert data["override_required"] is True
    # Should detect CKD from patient's medical history
    assert any("Chronic Kidney Disease" in i["agent_2"] for i in data["interactions"])


def test_p13_prescribe_intercept_blocks_without_rationale(setup_test_env):
    """Test that prescribing a contraindicated drug without override rationale is blocked with HTTP 422."""
    env = setup_test_env
    today = dt.date.today()
    payload = {
        "name": "Ibuprofen 400mg",
        "route": "oral",
        "start_date": str(today),
        "end_date": str(today + dt.timedelta(days=5)),
        "instructions": "Take with food",
        "schedules": [{"label": "Morning", "time_hhmm": "08:00"}]
        # No clinical_override_rationale
    }
    resp = client.post(
        f"/api/patients/{env['patient_id']}/medications",
        json=payload,
        headers={"Authorization": f"Bearer {env['doc1_token']}"}
    )
    assert resp.status_code == 422
    data = resp.json()
    assert "override rationale" in data["detail"]["message"].lower()


def test_p13_prescribe_succeeds_with_override_rationale(setup_test_env):
    """Test that providing a clinical override rationale permits prescription and audits the override."""
    env = setup_test_env
    today = dt.date.today()
    payload = {
        "name": "Ibuprofen 400mg",
        "route": "oral",
        "start_date": str(today),
        "end_date": str(today + dt.timedelta(days=2)),
        "instructions": "Ultra short course for acute refractory migraine",
        "clinical_override_rationale": "Short 48h emergency course with close renal function and hydration monitoring.",
        "schedules": [{"label": "Morning", "time_hhmm": "08:00"}]
    }
    resp = client.post(
        f"/api/patients/{env['patient_id']}/medications",
        json=payload,
        headers={"Authorization": f"Bearer {env['doc1_token']}"}
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Ibuprofen 400mg"


# ---------------------------------------------------------------------------
# Phase 14: Dynamic Clinician Roster & Auto-Load Balancing Tests
# ---------------------------------------------------------------------------

def test_p14_doctor_load_stats_endpoint(setup_test_env):
    """Test GET /api/doctors/load-stats returns live capacity metrics."""
    env = setup_test_env
    resp = client.get(
        "/api/doctors/load-stats",
        headers={"Authorization": f"Bearer {env['nurse_token']}"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 2
    doc_entries = data["doctors"]
    assert all("load_score" in d and "active_patients_count" in d for d in doc_entries)


def test_p14_auto_assign_patient(setup_test_env):
    """Test POST /api/patients/{pid}/auto-assign intelligently chooses least loaded doctor."""
    env = setup_test_env
    resp = client.post(
        f"/api/patients/{env['patient_id']}/auto-assign",
        json={"specialization": "general_medicine"},
        headers={"Authorization": f"Bearer {env['nurse_token']}"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "successfully" in data["message"].lower()
    assert data["assigned_doctor"]["doctor_id"] is not None


# ---------------------------------------------------------------------------
# Phase 15: Regulatory Audit Trail & Cryptographic Verification Tests
# ---------------------------------------------------------------------------

def test_p15_audit_hash_chain_integrity(setup_test_env):
    """Test cryptographic SHA-256 hash chaining and verify-chain endpoint."""
    env = setup_test_env
    resp = client.get(
        "/api/audit/verify-chain",
        headers={"Authorization": f"Bearer {env['admin_token']}"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_valid"] is True
    assert data["algorithm"] == "SHA-256"
    assert data["verified_entries"] > 0


def test_p15_audit_tampering_detection(setup_test_env):
    """Test that modifying an audit record immediately invalidates the cryptographic verification."""
    db = SessionLocal()
    try:
        latest = db.query(AuditLog).order_by(AuditLog.created_at.desc()).first()
        if latest and latest.entry_hash:
            original_action = latest.action
            try:
                # Intentionally tamper with the record
                latest.action = "UNAUTHORIZED_TAMPERED_ACTION"
                db.commit()

                res = verify_audit_log_chain(db)
                assert res["is_valid"] is False
                assert "tampering detected" in res["error"].lower()
            finally:
                # Revert to maintain clean database state
                latest.action = original_action
                db.commit()
    finally:
        db.close()


def test_p15_audit_export_csv_and_json(setup_test_env):
    """Test export of regulatory audit trail in CSV and JSON formats."""
    env = setup_test_env
    # CSV
    resp_csv = client.get("/api/audit/export?format=csv", headers={"Authorization": f"Bearer {env['admin_token']}"})
    assert resp_csv.status_code == 200
    assert "entry_hash" in resp_csv.text

    # JSON
    resp_json = client.get("/api/audit/export?format=json", headers={"Authorization": f"Bearer {env['admin_token']}"})
    assert resp_json.status_code == 200
    data = resp_json.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "entry_hash" in data[-1]


# ---------------------------------------------------------------------------
# Phase 16: Offline-First Field Clinic Synchronization Tests
# ---------------------------------------------------------------------------

def test_p16_offline_batch_sync(setup_test_env):
    """Test atomic synchronization of offline-collected check-ins and medication adherences."""
    env = setup_test_env
    client_now = dt.datetime.utcnow()

    # Create an active medication occurrence to test adherence sync
    db = SessionLocal()
    occ_id = None
    try:
        med = db.query(Medication).filter_by(patient_id=env["patient_id"]).first()
        if med:
            sched = db.query(MedicationSchedule).filter_by(medication_id=med.id).first()
            if sched:
                occ = MedicationOccurrence(
                    medication_id=med.id,
                    schedule_id=sched.id,
                    patient_id=env["patient_id"],
                    scheduled_at=client_now,
                    state="pending"
                )
                db.add(occ)
                db.commit()
                occ_id = occ.id
    finally:
        db.close()

    payload = {
        "client_sync_id": "test-sync-uuid-001",
        "device_id": "field-tablet-rural-01",
        "synced_at": client_now.isoformat(),
        "checkins": [
            {
                "patient_id": env["patient_id"],
                "client_timestamp": client_now.isoformat(),
                "pain_level": 3,
                "mobility_score": 8,
                "temperature": 98.4,
                "blood_pressure": "118/76",
                "symptoms": ["mild stiffness"],
                "notes": "Visited at home in rural outreach clinic."
            }
        ],
        "adherences": [
            {
                "occurrence_id": occ_id,
                "state": "taken",
                "reason": "Observed by field nurse",
                "client_timestamp": client_now.isoformat()
            }
        ] if occ_id else []
    }

    resp = client.post(
        "/api/sync/offline-batch",
        json=payload,
        headers={"Authorization": f"Bearer {env['nurse_token']}"}
    )
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "synced"
    assert res_data["checkins_synced"] >= 1
    if occ_id:
        assert res_data["adherences_synced"] >= 1
