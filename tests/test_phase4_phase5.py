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
from app.models.patient import Patient
from app.models.clinical_document import ClinicalDocument
from app.models.recovery_plan import RecoveryPlan
from app.models.medication import Medication, MedicationOccurrence, MedicationAdherence
from app.core.constants import Role, DocStatus, OccurrenceState

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_test_data():
    db = SessionLocal()
    try:
        doctor = db.query(User).filter_by(role=Role.DOCTOR.value).first()
        if not doctor:
            doctor = User(email="doctor.fixture.p4@hospital.org", password_hash="hash", name="Dr. Fixture P4", role=Role.DOCTOR.value, specialization="general_medicine")
            db.add(doctor)
            db.commit()

        nurse = db.query(User).filter_by(role=Role.NURSE.value).first()
        if not nurse:
            nurse = User(email="nurse.fixture.p4@hospital.org", password_hash="hash", name="Nurse Fixture P4", role=Role.NURSE.value)
            db.add(nurse)
            db.commit()

        patient_user = db.query(User).filter_by(role=Role.PATIENT.value).first()
        if not patient_user:
            patient_user = User(email="patient.fixture.p4@hospital.org", password_hash="hash", name="Patient Fixture P4", role=Role.PATIENT.value)
            db.add(patient_user)
            db.commit()

        patient = db.query(Patient).filter_by(user_id=patient_user.id).first()
        if not patient:
            patient = Patient(user_id=patient_user.id, patient_code="PTESTP4P5", active=True, assigned_doctor_id=doctor.id)
            db.add(patient)
            db.commit()

        doc_token = create_access_token(data={"sub": doctor.id, "email": doctor.email, "role": doctor.role})
        nurse_token = create_access_token(data={"sub": nurse.id, "email": nurse.email, "role": nurse.role})
        patient_token = create_access_token(data={"sub": patient_user.id, "email": patient_user.email, "role": Role.PATIENT.value})

        return {
            "doctor_id": doctor.id,
            "doctor_token": doc_token,
            "nurse_token": nurse_token,
            "patient_id": patient.id,
            "patient_user_id": patient_user.id,
            "patient_token": patient_token
        }
    finally:
        db.close()

def test_t07_recovery_plan_lifecycle_and_invalidation(setup_test_data):
    """
    T07: Recovery plan auto-generation, RBAC gate, and invalidation on source document change.
    """
    doc_token = setup_test_data["doctor_token"]
    nurse_token = setup_test_data["nurse_token"]
    pat_token = setup_test_data["patient_token"]
    pid = setup_test_data["patient_id"]

    db = SessionLocal()
    # Create an approved document for this patient to ground the plan
    source_doc = ClinicalDocument(
        patient_id=pid,
        doc_type="discharge_summary",
        filename="discharge_summary_cardiac.pdf",
        stored_path="backend/storage/dummy.pdf",
        status=DocStatus.APPROVED.value,
        extracted_text="Patient recovering from cardiac stent. Avoid lifting weights >5kg. Keep incision dry.",
        structured_data={"instructions": ["Avoid lifting weights >5kg", "Keep incision clean and dry"]},
        uploaded_by=setup_test_data["doctor_id"]
    )
    db.add(source_doc)
    db.commit()
    db.refresh(source_doc)
    doc_id = source_doc.id
    db.close()

    # 1. Generate plan draft from approved documents
    gen_res = client.post(
        f"/api/patients/{pid}/recovery-plan/generate",
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert gen_res.status_code == 201
    plan_data = gen_res.json()
    assert plan_data["status"] == DocStatus.DRAFT.value
    assert "instructions" in plan_data
    assert "warning_signs" in plan_data
    assert doc_id in plan_data["source_versions"]

    # 2. Safety gate: Patient CANNOT view plan while in DRAFT
    pat_view_draft = client.get(
        f"/api/patients/{pid}/recovery-plan",
        headers={"Authorization": f"Bearer {pat_token}"}
    )
    assert pat_view_draft.status_code == 404

    # 3. RBAC gate: Nurse CANNOT publish the recovery plan
    nurse_pub_res = client.post(
        f"/api/patients/{pid}/recovery-plan/publish",
        headers={"Authorization": f"Bearer {nurse_token}"}
    )
    assert nurse_pub_res.status_code == 403

    # 4. Doctor publishes the plan
    doc_pub_res = client.post(
        f"/api/patients/{pid}/recovery-plan/publish",
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert doc_pub_res.status_code == 200
    assert doc_pub_res.json()["status"] == DocStatus.PUBLISHED.value

    # 5. Patient CAN now view the published recovery plan
    pat_view_pub = client.get(
        f"/api/patients/{pid}/recovery-plan",
        headers={"Authorization": f"Bearer {pat_token}"}
    )
    assert pat_view_pub.status_code == 200
    assert pat_view_pub.json()["status"] == DocStatus.PUBLISHED.value

    # 6. Source change trigger: Doctor marks the approved source document as superseded
    sup_res = client.delete(
        f"/api/documents/{doc_id}",
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert sup_res.status_code == 200

    # 7. Recovery plan is automatically invalidated to NEEDS_REVIEW
    staff_view_res = client.get(
        f"/api/patients/{pid}/recovery-plan",
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert staff_view_res.status_code == 200
    assert staff_view_res.json()["status"] == DocStatus.NEEDS_REVIEW.value

    # 8. Clinical safety gate: Patient is again BLOCKED from viewing outdated plan
    pat_view_invalid = client.get(
        f"/api/patients/{pid}/recovery-plan",
        headers={"Authorization": f"Bearer {pat_token}"}
    )
    assert pat_view_invalid.status_code == 404

def test_t08_medication_occurrence_generation(setup_test_data):
    """
    T08: Medication occurrence generation
    - 3-day course × 2 sessions = exactly 6 occurrences
    - 5-day course × 3 sessions = exactly 15 occurrences
    """
    doc_token = setup_test_data["doctor_token"]
    pid = setup_test_data["patient_id"]

    # Course A: 3-day course with 2 sessions (Morning 08:00, Night 20:00)
    day1 = dt.date(2026, 11, 1)
    day3 = dt.date(2026, 11, 3)
    course_a = {
        "name": "Azithromycin",
        "dose": "500mg",
        "route": "oral",
        "start_date": day1.isoformat(),
        "end_date": day3.isoformat(),
        "instructions": "Take after meals",
        "schedules": [
            {"label": "Morning", "time_hhmm": "08:00", "dose": "500mg", "food_relation": "after_food"},
            {"label": "Night", "time_hhmm": "20:00", "dose": "500mg", "food_relation": "after_food"}
        ]
    }

    res_a = client.post(
        f"/api/patients/{pid}/medications",
        json=course_a,
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert res_a.status_code == 201
    med_a_id = res_a.json()["id"]

    # Verify occurrence count in DB for Course A
    db = SessionLocal()
    occ_a_count = db.query(MedicationOccurrence).filter(MedicationOccurrence.medication_id == med_a_id).count()
    assert occ_a_count == 6, f"Expected 6 occurrences for 3-day course with 2 sessions, got {occ_a_count}"

    # Course B: 5-day course with 3 sessions (Breakfast 08:00, Lunch 13:00, Dinner 20:00)
    day5 = dt.date(2026, 11, 5)
    course_b = {
        "name": "Amoxicillin-Clavulanate",
        "dose": "625mg",
        "route": "oral",
        "start_date": day1.isoformat(),
        "end_date": day5.isoformat(),
        "instructions": "Complete full antibiotic course",
        "schedules": [
            {"label": "Breakfast", "time_hhmm": "08:00", "dose": "625mg", "food_relation": "after_food"},
            {"label": "Lunch", "time_hhmm": "13:00", "dose": "625mg", "food_relation": "after_food"},
            {"label": "Dinner", "time_hhmm": "20:00", "dose": "625mg", "food_relation": "after_food"}
        ]
    }

    res_b = client.post(
        f"/api/patients/{pid}/medications",
        json=course_b,
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert res_b.status_code == 201
    med_b_id = res_b.json()["id"]

    occ_b_count = db.query(MedicationOccurrence).filter(MedicationOccurrence.medication_id == med_b_id).count()
    db.close()
    assert occ_b_count == 15, f"Expected 15 occurrences for 5-day course with 3 sessions, got {occ_b_count}"

def test_t09_medication_discontinuation_and_history_preservation(setup_test_data):
    """
    T09: Discontinuation cancels future pending occurrences while preserving past recorded adherence.
    """
    doc_token = setup_test_data["doctor_token"]
    pid = setup_test_data["patient_id"]

    today = dt.date.today()
    future_date = today + dt.timedelta(days=4)

    med_payload = {
        "name": "Ibuprofen",
        "dose": "400mg",
        "start_date": today.isoformat(),
        "end_date": future_date.isoformat(),
        "schedules": [
            {"label": "Morning", "time_hhmm": "09:00", "dose": "400mg"},
            {"label": "Evening", "time_hhmm": "18:00", "dose": "400mg"}
        ]
    }
    create_res = client.post(
        f"/api/patients/{pid}/medications",
        json=med_payload,
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert create_res.status_code == 201
    med_id = create_res.json()["id"]

    db = SessionLocal()
    # Pick first occurrence and record intake as TAKEN
    first_occ = db.query(MedicationOccurrence).filter(MedicationOccurrence.medication_id == med_id).order_by(MedicationOccurrence.scheduled_at.asc()).first()
    assert first_occ is not None
    occ_id = first_occ.id
    db.close()

    # Record adherence for first dose
    adh_res = client.post(
        f"/api/medication-occurrences/{occ_id}/adherence",
        json={"state": "taken"},
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert adh_res.status_code == 200

    # Doctor discontinues medication
    disc_res = client.post(
        f"/api/medications/{med_id}/discontinue?reason=Stomach+discomfort",
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert disc_res.status_code == 200
    assert disc_res.json()["status"] == "discontinued"

    db = SessionLocal()
    # Verify the taken dose remains TAKEN and adherence record is preserved
    preserved_occ = db.get(MedicationOccurrence, occ_id)
    assert preserved_occ.state == OccurrenceState.TAKEN.value

    adh_record = db.query(MedicationAdherence).filter(MedicationAdherence.occurrence_id == occ_id).first()
    assert adh_record is not None
    assert adh_record.state == "taken"

    # Verify future pending occurrences were cancelled
    future_cancelled = db.query(MedicationOccurrence).filter(
        MedicationOccurrence.medication_id == med_id,
        MedicationOccurrence.id != occ_id
    ).all()

    for f_occ in future_cancelled:
        assert f_occ.state == OccurrenceState.CANCELLED.value
    db.close()

def test_t10_website_intake_tracker(setup_test_data):
    """
    T10: Website intake tracker recording Taken / Not Taken with optional reasons,
    persisting properly and returning through occurrence queries.
    """
    doc_token = setup_test_data["doctor_token"]
    pat_token = setup_test_data["patient_token"]
    pid = setup_test_data["patient_id"]

    today = dt.date.today()
    med_payload = {
        "name": "Metformin",
        "dose": "500mg",
        "start_date": today.isoformat(),
        "end_date": today.isoformat(),
        "schedules": [
            {"label": "Morning", "time_hhmm": "08:30", "dose": "500mg", "food_relation": "with_food"},
            {"label": "Night", "time_hhmm": "20:30", "dose": "500mg", "food_relation": "with_food"}
        ]
    }
    create_res = client.post(
        f"/api/patients/{pid}/medications",
        json=med_payload,
        headers={"Authorization": f"Bearer {doc_token}"}
    )
    assert create_res.status_code == 201
    created_med_id = create_res.json()["id"]

    # Query occurrences for patient
    occ_res = client.get(
        f"/api/patients/{pid}/medication-occurrences?date={today.isoformat()}",
        headers={"Authorization": f"Bearer {pat_token}"}
    )
    assert occ_res.status_code == 200
    occurrences = [o for o in occ_res.json() if o["medication_id"] == created_med_id]
    assert len(occurrences) == 2

    occ1 = occurrences[0]
    occ2 = occurrences[1]

    # Record occ1 as TAKEN
    adh1 = client.post(
        f"/api/medication-occurrences/{occ1['id']}/adherence",
        json={"state": "taken"},
        headers={"Authorization": f"Bearer {pat_token}"}
    )
    assert adh1.status_code == 200
    assert adh1.json()["state"] == "taken"

    # Record occ2 as NOT_TAKEN with reason "felt_unwell"
    adh2 = client.post(
        f"/api/medication-occurrences/{occ2['id']}/adherence",
        json={"state": "not_taken", "reason": "felt_unwell"},
        headers={"Authorization": f"Bearer {pat_token}"}
    )
    assert adh2.status_code == 200
    assert adh2.json()["state"] == "not_taken"
    assert adh2.json()["reason"] == "felt_unwell"

    # Re-query occurrences to simulate page reload and verify state persistence
    re_query = client.get(
        f"/api/patients/{pid}/medication-occurrences?date={today.isoformat()}",
        headers={"Authorization": f"Bearer {pat_token}"}
    )
    assert re_query.status_code == 200
    re_occs = {o["id"]: o for o in re_query.json()}

    assert re_occs[occ1["id"]]["state"] == "taken"
    assert re_occs[occ1["id"]]["adherence"]["state"] == "taken"

    assert re_occs[occ2["id"]]["state"] == "not_taken"
    assert re_occs[occ2["id"]]["adherence"]["state"] == "not_taken"
    assert re_occs[occ2["id"]]["adherence"]["reason"] == "felt_unwell"
