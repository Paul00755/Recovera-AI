import pytest
import datetime as dt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.patient import Patient
from app.models.medication import Medication
from app.models.user import User
from app.core.constants import Role
from app.core.security import create_access_token
from app.services.handwriting_ml_service import register_medicine_in_library, COMMON_MEDICATIONS


def test_medication_library_deduplication():
    # 1. Existing medication check (e.g. Pan 40 or Paracetamol)
    added_dup = register_medicine_in_library(name="pan 40", generic="Pantoprazole Sodium")
    assert added_dup is False, "Duplicate medication must not be added to COMMON_MEDICATIONS"

    added_dup_case = register_medicine_in_library(name="PARACETAMOL", generic="Acetaminophen")
    assert added_dup_case is False, "Case-insensitive duplicate must not be added"

    # 2. Add brand new medication
    unique_name = f"TestMed_{int(dt.datetime.now(dt.timezone.utc).timestamp())}"
    added_new = register_medicine_in_library(name=unique_name, generic="Test Generic Substance", typical_doses=["100mg"])
    assert added_new is True, "New medication should be successfully registered in library"

    # 3. Re-adding the same one should now be detected as duplicate
    added_again = register_medicine_in_library(name=unique_name)
    assert added_again is False, "Re-adding newly added medicine must be detected as duplicate"


def test_checkin_7_day_cutoff_and_pretreatment_restrictions(client: TestClient, db_session: Session):
    # Retrieve test patient and their user
    patient = db_session.query(Patient).filter(Patient.active == True).first()
    if not patient:
        pytest.skip("No active patient found.")
    
    patient_user = db_session.get(User, patient.user_id)
    if not patient_user:
        pytest.skip("No patient user found.")

    token = create_access_token(data={"sub": patient_user.id, "role": Role.PATIENT.value})
    headers = {"Authorization": f"Bearer {token}"}

    today = dt.date.today()

    # Normalize existing medications for patient so start_date is within recent 7 days
    db_session.query(Medication).filter(Medication.patient_id == patient.id).update(
        {"start_date": today - dt.timedelta(days=4)}
    )
    db_session.commit()

    # Assign an active medication with start_date = 4 days ago
    med_start = today - dt.timedelta(days=4)
    med = Medication(
        patient_id=patient.id,
        name="Amoxicillin",
        generic_name="Amoxicillin Trihydrate",
        strength="500mg",
        dose="500mg",
        start_date=med_start,
        end_date=today + dt.timedelta(days=5),
        status="active",
        created_at=dt.datetime.now(dt.timezone.utc),
        created_by=patient_user.id
    )
    db_session.add(med)
    db_session.commit()

    # 1. Eligibility check returns earliest_medication_date
    elig_res = client.get("/api/checkins/eligibility", headers=headers)
    assert elig_res.status_code == 200
    elig_data = elig_res.json()
    assert elig_data["eligible"] is True
    assert elig_data["earliest_medication_date"] is not None

    # 2. Check-in submission older than 7 days should be blocked (400 Bad Request)
    old_date = today - dt.timedelta(days=8)
    res_old = client.post(
        "/api/checkins",
        headers=headers,
        json={"checkin_date": old_date.isoformat(), "pain_level": 3}
    )
    assert res_old.status_code == 400
    assert "older than 7 days" in res_old.json()["detail"]

    # 3. Check-in submission prior to medication start date should be blocked (400 Bad Request)
    # Use earliest_med_date from eligibility to test pre-treatment cutoff
    earliest_date_str = elig_data["earliest_medication_date"]
    earliest_date = dt.date.fromisoformat(earliest_date_str)
    pre_med_date = earliest_date - dt.timedelta(days=1)
    res_pre = client.post(
        "/api/checkins",
        headers=headers,
        json={"checkin_date": pre_med_date.isoformat(), "pain_level": 3}
    )
    assert res_pre.status_code == 400
    assert "prior to medication start date" in res_pre.json()["detail"]

    # 4. Valid missed check-in within window (e.g. 2 days ago) should succeed
    valid_missed_date = today - dt.timedelta(days=2)
    # Ensure no existing checkin for this test date
    from app.models.checkin import DailyCheckIn
    db_session.query(DailyCheckIn).filter(
        DailyCheckIn.patient_id == patient.id,
        DailyCheckIn.checkin_date == valid_missed_date
    ).delete()
    db_session.commit()

    res_valid = client.post(
        "/api/checkins",
        headers=headers,
        json={"checkin_date": valid_missed_date.isoformat(), "pain_level": 4, "pain_location": "knee"}
    )
    assert res_valid.status_code == 201
    checkin_data = res_valid.json()
    assert checkin_data["checkin_date"] == valid_missed_date.isoformat()

    # 5. Editing past checkin once calendar day has passed should be locked (403 Forbidden)
    res_lock = client.post(
        "/api/checkins",
        headers=headers,
        json={"checkin_date": valid_missed_date.isoformat(), "pain_level": 5}
    )
    assert res_lock.status_code == 403
    assert "Past daily check-ins are locked" in res_lock.json()["detail"]
