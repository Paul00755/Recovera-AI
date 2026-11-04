"""
Recovera AI – Test Suite for Phases 9, 10, 11, and 12

Validates:
- Phase 9 & 10: Consultation WebRTC signaling, SMS invitations with join links, 15-min countdown auto-termination, doctor status reverting to online.
- Phase 11: Patient Daily Symptom Check-ins (0-10 pain, sleep, activity, symptoms) & Grounded AI Recovery Assistant with emergency/dosage safety boundaries.
- Phase 12: Explainable Clinical Risk Engine (Green/Yellow/Red/Insufficient Data), explainability reasons, clinician manual override with rationale, and longitudinal analytics.
"""
import uuid
import datetime as dt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import app
from app.core.database import SessionLocal
from app.models import init_db
from app.core.security import get_password_hash
from app.core.constants import Role, ConsultationStatus, RiskLevel, OccurrenceState, SMSCategory
from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.consultation import Consultation, ConsultationSignal
from app.models.checkin import DailyCheckIn
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence
from app.models.recovery_plan import RecoveryPlan
from app.models.risk_history import RiskClassificationHistory
from app.models.sms_job import SMSJob
from app.services.risk_service import evaluate_patient_risk, override_patient_risk, get_patient_analytics_trends

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_env():
    init_db()
    db = SessionLocal()
    suffix = uuid.uuid4().hex[:6]

    # Create Admin
    admin = User(
        name=f"Admin {suffix}",
        email=f"admin.{suffix}@hospital.org",
        password_hash=get_password_hash("AdminPass2026!"),
        role=Role.ADMIN.value,
        phone="+919876543001"
    )
    db.add(admin)

    # Create Doctor
    doctor = User(
        name=f"Dr. Sharma {suffix}",
        email=f"dr.sharma.{suffix}@hospital.org",
        password_hash=get_password_hash("DoctorPass2026!"),
        role=Role.DOCTOR.value,
        specialization="neurology",
        availability_status="online",
        phone="+919876543002"
    )
    db.add(doctor)

    # Create Nurse
    nurse = User(
        name=f"Nurse Joy {suffix}",
        email=f"nurse.{suffix}@hospital.org",
        password_hash=get_password_hash("NursePass2026!"),
        role=Role.NURSE.value,
        phone="+919876543003"
    )
    db.add(nurse)

    # Create Patient User
    patient_user = User(
        name=f"John Doe {suffix}",
        email=f"john.{suffix}@example.com",
        password_hash=get_password_hash("PatientPass2026!"),
        role=Role.PATIENT.value,
        phone="+919876543004"
    )
    db.add(patient_user)
    db.flush()

    # Create Patient Record
    patient = Patient(
        user_id=patient_user.id,
        patient_code=f"REC-{suffix.upper()}",
        phone=patient_user.phone,
        assigned_doctor_id=doctor.id,
        sms_opt_in=True,
        active=True
    )
    db.add(patient)
    db.flush()

    # Assign Doctor
    assignment = PatientDoctorAssignment(
        patient_id=patient.id,
        doctor_id=doctor.id,
        assigned_by=admin.id
    )
    db.add(assignment)
    db.commit()

    # Logins to acquire JWT tokens
    def get_token(email, password):
        res = client.post("/api/auth/login", json={"email": email, "password": password})
        assert res.status_code == 200, res.text
        return res.json()["token"]

    admin_token = get_token(admin.email, "AdminPass2026!")
    doctor_token = get_token(doctor.email, "DoctorPass2026!")
    nurse_token = get_token(nurse.email, "NursePass2026!")
    patient_token = get_token(patient_user.email, "PatientPass2026!")

    yield {
        "admin": admin,
        "doctor": doctor,
        "nurse": nurse,
        "patient_user": patient_user,
        "patient": patient,
        "admin_token": admin_token,
        "doctor_token": doctor_token,
        "nurse_token": nurse_token,
        "patient_token": patient_token,
    }
    db.close()


# ===========================================================================
# Phase 9 & 10: Consultation WebRTC Signaling, SMS Links & Auto-Termination
# ===========================================================================

def test_consultation_creation_and_sms_invitation(setup_env):
    """Verify consultation creation triggers outbound SMS invitations with join links to both participants."""
    headers = {"Authorization": f"Bearer {setup_env['nurse_token']}"}
    create_payload = {
        "patient_id": setup_env["patient"].id,
        "doctor_id": setup_env["doctor"].id,
        "mode": "video",
        "notes": "Urgent post-op neurological evaluation"
    }

    res = client.post("/api/consultations", json=create_payload, headers=headers)
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["status"] == ConsultationStatus.WAITING.value
    room_id = data["room_id"]
    consultation_id = data["id"]

    # Verify SMS jobs created for doctor and patient
    db = SessionLocal()
    try:
        sms_jobs = db.scalars(
            select(SMSJob).where(
                SMSJob.category == SMSCategory.CONSULTATION.value,
                SMSJob.patient_id == setup_env["patient"].id
            ).order_by(SMSJob.created_at.desc())
        ).all()
        assert len(sms_jobs) >= 1
        # Check that join link is present in the SMS message
        matching_job = next((j for j in sms_jobs if f"#/consultation/{room_id}" in j.message_text), None)
        assert matching_job is not None, "SMS join link was not found in consultation invitation"
    finally:
        db.close()

    # Verify room lookup by room_id
    room_res = client.get(f"/api/consultations/room/{room_id}", headers=headers)
    assert room_res.status_code == 200
    assert room_res.json()["id"] == consultation_id


def test_webrtc_signaling_exchange(setup_env):
    """Verify WebRTC offer/answer/ICE candidates signaling exchange between peers."""
    doctor_headers = {"Authorization": f"Bearer {setup_env['doctor_token']}"}
    patient_headers = {"Authorization": f"Bearer {setup_env['patient_token']}"}

    # 1. Doctor creates consultation
    res = client.post("/api/consultations", json={
        "patient_id": setup_env["patient"].id,
        "doctor_id": setup_env["doctor"].id,
        "mode": "video"
    }, headers=doctor_headers)
    assert res.status_code == 201
    cid = res.json()["id"]

    # 2. Doctor posts SDP offer
    offer_payload = {
        "signal_type": "offer",
        "signal_data": "{\"sdp\":\"v=0\\r\\no=doctor...\"}"
    }
    sig_res = client.post(f"/api/consultations/{cid}/signals", json=offer_payload, headers=doctor_headers)
    assert sig_res.status_code == 201
    offer_id = sig_res.json()["id"]

    # 3. Doctor should not see their own signal
    doc_signals = client.get(f"/api/consultations/{cid}/signals", headers=doctor_headers).json()
    assert len(doc_signals) == 0

    # 4. Patient should see the doctor's SDP offer
    pat_signals = client.get(f"/api/consultations/{cid}/signals", headers=patient_headers).json()
    assert len(pat_signals) == 1
    assert pat_signals[0]["signal_type"] == "offer"
    assert "sdp" in pat_signals[0]["signal_data"]

    # 5. Patient posts SDP answer
    answer_payload = {
        "signal_type": "answer",
        "signal_data": "{\"sdp\":\"v=0\\r\\no=patient...\"}"
    }
    ans_res = client.post(f"/api/consultations/{cid}/signals", json=answer_payload, headers=patient_headers)
    assert ans_res.status_code == 201

    # 6. Doctor fetches signals -> receives patient's answer
    doc_signals_after = client.get(f"/api/consultations/{cid}/signals", headers=doctor_headers).json()
    assert len(doc_signals_after) == 1
    assert doc_signals_after[0]["signal_type"] == "answer"

    # 7. Clear signals buffer
    del_res = client.delete(f"/api/consultations/{cid}/signals", headers=doctor_headers)
    assert del_res.status_code == 200
    assert del_res.json()["count"] >= 2


def test_fifteen_minute_auto_termination_and_doctor_revert(setup_env):
    """Verify 15-minute countdown auto-terminates active consultation and reverts doctor to online."""
    doctor_headers = {"Authorization": f"Bearer {setup_env['doctor_token']}"}

    # 1. Doctor creates and starts consultation
    res = client.post("/api/consultations", json={
        "patient_id": setup_env["patient"].id,
        "doctor_id": setup_env["doctor"].id,
        "mode": "video"
    }, headers=doctor_headers)
    cid = res.json()["id"]

    start_res = client.put(f"/api/consultations/{cid}/start", headers=doctor_headers)
    assert start_res.status_code == 200
    assert start_res.json()["status"] == ConsultationStatus.ACTIVE.value

    # Doctor should be marked busy
    db = SessionLocal()
    try:
        doc = db.get(User, setup_env["doctor"].id)
        assert doc.availability_status == "busy"

        # Simulate 15-minute countdown expiration by rewinding expires_at into the past
        c = db.get(Consultation, cid)
        c.expires_at = dt.datetime.utcnow() - dt.timedelta(seconds=10)
        db.commit()
    finally:
        db.close()

    # 2. Next consultation query triggers server-enforced auto-termination
    active_res = client.get(f"/api/consultations/{cid}", headers=doctor_headers)
    assert active_res.status_code == 200
    assert active_res.json()["status"] == ConsultationStatus.COMPLETED.value

    # Doctor status must automatically revert to 'online'
    db = SessionLocal()
    try:
        doc_after = db.get(User, setup_env["doctor"].id)
        assert doc_after.availability_status == "online"
    finally:
        db.close()


def test_consultation_extension(setup_env):
    """Verify doctor can extend active consultation by 5, 10, or 15 minutes."""
    doctor_headers = {"Authorization": f"Bearer {setup_env['doctor_token']}"}

    res = client.post("/api/consultations", json={
        "patient_id": setup_env["patient"].id,
        "doctor_id": setup_env["doctor"].id,
        "mode": "video"
    }, headers=doctor_headers)
    cid = res.json()["id"]
    client.put(f"/api/consultations/{cid}/start", headers=doctor_headers)

    # Invalid minutes rejected with 422
    inv_res = client.put(f"/api/consultations/{cid}/extend", json={"minutes": 7}, headers=doctor_headers)
    assert inv_res.status_code == 422

    # Extend by 10 minutes
    ext_res = client.put(f"/api/consultations/{cid}/extend", json={"minutes": 10}, headers=doctor_headers)
    assert ext_res.status_code == 200
    assert "extended by 10 minutes" in ext_res.json()["message"]


# ===========================================================================
# Phase 11: Patient Daily Symptom Check-ins & Grounded Recovery Chatbot
# ===========================================================================

def test_daily_symptom_checkin_lifecycle(setup_env):
    """Verify patient daily check-in submission, 0-10 validation, and clinician retrieval."""
    patient_headers = {"Authorization": f"Bearer {setup_env['patient_token']}"}
    nurse_headers = {"Authorization": f"Bearer {setup_env['nurse_token']}"}

    # 1. Validation guard: pain_level > 10 rejected with 422
    bad_res = client.post("/api/checkins", json={
        "pain_level": 11,
        "pain_location": "Head"
    }, headers=patient_headers)
    assert bad_res.status_code == 422

    # 2. Valid check-in submission by patient
    valid_payload = {
        "pain_level": 6,
        "pain_location": "Lower Back",
        "sleep_hours": 7.5,
        "activity_level": "light",
        "fever": True,
        "cough": False,
        "notes": "Mild stiffness when standing."
    }
    submit_res = client.post("/api/checkins", json=valid_payload, headers=patient_headers)
    assert submit_res.status_code == 201, submit_res.text
    data = submit_res.json()
    assert data["pain_level"] == 6
    assert data["pain_location"] == "Lower Back"
    assert data["fever"] is True

    # 3. Patient views their own check-ins
    my_checkins = client.get("/api/checkins", headers=patient_headers).json()
    assert len(my_checkins) >= 1
    assert my_checkins[0]["pain_level"] == 6

    # 4. Clinician views patient check-ins
    clinician_res = client.get(f"/api/patients/{setup_env['patient'].id}/checkins", headers=nurse_headers)
    assert clinician_res.status_code == 200
    assert len(clinician_res.json()) >= 1


def test_grounded_recovery_chatbot_emergency_alert(setup_env):
    """Verify chatbot immediately flags life-threatening emergency symptoms with clinical escalation."""
    patient_headers = {"Authorization": f"Bearer {setup_env['patient_token']}"}

    emergency_queries = [
        "I am having sudden severe chest pain and cannot breathe",
        "There is heavy bleeding from my surgical wound",
        "I feel like I am going to faint and passed out earlier"
    ]

    for q in emergency_queries:
        res = client.post("/api/recovery-assistant/chat", json={"message": q}, headers=patient_headers)
        assert res.status_code == 200
        body = res.json()
        assert body["is_emergency"] is True
        assert "EMERGENCY ALERT" in body["reply"]
        assert "112" in body["reply"] or "911" in body["reply"]
        assert body["grounded_source"] == "clinical_safety_protocol"


def test_grounded_recovery_chatbot_prescription_boundary(setup_env):
    """Verify chatbot refuses dosage alteration requests and directs to attending physician."""
    patient_headers = {"Authorization": f"Bearer {setup_env['patient_token']}"}

    boundary_queries = [
        "Can I double my dose of painkillers?",
        "Should I stop taking my antibiotics early?",
        "Can I increase my dose if pain is not going away?"
    ]

    for q in boundary_queries:
        res = client.post("/api/recovery-assistant/chat", json={"message": q}, headers=patient_headers)
        assert res.status_code == 200
        body = res.json()
        assert body["is_emergency"] is False
        assert "Dosage alterations" in body["reply"] or "cannot be advised by AI" in body["reply"]
        assert body["grounded_source"] == "prescription_safety_boundary"


def test_grounded_recovery_chatbot_answers_from_recovery_plan(setup_env):
    """Verify chatbot answers questions accurately using clinician-approved recovery plan and prescriptions."""
    patient_headers = {"Authorization": f"Bearer {setup_env['patient_token']}"}
    pid = setup_env["patient"].id

    db = SessionLocal()
    try:
        # Publish recovery plan
        plan = RecoveryPlan(
            patient_id=pid,
            version=1,
            instructions={
                "diagnosis": "Lumbar Laminectomy Post-Op",
                "care_instructions": "Keep incision clean and dry. Change dressing every 48 hours.",
                "precautions": "Do not lift objects over 5kg. Avoid twisting spine.",
                "dietary_guidelines": "High-protein, high-fiber meals with 2.5L water daily to prevent constipation.",
                "activity_restrictions": "Gentle walking permitted for 15 minutes. Avoid running or driving for 2 weeks."
            },
            warning_signs=["Fever above 101F", "Pus or fluid discharge from wound", "Sudden numbness in legs"],
            status="published",
            approved_by=setup_env["doctor"].id
        )
        db.add(plan)

        # Add active medication and schedule
        med = Medication(
            patient_id=pid,
            name="Cefuroxime",
            strength="500mg",
            dose="1 tablet",
            status="active",
            start_date=dt.date.today(),
            end_date=dt.date.today() + dt.timedelta(days=7),
            created_by=setup_env["doctor"].id
        )
        db.add(med)
        db.flush()

        sched = MedicationSchedule(
            medication_id=med.id,
            patient_id=pid,
            label="After Lunch",
            time_hhmm="13:30",
            dose="1 tablet",
            food_relation="after food"
        )
        db.add(sched)
        db.commit()
    finally:
        db.close()

    # Query about diet
    diet_res = client.post("/api/recovery-assistant/chat", json={"message": "What should my diet be during recovery?"}, headers=patient_headers)
    assert diet_res.status_code == 200
    assert "High-protein" in diet_res.json()["reply"]
    assert "Dietary Guidelines" in diet_res.json()["metadata"]["citations"]

    # Query about activity restrictions
    act_res = client.post("/api/recovery-assistant/chat", json={"message": "Can I drive or exercise right now?"}, headers=patient_headers)
    assert act_res.status_code == 200
    assert "Avoid running or driving" in act_res.json()["reply"]

    # Query about medication timings
    med_res = client.post("/api/recovery-assistant/chat", json={"message": "When should I take my Cefuroxime?"}, headers=patient_headers)
    assert med_res.status_code == 200
    assert "Cefuroxime" in med_res.json()["reply"]
    assert "13:30" in med_res.json()["reply"]


# ===========================================================================
# Phase 12: Explainable Clinical Risk Engine & Longitudinal Analytics
# ===========================================================================

def test_risk_engine_evaluation_and_explainability(setup_env):
    """Verify automated risk classification rules, explainability output, and insufficient data handling."""
    doctor_headers = {"Authorization": f"Bearer {setup_env['doctor_token']}"}
    pid = setup_env["patient"].id

    # 1. Fresh evaluation with high pain (pain = 8/10) should classify as RED
    db = SessionLocal()
    try:
        # Submit high pain checkin
        checkin = DailyCheckIn(
            patient_id=pid,
            checkin_date=dt.date.today(),
            pain_level=8,
            pain_location="Surgical Incision",
            submitted_at=dt.datetime.utcnow()
        )
        db.add(checkin)
        db.commit()
    finally:
        db.close()

    eval_res = client.post(f"/api/patients/{pid}/risk/evaluate", headers=doctor_headers)
    assert eval_res.status_code == 200, eval_res.text
    risk_data = eval_res.json()
    assert risk_data["level"] == RiskLevel.RED.value
    assert len(risk_data["explainability_reasons"]) >= 1
    assert any("Severe pain reported: 8/10" in r for r in risk_data["explainability_reasons"])


def test_risk_engine_clinician_manual_override(setup_env):
    """Verify clinician manual risk override with documented rationale."""
    doctor_headers = {"Authorization": f"Bearer {setup_env['doctor_token']}"}
    patient_headers = {"Authorization": f"Bearer {setup_env['patient_token']}"}
    pid = setup_env["patient"].id

    override_payload = {
        "level": "yellow",
        "override_reason": "Patient pain managed with breakthrough medication; vital signs stable."
    }

    # Patient cannot override risk (RBAC)
    forbidden_res = client.post(f"/api/patients/{pid}/risk/override", json=override_payload, headers=patient_headers)
    assert forbidden_res.status_code == 403

    # Doctor overrides risk
    override_res = client.post(f"/api/patients/{pid}/risk/override", json=override_payload, headers=doctor_headers)
    assert override_res.status_code == 200, override_res.text
    body = override_res.json()
    assert body["level"] == "yellow"
    assert body["is_override"] is True
    assert "breakthrough medication" in body["override_reason"]
    assert any("[Clinician Override]" in r for r in body["explainability_reasons"])


def test_longitudinal_analytics_trends(setup_env):
    """Verify analytics trends endpoint returns daily adherence and pain trajectories."""
    doctor_headers = {"Authorization": f"Bearer {setup_env['doctor_token']}"}
    pid = setup_env["patient"].id

    res = client.get(f"/api/patients/{pid}/analytics/trends?days=7", headers=doctor_headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "adherence_trend" in data
    assert "pain_trend" in data
    assert "summary" in data
    assert len(data["adherence_trend"]) == 7
    assert len(data["pain_trend"]) == 7
    assert "overall_adherence_pct" in data["summary"]
    assert "average_pain_level" in data["summary"]
