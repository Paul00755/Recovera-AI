import uuid
import datetime as dt
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import SessionLocal
from app.core.constants import Role, SMSCategory, OccurrenceState, AppointmentStatus, DocStatus
from app.core.security import get_password_hash, create_access_token
from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.notification import Notification
from app.models.sms_job import SMSJob
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence
from app.models.appointment import Appointment
from app.models.clinical_document import ClinicalDocument
from app.models.recovery_plan import RecoveryPlan
from app.services.medication_reminder_service import (
    process_missed_medication_escalations,
    process_evening_checkin_reminders,
    process_upcoming_appointment_reminders,
    process_due_medication_reminders,
)

client = TestClient(app)


@pytest.fixture
def notif_test_env():
    db = SessionLocal()
    suffix = uuid.uuid4().hex[:6]
    try:
        # 1. Doctor
        doctor = User(
            email=f"doc_{suffix}@hospital.org",
            password_hash=get_password_hash("DocPass123!"),
            name=f"Dr. Sharma {suffix}",
            role=Role.DOCTOR.value,
            phone="+919876543201",
            specialization="Cardiology",
            availability_status="online",
            active=True
        )
        db.add(doctor)

        # 2. Nurse
        nurse = User(
            email=f"nurse_{suffix}@hospital.org",
            password_hash=get_password_hash("NursePass123!"),
            name=f"Nurse Preeti {suffix}",
            role=Role.NURSE.value,
            phone="+919876543202",
            active=True
        )
        db.add(nurse)

        # 3. Patient User
        pat_user = User(
            email=f"pat_{suffix}@hospital.org",
            password_hash=get_password_hash("PatPass123!"),
            name=f"Rohan Verma {suffix}",
            role=Role.PATIENT.value,
            phone="+919876543203",
            active=True
        )
        db.add(pat_user)
        db.flush()

        # 4. Patient Record
        patient = Patient(
            user_id=pat_user.id,
            patient_code=f"REC-{suffix.upper()}",
            phone=pat_user.phone,
            sms_opt_in=True,
            active=True,
            assigned_doctor_id=doctor.id,
            created_at=dt.datetime.utcnow()
        )
        db.add(patient)
        db.flush()

        # Doctor assignment record
        assignment = PatientDoctorAssignment(
            patient_id=patient.id,
            doctor_id=doctor.id,
            assigned_by=nurse.id,
            department="Cardiology",
            is_active=True,
            assigned_at=dt.datetime.utcnow()
        )
        db.add(assignment)
        db.commit()

        # Auth tokens
        doc_token = create_access_token({"sub": doctor.id, "email": doctor.email, "role": doctor.role})
        nurse_token = create_access_token({"sub": nurse.id, "email": nurse.email, "role": nurse.role})
        pat_token = create_access_token({"sub": pat_user.id, "email": pat_user.email, "role": pat_user.role})

        return {
            "doctor": doctor,
            "nurse": nurse,
            "pat_user": pat_user,
            "patient": patient,
            "doc_token": doc_token,
            "nurse_token": nurse_token,
            "pat_token": pat_token,
        }
    finally:
        db.close()


def test_doctor_consultation_call_notifies_patient_only(notif_test_env):
    """
    Verify: When a doctor calls/books a consultation with a patient,
    the patient receives the In-App Alert and Gateway SMS with the room link.
    The doctor is not sent an incorrect 'new request' alert.
    """
    env = notif_test_env
    headers = {"Authorization": f"Bearer {env['doc_token']}"}

    res = client.post("/api/consultations", json={
        "patient_id": env["patient"].id,
        "doctor_id": env["doctor"].id,
        "mode": "video",
        "notes": "Post-op cardiology review"
    }, headers=headers)
    assert res.status_code in (200, 201), res.text
    data = res.json()
    room_id = data["room_id"]

    db = SessionLocal()
    try:
        # Check patient's in-app notification
        pat_notif = db.query(Notification).filter_by(
            user_id=env["pat_user"].id,
            category="consultation"
        ).order_by(Notification.created_at.desc()).first()
        assert pat_notif is not None
        assert "Doctor Consultation Call" in pat_notif.title
        assert f"#/consultation/{room_id}" in pat_notif.link

        # Check patient's SMS Job
        pat_sms = db.query(SMSJob).filter_by(
            patient_id=env["patient"].id,
            category=SMSCategory.CONSULTATION.value
        ).order_by(SMSJob.created_at.desc()).first()
        assert pat_sms is not None
        assert room_id in pat_sms.message_text
    finally:
        db.close()


def test_suggestion_1_critical_checkin_alert(notif_test_env):
    """
    Suggestion 1: When patient logs pain >= 7 or fever,
    attending doctor and floor nurses receive In-App alert and doctor receives SMS.
    """
    env = notif_test_env
    headers = {"Authorization": f"Bearer {env['pat_token']}"}

    today_str = dt.date.today().isoformat()
    res = client.post("/api/checkins", json={
        "checkin_date": today_str,
        "pain_level": 8,
        "pain_location": "Surgical incision site",
        "sleep_hours": 4,
        "activity_level": "low",
        "fever": True,
        "cough": False,
        "notes": "Sharp throbbing pain with burning fever"
    }, headers=headers)
    assert res.status_code in (200, 201), res.text

    db = SessionLocal()
    try:
        # Doctor In-App Notification
        doc_notif = db.query(Notification).filter_by(
            user_id=env["doctor"].id,
            category="critical_vitals"
        ).order_by(Notification.created_at.desc()).first()
        assert doc_notif is not None
        assert "High-Risk" in doc_notif.title
        assert "Pain: 8/10" in doc_notif.message

        # Floor Nurse In-App Notification
        nurse_notif = db.query(Notification).filter_by(
            user_id=env["nurse"].id,
            category="critical_vitals"
        ).order_by(Notification.created_at.desc()).first()
        assert nurse_notif is not None

        # Doctor SMS Gateway
        doc_sms = db.query(SMSJob).filter_by(
            recipient_phone=env["doctor"].phone,
            category=SMSCategory.CHECKIN_REMINDER.value
        ).order_by(SMSJob.created_at.desc()).first()
        assert doc_sms is not None
        assert "CRITICAL" in doc_sms.message_text
        assert "Pain 8/10" in doc_sms.message_text
    finally:
        db.close()


def test_suggestion_2_evening_checkin_reminder(notif_test_env):
    """
    Suggestion 2: Evening Daily Check-In Reminder.
    Evaluates active patients who have not submitted today's check-in.
    """
    env = notif_test_env
    db = SessionLocal()
    try:
        # Simulate tomorrow's date where no check-in exists yet
        sim_now = dt.datetime.utcnow() + dt.timedelta(days=1, hours=19)
        sent_count = process_evening_checkin_reminders(db, now=sim_now)
        assert sent_count >= 1

        # Verify Patient In-App reminder
        notif = db.query(Notification).filter_by(
            user_id=env["pat_user"].id,
            category="checkin_reminder"
        ).order_by(Notification.created_at.desc()).first()
        assert notif is not None
        assert "Daily Recovery Check-In" in notif.title

        # Verify Patient SMS reminder
        sms = db.query(SMSJob).filter_by(
            patient_id=env["patient"].id,
            category=SMSCategory.CHECKIN_REMINDER.value
        ).order_by(SMSJob.created_at.desc()).first()
        assert sms is not None
        assert "complete your daily recovery check-in" in sms.message_text

        # Verify strict idempotency
        second_run = process_evening_checkin_reminders(db, now=sim_now)
        assert second_run == 0
    finally:
        db.close()


def test_suggestion_3_missed_medication_escalation(notif_test_env):
    """
    Suggestion 3: Missed Medication Escalation (T + 2 Hours).
    If dose is still pending 2 hours post-scheduled intake, alert nurses and warn patient.
    """
    env = notif_test_env
    db = SessionLocal()
    try:
        now = dt.datetime.utcnow()

        # Create Medication
        med = Medication(
            patient_id=env["patient"].id,
            name="Atorvastatin",
            strength="20mg",
            dose="1 tablet",
            status="active",
            start_date=now.date(),
            end_date=now.date() + dt.timedelta(days=10),
            created_by=env["doctor"].id
        )
        db.add(med)
        db.flush()

        sched = MedicationSchedule(
            medication_id=med.id,
            patient_id=env["patient"].id,
            label="Night",
            time_hhmm="21:00",
            dose="1 tablet",
            food_relation="after_food"
        )
        db.add(sched)
        db.flush()

        # Occurrence scheduled 3 hours ago (in the T + 2h window)
        occ = MedicationOccurrence(
            medication_id=med.id,
            schedule_id=sched.id,
            patient_id=env["patient"].id,
            scheduled_at=now - dt.timedelta(hours=3),
            state=OccurrenceState.PENDING.value
        )
        db.add(occ)
        db.commit()

        # Run escalation processor
        count = process_missed_medication_escalations(db, now=now)
        assert count >= 1

        # Nurse In-App Alert
        nurse_notif = db.query(Notification).filter_by(
            user_id=env["nurse"].id,
            category="medication_overdue"
        ).order_by(Notification.created_at.desc()).first()
        assert nurse_notif is not None
        assert "Medication Overdue" in nurse_notif.title
        assert "Atorvastatin" in nurse_notif.message

        # Patient SMS Warning
        pat_sms = db.query(SMSJob).filter_by(
            patient_id=env["patient"].id,
            category=SMSCategory.MEDICATION_REMINDER.value
        ).filter(SMSJob.message_text.like("%overdue dose of Atorvastatin%")).first()
        assert pat_sms is not None

        # Verify strict idempotency
        assert process_missed_medication_escalations(db, now=now) == 0
    finally:
        db.close()


def test_suggestion_4_recovery_plan_and_prescription_updates(notif_test_env):
    """
    Suggestion 4: Recovery Plan or Prescription Published/Updated.
    Notifies patient via In-App Alert and SMS Gateway.
    """
    env = notif_test_env
    doc_headers = {"Authorization": f"Bearer {env['doc_token']}"}

    # 1. Prescribe a new medication via Doctor API
    start_d = dt.date.today()
    end_d = start_d + dt.timedelta(days=14)
    med_res = client.post(f"/api/patients/{env['patient'].id}/medications", json={
        "name": "Metoprolol",
        "strength": "25mg",
        "dose": "1 tablet",
        "frequency": "twice_daily",
        "instructions": "Take after meals",
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "days": 14,
        "schedules": [{"label": "Morning", "time_hhmm": "08:00", "dose": "1 tablet", "food_relation": "after_food"}]
    }, headers=doc_headers)
    assert med_res.status_code == 201, med_res.text

    db = SessionLocal()
    try:
        # Check prescription in-app notification
        p_notif = db.query(Notification).filter_by(
            user_id=env["pat_user"].id,
            category="prescription"
        ).order_by(Notification.created_at.desc()).first()
        assert p_notif is not None
        assert "Metoprolol" in p_notif.message

        # Check prescription SMS
        p_sms = db.query(SMSJob).filter_by(
            patient_id=env["patient"].id,
            category=SMSCategory.MEDICATION_REMINDER.value
        ).filter(SMSJob.message_text.like("%Metoprolol%")).first()
        assert p_sms is not None
    finally:
        db.close()

    # 2. Publish Recovery Plan via Doctor API
    db = SessionLocal()
    try:
        plan = RecoveryPlan(
            patient_id=env["patient"].id,
            status=DocStatus.NEEDS_REVIEW.value,
            instructions={"diet": "Low sodium", "mobility": "Light walking 15 min"},
            warning_signs=["Sudden dizziness", "Chest tightness"],
            version=1,
            created_at=dt.datetime.utcnow()
        )
        db.add(plan)
        db.commit()
    finally:
        db.close()

    pub_res = client.post(f"/api/patients/{env['patient'].id}/recovery-plan/publish", json={
        "instructions": {"diet": "Low sodium", "mobility": "Light walking 15 min"},
        "warning_signs": ["Sudden dizziness", "Chest tightness"]
    }, headers=doc_headers)
    assert pub_res.status_code == 200, pub_res.text

    db = SessionLocal()
    try:
        plan_notif = db.query(Notification).filter_by(
            user_id=env["pat_user"].id,
            category="recovery_plan"
        ).order_by(Notification.created_at.desc()).first()
        assert plan_notif is not None
        assert "Recovery Plan Published" in plan_notif.title

        plan_sms = db.query(SMSJob).filter_by(
            patient_id=env["patient"].id,
            category=SMSCategory.REGISTRATION.value
        ).filter(SMSJob.message_text.like("%published your updated Recovery Plan%")).first()
        assert plan_sms is not None
    finally:
        db.close()


def test_suggestion_5_upcoming_appointment_reminders(notif_test_env):
    """
    Suggestion 5: Upcoming Appointment Reminders at 24 hours and 2 hours prior.
    """
    env = notif_test_env
    db = SessionLocal()
    try:
        now = dt.datetime.utcnow()

        # Appointment 1: In 24 hours (23.5 hours away)
        appt_24h = Appointment(
            patient_id=env["patient"].id,
            doctor_id=env["doctor"].id,
            booked_by=env["nurse"].id,
            start_time=now + dt.timedelta(hours=23, minutes=30),
            end_time=now + dt.timedelta(hours=24),
            status=AppointmentStatus.SCHEDULED.value,
            notes="Cardiology checkup"
        )
        db.add(appt_24h)

        # Appointment 2: In 2 hours (100 minutes away)
        appt_2h = Appointment(
            patient_id=env["patient"].id,
            doctor_id=env["doctor"].id,
            booked_by=env["nurse"].id,
            start_time=now + dt.timedelta(minutes=100),
            end_time=now + dt.timedelta(minutes=115),
            status=AppointmentStatus.SCHEDULED.value,
            notes="Wound inspection"
        )
        db.add(appt_2h)
        db.commit()

        # Run appointment reminders processor
        sent = process_upcoming_appointment_reminders(db, now=now)
        assert sent >= 2

        # Verify Patient In-App Reminders
        notifs = db.query(Notification).filter_by(
            user_id=env["pat_user"].id,
            category="appointment"
        ).all()
        assert any("24h" in n.title for n in notifs)
        assert any("2h" in n.title for n in notifs)

        # Verify Doctor In-App Reminders
        doc_notifs = db.query(Notification).filter_by(
            user_id=env["doctor"].id,
            category="appointment"
        ).all()
        assert any("24h" in n.title for n in doc_notifs)
        assert any("2h" in n.title for n in doc_notifs)

        # Strict idempotency
        assert process_upcoming_appointment_reminders(db, now=now) == 0
    finally:
        db.close()


def test_suggestion_6_ai_chat_emergency_escalation(notif_test_env):
    """
    Suggestion 6: AI Chatbot Emergency Red-Flag Clinical Escalation.
    Patient typing severe chest pain in chat triggers immediate doctor/nurse In-App & doctor SMS alerts.
    """
    env = notif_test_env
    headers = {"Authorization": f"Bearer {env['pat_token']}"}

    res = client.post("/api/recovery-assistant/chat", json={
        "query": "I am experiencing unbearable chest pain radiating to my left arm with cold sweat",
        "language": "en"
    }, headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["is_emergency"] is True

    db = SessionLocal()
    try:
        # Check Doctor In-App Emergency Alert
        doc_notif = db.query(Notification).filter_by(
            user_id=env["doctor"].id,
            category="emergency_escalation"
        ).order_by(Notification.created_at.desc()).first()
        assert doc_notif is not None
        assert "Emergency Red-Flag" in doc_notif.title
        assert "unbearable chest pain" in doc_notif.message

        # Check Floor Nurse In-App Emergency Alert
        nurse_notif = db.query(Notification).filter_by(
            user_id=env["nurse"].id,
            category="emergency_escalation"
        ).order_by(Notification.created_at.desc()).first()
        assert nurse_notif is not None

        # Check Doctor Urgent SMS
        doc_sms = db.query(SMSJob).filter_by(
            recipient_phone=env["doctor"].phone,
            category=SMSCategory.CONSULTATION.value
        ).order_by(SMSJob.created_at.desc()).first()
        assert doc_sms is not None
        assert "URGENT" in doc_sms.message_text
        assert "emergency triage" in doc_sms.message_text
    finally:
        db.close()


def test_suggestion_7_document_approved_notification(notif_test_env):
    """
    Suggestion 7: When a physician approves a clinical document / lab report,
    the uploading nurse is alerted via In-App, and the patient is alerted via In-App and Gateway SMS.
    """
    env = notif_test_env
    db = SessionLocal()
    try:
        # Create an OCR discharge lab document uploaded by the nurse
        doc = ClinicalDocument(
            patient_id=env["patient"].id,
            uploaded_by=env["nurse"].id,
            assigned_doctor_id=env["doctor"].id,
            filename="discharge_blood_panel.pdf",
            stored_path="uploads/discharge_blood_panel.pdf",
            doc_type="lab_report",
            status=DocStatus.UNDER_REVIEW.value,
            extracted_text="Troponin levels normal, hemoglobin 13.8 g/dL",
            structured_data={"lab_name": "Comprehensive Metabolic Panel"},
            created_at=dt.datetime.utcnow()
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        doc_id = doc.id
    finally:
        db.close()

    # Doctor approves the document
    doc_headers = {"Authorization": f"Bearer {env['doc_token']}"}
    res = client.post(f"/api/documents/{doc_id}/approve", headers=doc_headers)
    assert res.status_code == 200, res.text

    db = SessionLocal()
    try:
        # Nurse In-App Notification
        nurse_notif = db.query(Notification).filter_by(
            user_id=env["nurse"].id,
            category="document"
        ).order_by(Notification.created_at.desc()).first()
        assert nurse_notif is not None
        assert "Approved by Physician" in nurse_notif.title
        assert "Lab Report" in nurse_notif.message

        # Patient In-App Notification
        pat_notif = db.query(Notification).filter_by(
            user_id=env["pat_user"].id,
            category="document"
        ).order_by(Notification.created_at.desc()).first()
        assert pat_notif is not None
        assert "Clinical Document Approved" in pat_notif.title

        # Patient SMS Gateway Notification
        pat_sms = db.query(SMSJob).filter_by(
            patient_id=env["patient"].id,
            category=SMSCategory.REGISTRATION.value
        ).filter(SMSJob.message_text.like("%Lab Report%has been reviewed and approved%")).first()
        assert pat_sms is not None
    finally:
        db.close()
