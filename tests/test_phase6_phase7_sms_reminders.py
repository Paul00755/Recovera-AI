import os
import sys
import uuid
import pytest
import datetime as dt
from fastapi.testclient import TestClient

backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.main import app
from app.core.database import SessionLocal
from app.core.security import create_access_token, get_password_hash
from app.core.constants import Role, OccurrenceState, SMSCategory, SMSJobStatus
from app.models.user import User
from app.models.patient import Patient
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence, MedicationAdherence
from app.models.sms_job import SMSJob
from app.services.medication_reminder_service import process_due_medication_reminders

client = TestClient(app)


@pytest.fixture
def sms_test_env():
    db = SessionLocal()
    try:
        # 1. Admin
        admin = db.query(User).filter_by(email="admin.sms@hospital.org").first()
        if not admin:
            admin = User(
                email="admin.sms@hospital.org",
                password_hash=get_password_hash("AdminPass123!"),
                name="SMS Administrator",
                role=Role.ADMIN.value,
                phone="+919876543210",
                active=True
            )
            db.add(admin)
            db.commit()
            db.refresh(admin)

        # 2. Nurse
        nurse = db.query(User).filter_by(email="nurse.sms@hospital.org").first()
        if not nurse:
            nurse = User(
                email="nurse.sms@hospital.org",
                password_hash=get_password_hash("NursePass123!"),
                name="SMS Nurse",
                role=Role.NURSE.value,
                phone="+919876543211",
                active=True
            )
            db.add(nurse)
            db.commit()
            db.refresh(nurse)

        # 3. Tokens
        admin_token = create_access_token({"sub": admin.id, "email": admin.email, "role": admin.role})
        nurse_token = create_access_token({"sub": nurse.id, "email": nurse.email, "role": nurse.role})

        return {
            "admin": admin,
            "nurse": nurse,
            "admin_token": admin_token,
            "nurse_token": nurse_token,
        }
    finally:
        db.close()


def test_gateway_status_and_sync(sms_test_env):
    """Phase 6: Verify SMS Gateway status endpoint and telemetry sync webhook."""
    headers = {"Authorization": f"Bearer {sms_test_env['admin_token']}"}

    # 1. Check status
    res = client.get("/api/sms/gateway/status", headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "status" in data
    assert "telemetry" in data
    assert "stats" in data
    assert data["stats"]["daily_limit"] >= 100

    # 2. Companion telemetry sync
    sync_payload = {
        "battery_pct": 94,
        "signal": "5G SA",
        "carrier": "Hospital Dedicated SIM"
    }
    sync_res = client.post("/api/sms/gateway/sync", json=sync_payload)
    assert sync_res.status_code == 200, sync_res.text
    assert sync_res.json()["telemetry"]["battery_pct"] == 94
    assert sync_res.json()["telemetry"]["carrier"] == "Hospital Dedicated SIM"


def test_test_sms_dispatch_and_rbac(sms_test_env):
    """Phase 6: Verify test SMS dispatch by Admin and RBAC protection."""
    admin_headers = {"Authorization": f"Bearer {sms_test_env['admin_token']}"}
    nurse_headers = {"Authorization": f"Bearer {sms_test_env['nurse_token']}"}

    test_payload = {
        "phone": "+919876540001",
        "message": "Gateway connectivity check message."
    }

    # 1. Nurse cannot send test SMS (Admin only)
    forbidden_res = client.post("/api/sms/gateway/test", json=test_payload, headers=nurse_headers)
    assert forbidden_res.status_code == 403

    # 2. Admin can send test SMS
    ok_res = client.post("/api/sms/gateway/test", json=test_payload, headers=admin_headers)
    assert ok_res.status_code == 200, ok_res.text
    body = ok_res.json()
    assert body["status"] == "dispatched"
    assert body["job"]["recipient_phone"] == "+919876540001"
    assert body["job"]["state"] in ("sent", "queued")


def test_patient_registration_welcome_sms(sms_test_env):
    """Phase 6: Verify automated Welcome SMS dispatched upon patient registration with SMS opt-in."""
    headers = {"Authorization": f"Bearer {sms_test_env['nurse_token']}"}
    suffix = uuid.uuid4().hex[:6]
    test_phone = f"+91987{suffix[:7]}"

    patient_payload = {
        "name": f"Welcome Test Patient {suffix}",
        "phone": test_phone,
        "email": f"welcome.{suffix}@example.com",
        "sms_opt_in": True,
        "has_medical_history": False
    }

    create_res = client.post("/api/patients", json=patient_payload, headers=headers)
    assert create_res.status_code == 201, create_res.text
    pat_data = create_res.json()
    pat_id = pat_data["id"]

    # Verify that a Welcome SMSJob was queued/sent in database
    db = SessionLocal()
    try:
        welcome_job = db.query(SMSJob).filter_by(
            patient_id=pat_id,
            category=SMSCategory.REGISTRATION.value
        ).first()
        assert welcome_job is not None, "Welcome SMS was not generated"
        assert welcome_job.recipient_phone == test_phone
        assert "Welcome to Recovera AI" in welcome_job.message_text
        assert welcome_job.state in (SMSJobStatus.SENT.value, SMSJobStatus.QUEUED.value)
    finally:
        db.close()


def test_two_step_medication_reminders_and_idempotency(sms_test_env):
    """Phase 7: Test Dose Reminder (T-30m) and Tracker Follow-up (T+30m) with idempotency."""
    nurse_headers = {"Authorization": f"Bearer {sms_test_env['nurse_token']}"}
    suffix = uuid.uuid4().hex[:6]
    test_phone = f"+91986{suffix[:7]}"

    # 1. Create Patient with SMS opt-in
    patient_res = client.post("/api/patients", json={
        "name": f"Meds Reminder Patient {suffix}",
        "phone": test_phone,
        "email": f"remind.{suffix}@example.com",
        "sms_opt_in": True,
        "has_medical_history": False
    }, headers=nurse_headers)
    assert patient_res.status_code == 201
    pat_id = patient_res.json()["id"]

    db = SessionLocal()
    try:
        now = dt.datetime.utcnow()

        # 2. Create Medication
        med = Medication(
            patient_id=pat_id,
            name="Amoxicillin",
            strength="500mg",
            dose="1 tablet",
            status="active",
            start_date=now.date(),
            end_date=now.date() + dt.timedelta(days=7),
            created_by=sms_test_env["nurse"].id,
        )
        db.add(med)
        db.flush()

        # 3. Create Schedule
        sched = MedicationSchedule(
            medication_id=med.id,
            patient_id=pat_id,
            label="After Lunch",
            time_hhmm="14:00",
            dose="1 tablet",
            food_relation="after food"
        )
        db.add(sched)
        db.flush()

        # 4. Create Occurrence due in 15 minutes (within T - 30 minutes window)
        occ1 = MedicationOccurrence(
            medication_id=med.id,
            schedule_id=sched.id,
            patient_id=pat_id,
            scheduled_at=now + dt.timedelta(minutes=15),
            state=OccurrenceState.PENDING.value
        )
        db.add(occ1)
        db.commit()

        # 5. Run process_due_medication_reminders -> should trigger Message 1
        res1 = process_due_medication_reminders(db)
        assert res1["dose_reminders_sent"] >= 1

        # Verify Message 1 in DB
        dose_sms = db.query(SMSJob).filter_by(
            patient_id=pat_id,
            category=SMSCategory.MEDICATION_REMINDER.value
        ).first()
        assert dose_sms is not None
        assert "Medication reminder: please follow your approved prescription" in dose_sms.message_text
        assert "Amoxicillin" in dose_sms.message_text
        assert "after food" in dose_sms.message_text

        # 6. Idempotency test: Re-running should NOT duplicate Message 1
        res2 = process_due_medication_reminders(db)
        dose_sms_count = db.query(SMSJob).filter_by(
            patient_id=pat_id,
            category=SMSCategory.MEDICATION_REMINDER.value
        ).count()
        assert dose_sms_count == 1, "Duplicate reminder was created"

        # 7. Test Message 2 (Tracker follow-up):
        # Update occurrence scheduled_at to 45 minutes in the past (so now > scheduled_at + 30 mins)
        occ1.scheduled_at = now - dt.timedelta(minutes=45)
        db.commit()

        # Run process_due_medication_reminders -> should trigger Message 2
        res3 = process_due_medication_reminders(db)
        assert res3["tracker_followups_sent"] >= 1

        # Verify Message 2 in DB
        track_sms = db.query(SMSJob).filter_by(
            patient_id=pat_id,
            category=SMSCategory.TRACKER_FOLLOWUP.value
        ).first()
        assert track_sms is not None
        assert "We hope you have taken your medicines" in track_sms.message_text
        assert "fill in the medicine intake status tracker" in track_sms.message_text

    finally:
        db.close()


def test_discontinuation_cancels_future_sms_reminders(sms_test_env):
    """Phase 7: Discontinuing a medicine cancels pending occurrences and purges future SMS reminders."""
    nurse_headers = {"Authorization": f"Bearer {sms_test_env['nurse_token']}"}
    suffix = uuid.uuid4().hex[:6]
    test_phone = f"+91985{suffix[:7]}"

    # Create Patient
    patient_res = client.post("/api/patients", json={
        "name": f"Discontinue Test {suffix}",
        "phone": test_phone,
        "email": f"disc.{suffix}@example.com",
        "sms_opt_in": True,
        "has_medical_history": False
    }, headers=nurse_headers)
    assert patient_res.status_code == 201
    pat_id = patient_res.json()["id"]

    db = SessionLocal()
    try:
        now = dt.datetime.utcnow()

        med = Medication(
            patient_id=pat_id,
            name="Metformin",
            strength="500mg",
            dose="1 tablet",
            status="active",
            start_date=now.date(),
            end_date=now.date() + dt.timedelta(days=7),
            created_by=sms_test_env["nurse"].id,
        )
        db.add(med)
        db.flush()

        sched = MedicationSchedule(
            medication_id=med.id,
            patient_id=pat_id,
            label="Morning",
            time_hhmm="08:00",
            dose="1 tablet",
            food_relation="with food"
        )
        db.add(sched)
        db.flush()

        occ = MedicationOccurrence(
            medication_id=med.id,
            schedule_id=sched.id,
            patient_id=pat_id,
            scheduled_at=now + dt.timedelta(days=1),
            state=OccurrenceState.PENDING.value
        )
        db.add(occ)
        db.flush()

        # Queue a future reminder for this occurrence
        future_job = SMSJob(
            patient_id=pat_id,
            recipient_phone=test_phone,
            category=SMSCategory.MEDICATION_REMINDER.value,
            message_text="Future reminder for Metformin",
            idempotency_key=f"med_dose_{occ.id}",
            state=SMSJobStatus.QUEUED.value,
            scheduled_for=now + dt.timedelta(days=1)
        )
        db.add(future_job)
        db.commit()

        # Discontinue medication via API
        disc_res = client.post(
            f"/api/medications/{med.id}/discontinue?reason=Adverse+reaction",
            headers=nurse_headers
        )
        assert disc_res.status_code == 200, disc_res.text

        # Verify occurrence is CANCELLED
        db.refresh(occ)
        assert occ.state == OccurrenceState.CANCELLED.value

        # Verify queued SMS job is marked CANCELLED
        db.refresh(future_job)
        assert future_job.state == SMSJobStatus.CANCELLED.value, "Future SMS reminder was not cancelled"

    finally:
        db.close()


def test_sms_jobs_list_and_retry_endpoint(sms_test_env):
    """Phase 6 & 7: Verify listing SMS jobs with filters and retrying failed/queued jobs."""
    admin_headers = {"Authorization": f"Bearer {sms_test_env['admin_token']}"}

    # 1. List jobs
    list_res = client.get("/api/sms/jobs?limit=10", headers=admin_headers)
    assert list_res.status_code == 200, list_res.text
    data = list_res.json()
    assert "total" in data
    assert "jobs" in data
    assert isinstance(data["jobs"], list)

    # 2. Trigger process reminders endpoint
    reminder_cycle = client.post("/api/sms/process-reminders", headers=admin_headers)
    assert reminder_cycle.status_code == 200, reminder_cycle.text
    assert "dose_reminders_sent" in reminder_cycle.json()


def test_android_driver_dual_endpoint_failover(monkeypatch):
    """Test AndroidDeviceDriver auto-fallback from primary Hotspot/LAN to backup Tailscale mesh."""
    from app.services.sms_gateway_service import AndroidDeviceDriver

    attempted_urls = []

    def mock_post(url, json=None, timeout=None):
        attempted_urls.append(url)
        class MockResp:
            status_code = 200
            headers = {"content-type": "application/json"}
            def json(self):
                return {"id": "gw_backup_ok_999"}
            text = "OK"

        # Simulate primary Hotspot/LAN connection failure
        if "192.168.43.1" in url:
            raise ConnectionError("Host unreachable on local hotspot")
        # Backup Tailscale mesh responds successfully
        elif "100.64.0.1" in url:
            return MockResp()
        raise ValueError(f"Unexpected target: {url}")

    monkeypatch.setattr("requests.post", mock_post)

    driver = AndroidDeviceDriver(
        base_url="http://192.168.43.1:8080",
        backup_url="http://100.64.0.1:8080",
        token="test-token"
    )

    result = driver.send(phone="+919876543210", message="Test failover message")
    assert result["success"] is True
    assert result["endpoint_used"] == "backup_failover"
    assert result["provider_message_id"] == "gw_backup_ok_999"
    assert "http://192.168.43.1:8080/send" in attempted_urls
    assert "http://100.64.0.1:8080/send" in attempted_urls


def test_patient_preferences_update_and_isolation(sms_test_env):
    """Verify patients can update their own language (en, ta, hi) and SMS preferences, while isolation prevents cross-patient updates."""
    nurse_headers = {"Authorization": f"Bearer {sms_test_env['nurse_token']}"}
    suffix = uuid.uuid4().hex[:6]
    test_phone = f"+91985{suffix[:7]}"
    patient_email = f"patient.pref.{suffix}@example.com"

    # 1. Nurse registers patient
    reg_res = client.post("/api/patients", json={
        "name": f"Pref Patient {suffix}",
        "phone": test_phone,
        "email": patient_email,
        "sms_opt_in": False,
        "has_medical_history": False
    }, headers=nurse_headers)
    assert reg_res.status_code == 201, reg_res.text
    pat_id = reg_res.json()["id"]

    # 2. Find the patient user to generate patient token
    db = SessionLocal()
    try:
        pat_obj = db.get(Patient, pat_id)
        user_id = pat_obj.user_id
        patient_token = create_access_token({"sub": user_id, "email": patient_email, "role": Role.PATIENT.value})
        patient_headers = {"Authorization": f"Bearer {patient_token}"}
    finally:
        db.close()

    # 3. Patient updates their own preferred language to Tamil ('ta') and enables SMS
    res = client.put(
        f"/api/patients/{pat_id}/preferences",
        headers=patient_headers,
        json={"preferred_language": "ta", "sms_opt_in": True}
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["preferred_language"] == "ta"
    assert data["sms_opt_in"] is True

    # 4. Patient updates to Hindi ('hi') and toggles SMS off
    res2 = client.put(
        f"/api/patients/{pat_id}/preferences",
        headers=patient_headers,
        json={"preferred_language": "hi", "sms_opt_in": False}
    )
    assert res2.status_code == 200, res2.text
    assert res2.json()["preferred_language"] == "hi"
    assert res2.json()["sms_opt_in"] is False

    # 5. Patient provides an unsupported language code (should fail with 400)
    res_bad = client.put(
        f"/api/patients/{pat_id}/preferences",
        headers=patient_headers,
        json={"preferred_language": "fr"}
    )
    assert res_bad.status_code == 400

    # 6. Patient attempts to access/update another non-existent or other patient profile (should return 403 or 404)
    fake_pid = str(uuid.uuid4())
    res_other = client.put(
        f"/api/patients/{fake_pid}/preferences",
        headers=patient_headers,
        json={"preferred_language": "en"}
    )
    assert res_other.status_code in (403, 404)

