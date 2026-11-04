import os
import sys
import pytest

# Add backend directory to sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.core.config import settings
from app.core.constants import (
    Role,
    DocStatus,
    OccurrenceState,
    DoctorStatus,
    AppointmentStatus,
    ConsultationMode,
    ConsultationStatus,
    SMSJobStatus,
    SMSCategory,
    RiskLevel,
)
from app.core.security import validate_password_length, get_password_hash, verify_password
from fastapi import HTTPException


def test_no_whatsapp_in_settings_or_env():
    """Verify complete removal of WhatsApp from configuration and environment baseline."""
    env_example_path = os.path.join(backend_path, ".env.example")
    assert os.path.exists(env_example_path), ".env.example must exist"
    with open(env_example_path, "r", encoding="utf-8") as f:
        content = f.read().lower()
    assert "whatsapp" not in content, "WhatsApp must not be in .env.example"
    assert "twilio" not in content, "Twilio must not be in .env.example"

    # Verify settings attributes
    settings_dict = settings.model_dump()
    for key in settings_dict.keys():
        assert "whatsapp" not in key.lower(), f"Settings must not contain WhatsApp key: {key}"
        assert "twilio" not in key.lower(), f"Settings must not contain Twilio key: {key}"


def test_oneplus_sms_gateway_configuration():
    """Verify OnePlus 10T 5G SMS gateway settings are correctly configured."""
    assert hasattr(settings, "ONEPLUS_GATEWAY_URL")
    assert hasattr(settings, "ONEPLUS_GATEWAY_TOKEN")
    assert hasattr(settings, "ONEPLUS_GATEWAY_ENABLED")
    assert hasattr(settings, "SMS_RATE_LIMIT_SECONDS")
    assert hasattr(settings, "SMS_DAILY_LIMIT")
    assert settings.SMS_RATE_LIMIT_SECONDS >= 1.0
    assert settings.CONSULTATION_MAX_MINUTES == 15
    assert settings.APPOINTMENT_DURATION_MINUTES == 60


def test_state_machines_and_enums_contract():
    """Verify domain enums match Version 3.0 specification."""
    # Roles
    assert set(r.value for r in Role) == {"admin", "doctor", "nurse", "patient"}
    
    # Document Statuses
    assert DocStatus.NEEDS_REVIEW.value == "needs_review"
    assert DocStatus.APPROVED.value == "approved"
    assert DocStatus.REJECTED.value == "rejected"
    
    # Medication Occurrence States
    assert set(s.value for s in OccurrenceState) == {"pending", "taken", "not_taken", "missed", "cancelled"}
    
    # Appointments (60 minutes, conflict-free)
    assert AppointmentStatus.SCHEDULED.value == "scheduled"
    assert AppointmentStatus.RESCHEDULED.value == "rescheduled"
    assert DoctorStatus.ONLINE.value == "online"
    assert DoctorStatus.AWAY.value == "away"
    
    # Consultations (15 minutes maximum, real-time modes)
    assert set(m.value for m in ConsultationMode) == {"audio", "video", "chat"}
    assert ConsultationStatus.EXPIRED.value == "expired"
    assert ConsultationStatus.ACTIVE.value == "active"
    
    # SMS One-Way Categories
    assert set(c.value for c in SMSCategory) == {
        "registration",
        "medication_reminder",
        "tracker_followup",
        "appointment",
        "consultation",
    }
    
    # Risk Classification
    assert set(rl.value for rl in RiskLevel) == {"green", "yellow", "red", "insufficient_data"}


def test_documentation_artifacts_exist():
    """Verify all required Phase 0 design artifacts exist in docs/."""
    docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
    required_docs = [
        "ARCHITECTURE.md",
        "SMS_GATEWAY_INTEGRATION.md",
        "API_CONTRACTS.md",
        "ACCEPTANCE_TEST_MATRIX.md",
        "THREAT_MODEL.md",
    ]
    for doc in required_docs:
        doc_path = os.path.join(docs_dir, doc)
        assert os.path.exists(doc_path), f"Missing architecture documentation: {doc}"
        assert os.path.getsize(doc_path) > 100, f"Documentation file {doc} appears empty"


def test_bcrypt_72_byte_limit_validation():
    """Verify bcrypt passwords > 72 UTF-8 bytes are rejected safely."""
    # Valid password
    valid_pw = "ValidHospitalPassword2026!"
    hashed = get_password_hash(valid_pw)
    assert verify_password(valid_pw, hashed) is True

    # Password exceeding 72 bytes
    long_pw = "A" * 75
    with pytest.raises(HTTPException) as exc_info:
        validate_password_length(long_pw)
    assert exc_info.value.status_code == 422
    assert "72 bytes" in str(exc_info.value.detail)
