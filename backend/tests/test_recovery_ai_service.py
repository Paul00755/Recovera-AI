# -*- coding: utf-8 -*-
"""
Tests for Recovery AI Service:
1. Healthcare relevance filter (accepts recovery/clinical questions, rejects off-topic).
2. Emergency red-flag detection (chest pain, dyspnea, heavy bleeding).
3. Prescription boundary enforcement.
4. Context gathering from patient records.
5. Deterministic offline fallback engine.
"""
import pytest
from app.services.recovery_ai_service import (
    is_healthcare_relevant,
    detect_regex,
    gather_clinical_context,
    generate_fallback_response,
    call_gemini_assistant
)
from app.models.user import User
from app.models.patient import Patient
from app.core.constants import Role

def test_healthcare_relevance_filter():
    # Valid healthcare recovery queries
    assert is_healthcare_relevant("Can I take my blood pressure tablet after food?") is True
    assert is_healthcare_relevant("My surgical wound is slightly red and itching") is True
    assert is_healthcare_relevant("What are the side effects of paracetamol?") is True
    assert is_healthcare_relevant("வலி அதிகமானால் என்ன செய்ய வேண்டும்?") is True
    assert is_healthcare_relevant("दवा कब लेनी है?") is True

    # Off-topic non-healthcare queries
    assert is_healthcare_relevant("What is the capital of France?") is False
    assert is_healthcare_relevant("Write a Python script to sort a list") is False
    assert is_healthcare_relevant("Who won the election yesterday?") is False

def test_emergency_escalation_detection():
    query = "I am having severe crushing chest pain and difficulty breathing"
    ctx = {
        "user_name": "John Doe",
        "patient_name": "John Doe",
        "patient_id": "test-pt",
        "doctor_name": "Smith",
        "doctor_phone": "+919876543210",
        "medications": [],
        "instructions": [],
        "warning_signs": [],
        "appointments": [],
        "checkins": [],
        "has_history": False,
        "history_conditions": [],
        "history_medications": [],
        "history_notes": ""
    }
    resp = generate_fallback_response(query, ctx, lang="en")
    assert resp["is_emergency"] is True
    assert "emergency" in resp["reply"].lower()

def test_prescription_boundary_enforcement():
    query = "Can you prescribe me an extra dose of antibiotics?"
    ctx = {
        "user_name": "Jane",
        "patient_name": "Jane",
        "patient_id": "test-pt-2",
        "doctor_name": "Smith",
        "doctor_phone": "+919876543210",
        "medications": [],
        "instructions": [],
        "warning_signs": [],
        "appointments": [],
        "checkins": [],
        "has_history": False,
        "history_conditions": [],
        "history_medications": [],
        "history_notes": ""
    }
    resp = generate_fallback_response(query, ctx, lang="en")
    assert resp["prescription_boundary"] is True
    assert resp["suggested_actions"] is not None
    assert len(resp["suggested_actions"]) > 0

def test_context_gathering(db_session):
    # Find a patient with an associated user
    patient = db_session.query(Patient).filter(Patient.active == True).first()
    assert patient is not None, "Active patient required."
    
    # Use patient's user or fallback to a patient role user
    user = patient.user if patient.user else db_session.query(User).filter(User.role == Role.PATIENT.value).first()
    if not user:
        user = db_session.query(User).first()
    
    ctx = gather_clinical_context(db_session, user, patient.id)
    assert ctx is not None
    assert "user_role" in ctx
    assert "user_name" in ctx
    assert "timestamp" in ctx

def test_ai_assistant_live_or_fallback(db_session):
    patient = db_session.query(Patient).filter(Patient.active == True).first()
    user = patient.user if (patient and patient.user) else db_session.query(User).first()
    
    ctx = gather_clinical_context(db_session, user, patient.id if patient else None)
    resp = call_gemini_assistant(
        user_query="How should I manage mild wound swelling after my procedure?",
        ctx=ctx,
        lang="en"
    )
    assert resp is not None
    assert "reply" in resp
    assert len(resp["reply"]) > 20
    assert "model_used" in resp
    assert "suggested_actions" in resp
    assert isinstance(resp["suggested_actions"], list)


def test_ai_assistant_multilingual_tamil(db_session):
    patient = db_session.query(Patient).filter(Patient.active == True).first()
    user = patient.user if (patient and patient.user) else db_session.query(User).first()
    ctx = gather_clinical_context(db_session, user, patient.id if patient else None)

    # 1. Explicit Tamil language request
    resp_ta = call_gemini_assistant(
        user_query="இன்று நான் என்ன மருந்துகளை உட்கொள்ள வேண்டும்?",
        ctx=ctx,
        lang="ta"
    )
    assert resp_ta is not None
    assert any(ord(c) >= 0x0B80 and ord(c) <= 0x0BFF for c in resp_ta["reply"]), "Reply must contain Tamil characters"

    # 2. Script auto-detection: Tamil script query with lang='en'
    resp_auto = call_gemini_assistant(
        user_query="எனது உணவு வழிகாட்டுதல்கள் என்ன?",
        ctx=ctx,
        lang="en"
    )
    assert any(ord(c) >= 0x0B80 and ord(c) <= 0x0BFF for c in resp_auto["reply"]), "Script auto-detection should respond in Tamil"


def test_ai_assistant_multilingual_hindi(db_session):
    patient = db_session.query(Patient).filter(Patient.active == True).first()
    user = patient.user if (patient and patient.user) else db_session.query(User).first()
    ctx = gather_clinical_context(db_session, user, patient.id if patient else None)

    # 1. Explicit Hindi language request
    resp_hi = call_gemini_assistant(
        user_query="मुझे आज कौन सी दवाएं किस उद्देश्य के लिए लेनी हैं?",
        ctx=ctx,
        lang="hi"
    )
    assert resp_hi is not None
    assert any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in resp_hi["reply"]), "Reply must contain Hindi/Devanagari characters"

    # 2. Script auto-detection: Hindi script query with lang='en'
    resp_auto = call_gemini_assistant(
        user_query="घाव की देखभाल के लिए क्या निर्देश हैं?",
        ctx=ctx,
        lang="en"
    )
    assert any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in resp_auto["reply"]), "Script auto-detection should respond in Hindi"

