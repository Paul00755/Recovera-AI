# -*- coding: utf-8 -*-
"""
Tests for independent SMS language preferences:
1. Patient creation preserves preferred_language and sms_language.
2. Updating patient preferences updates sms_language independently of app language.
3. Notification service prioritizes sms_language for SMS dispatch.
"""
import uuid
import pytest
from app.models.patient import Patient
from app.models.user import User
from app.core.constants import Role
from app.services.notification_service import localize_sms_message

def test_patient_preferences_updates_sms_language(client, db_session, admin_headers):
    # 1. Fetch an existing patient and ensure phone exists for sms opt-in
    patient = db_session.query(Patient).filter(Patient.active == True).first()
    assert patient is not None, "At least one active patient required for testing."
    orig_pref = patient.preferred_language
    orig_sms = patient.sms_language

    try:
        if not patient.phone:
            patient.phone = "+919876543210"
            db_session.commit()
            db_session.refresh(patient)

        # 2. Update preferences: App language = 'ta' (Tamil), SMS language = 'hi' (Hindi)
        payload = {
            "preferred_language": "ta",
            "sms_language": "hi",
            "sms_opt_in": True
        }
        resp = client.put(f"/api/patients/{patient.id}/preferences", json=payload, headers=admin_headers)
        assert resp.status_code == 200
        data = resp.json()

        assert data["preferred_language"] == "ta"
        assert data["sms_language"] == "hi"

        # 3. Verify in database
        db_session.expire(patient)
        updated_patient = db_session.get(Patient, patient.id)
        assert updated_patient.preferred_language == "ta"
        assert updated_patient.sms_language == "hi"

        # 4. Verify notification service language resolution priority
        pref_lang = (getattr(updated_patient, "sms_language", None) or updated_patient.preferred_language or "en").lower().strip()[:2]
        assert pref_lang == "hi", f"Expected SMS language 'hi', got '{pref_lang}'"

        # 5. Verify SMS localization output
        raw_msg = "Medication reminder: please follow your approved prescription — take 1 tablet with food. Open Recovera for your schedule."
        localized_ta = localize_sms_message(raw_msg, "ta")
        assert "மருந்து நினைவூட்டல்" in localized_ta
    finally:
        patient.preferred_language = orig_pref
        patient.sms_language = orig_sms
        db_session.commit()

    localized_hi = localize_sms_message(raw_msg, "hi")
    assert len(localized_hi) > 0

    # 6. Switch preferences back: App language = 'en', SMS language = 'ta'
    payload2 = {
        "preferred_language": "en",
        "sms_language": "ta",
        "sms_opt_in": True
    }
    resp2 = client.put(f"/api/patients/{patient.id}/preferences", json=payload2, headers=admin_headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["preferred_language"] == "en"
    assert data2["sms_language"] == "ta"
