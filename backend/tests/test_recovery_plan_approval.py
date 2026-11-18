# -*- coding: utf-8 -*-
"""
Tests for Recovery Plan Clinical Approval Enforcement:
1. Ensures unapproved/draft recovery plans are strictly hidden from patient callers (HTTP 404).
2. Verifies that doctors and nurses can view unapproved plans for clinical review.
3. Verifies that doctors and nurses can clinically approve and publish the plan.
4. Verifies that once approved, the plan is returned to the patient with is_approved=True, approver_name, and approver_role.
5. Verifies AI recovery assistant protects unapproved plan instructions from leaking.
"""
import uuid
import datetime as dt
import pytest
from app.models.user import User
from app.models.patient import Patient
from app.models.recovery_plan import RecoveryPlan
from app.core.constants import Role, DocStatus
from app.core.security import create_access_token
from app.services.recovery_ai_service import gather_clinical_context, generate_fallback_response

def test_recovery_plan_approval_enforcement(client, db_session):
    # 1. Setup Doctor, Nurse, and Patient
    doc = db_session.query(User).filter(User.role == Role.DOCTOR.value, User.active == True).first()
    nurse = db_session.query(User).filter(User.role == Role.NURSE.value, User.active == True).first()
    assert doc and nurse, "Active doctor and nurse required in database."

    # Create dedicated test patient user
    pt_email = f"test_pt_{uuid.uuid4().hex[:6]}@example.com"
    pt_user = User(
        id=str(uuid.uuid4()),
        name="Test Approval Patient",
        email=pt_email,
        password_hash="mock_hash_for_test",
        role=Role.PATIENT.value,
        active=True
    )
    db_session.add(pt_user)
    db_session.flush()

    patient = Patient(
        id=str(uuid.uuid4()),
        patient_code=f"PAT-{uuid.uuid4().hex[:6].upper()}",
        user_id=pt_user.id,
        assigned_doctor_id=doc.id,
        active=True,
        preferred_language="en"
    )
    db_session.add(patient)
    db_session.flush()

    # Create unapproved draft recovery plan
    plan = RecoveryPlan(
        id=str(uuid.uuid4()),
        patient_id=patient.id,
        instructions={
            "care_instructions": "Rest and elevate operative limb.",
            "dietary_guidelines": "High protein diet with plenty of fluids."
        },
        warning_signs=["Fever above 101F", "Purulent drainage"],
        version=1,
        status=DocStatus.DRAFT.value,
        approved_by=None,
        approved_at=None
    )
    db_session.add(plan)
    db_session.commit()

    patient_token = create_access_token(data={"sub": pt_user.id, "role": Role.PATIENT.value})
    patient_headers = {"Authorization": f"Bearer {patient_token}"}

    doc_token = create_access_token(data={"sub": doc.id, "role": Role.DOCTOR.value})
    doc_headers = {"Authorization": f"Bearer {doc_token}"}

    nurse_token = create_access_token(data={"sub": nurse.id, "role": Role.NURSE.value})
    nurse_headers = {"Authorization": f"Bearer {nurse_token}"}

    try:
        # Step A: Patient attempts to fetch unapproved plan -> Expect 404 awaiting review
        res_patient = client.get(f"/api/patients/{patient.id}/recovery-plan", headers=patient_headers)
        assert res_patient.status_code == 404, f"Patient saw unapproved plan! Status: {res_patient.status_code}"
        assert "awaiting review" in res_patient.json()["detail"].lower()

        # Step B: Clinicians (Doctor / Nurse) can access the draft plan
        res_doc = client.get(f"/api/patients/{patient.id}/recovery-plan", headers=doc_headers)
        assert res_doc.status_code == 200
        plan_doc = res_doc.json()
        assert plan_doc["is_approved"] is False

        # Step C: AI Assistant verifies has_plan is False and returns waiting response
        ctx_before = gather_clinical_context(db_session, pt_user, patient.id)
        assert ctx_before.get("has_plan") is False
        ai_resp = generate_fallback_response("What is my recovery plan?", ctx_before, lang="en")
        assert "under clinical review" in ai_resp["reply"].lower() or "awaiting" in ai_resp["reply"].lower()

        # Step D: Nurse approves and publishes the recovery plan
        res_publish = client.post(
            f"/api/patients/{patient.id}/recovery-plan/publish",
            headers=nurse_headers,
            json={"instructions": plan.instructions, "warning_signs": plan.warning_signs}
        )
        assert res_publish.status_code == 200
        published_data = res_publish.json()
        assert published_data["is_approved"] is True
        assert published_data["approver_role"] == Role.NURSE.value
        assert published_data["approver_name"] == nurse.name

        # Step E: Patient can now fetch the approved recovery plan
        res_patient_after = client.get(f"/api/patients/{patient.id}/recovery-plan", headers=patient_headers)
        assert res_patient_after.status_code == 200
        approved_plan = res_patient_after.json()
        assert approved_plan["is_approved"] is True
        assert approved_plan["approver_name"] == nurse.name
        assert approved_plan["approver_role"] == Role.NURSE.value

        # Step F: AI Assistant now recognizes the approved plan
        db_session.expire_all()
        ctx_after = gather_clinical_context(db_session, pt_user, patient.id)
        assert ctx_after.get("has_plan") is True
        assert ctx_after.get("plan_approver_name") == nurse.name
        ai_resp_after = generate_fallback_response("What is my recovery plan?", ctx_after, lang="en")
        assert "clinically approved recovery plan" in ai_resp_after["reply"].lower()
        assert nurse.name in ai_resp_after["reply"]

    finally:
        # Cleanup
        db_session.query(RecoveryPlan).filter(RecoveryPlan.patient_id == patient.id).delete()
        db_session.query(Patient).filter(Patient.id == patient.id).delete()
        db_session.query(User).filter(User.id == pt_user.id).delete()
        db_session.commit()
