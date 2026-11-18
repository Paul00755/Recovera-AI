import datetime as dt
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.constants import Role, DocStatus
from app.core.audit import record_audit
from app.models.recovery_plan import RecoveryPlan
from app.models.patient import Patient
from app.models.user import User
from app.models.clinical_document import ClinicalDocument
from app.models.medication import Medication
from app.schemas.recovery_plan import RecoveryPlanOut, RecoveryPlanUpdate
from app.services.recovery_service import (
    generate_plan_from_approved_documents,
    publish_recovery_plan,
)
from app.api.deps import get_current_user, require_role, enforce_patient_isolation

router = APIRouter(tags=["Recovery Plans"])

@router.get("/api/patients/{pid}/recovery-plan", response_model=RecoveryPlanOut)
def get_recovery_plan(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)

    if current_user.role == Role.PATIENT.value:
        # STRICT CLINICAL RULE: Patient can ONLY view recovery plans that have been
        # explicitly reviewed, approved, and published by their assigned doctor or nurse.
        plan = db.scalar(
            select(RecoveryPlan)
            .where(
                RecoveryPlan.patient_id == pid,
                RecoveryPlan.status.in_([DocStatus.PUBLISHED.value, "approved"]),
                RecoveryPlan.approved_by.isnot(None)
            )
            .order_by(RecoveryPlan.version.desc(), RecoveryPlan.created_at.desc())
        )
        if not plan:
            raise HTTPException(
                status_code=404,
                detail="Your recovery plan is currently awaiting review and approval by your assigned care team (Doctor / Nurse). Please check back once your clinician has finalized and approved your care plan."
            )
    else:
        # Clinicians (Doctors, Nurses, Admins) view the latest plan (including drafts for review)
        plan = db.scalar(
            select(RecoveryPlan)
            .where(RecoveryPlan.patient_id == pid)
            .order_by(RecoveryPlan.version.desc(), RecoveryPlan.created_at.desc())
        )

    if not plan:
        raise HTTPException(status_code=404, detail="No recovery plan found for this patient.")

    plan_out = RecoveryPlanOut.model_validate(plan)
    plan_out.is_approved = (plan.status in [DocStatus.PUBLISHED.value, "approved"]) and bool(plan.approved_by)
    if plan.approved_by:
        approver = db.get(User, plan.approved_by)
        if approver:
            plan_out.approver_name = approver.name
            plan_out.approver_role = approver.role
    return plan_out

@router.post("/api/patients/{pid}/recovery-plan/generate", response_model=RecoveryPlanOut, status_code=status.HTTP_201_CREATED)
def generate_recovery_plan(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    plan = generate_plan_from_approved_documents(patient_id=pid, db=db, user_id=current_user.id, force=True)
    plan_out = RecoveryPlanOut.model_validate(plan)
    plan_out.is_approved = (plan.status in [DocStatus.PUBLISHED.value, "approved"]) and bool(plan.approved_by)
    return plan_out

@router.put("/api/patients/{pid}/recovery-plan", response_model=RecoveryPlanOut)
def update_recovery_plan(
    pid: str,
    payload: RecoveryPlanUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    plan = db.scalar(
        select(RecoveryPlan)
        .where(RecoveryPlan.patient_id == pid)
        .order_by(RecoveryPlan.created_at.desc())
    )
    if not plan:
        raise HTTPException(status_code=404, detail="No recovery plan found to update.")

    if payload.instructions is not None:
        plan.instructions = payload.instructions
    if payload.warning_signs is not None:
        plan.warning_signs = payload.warning_signs
    if payload.status:
        # If transitioning to published/approved, record clinical approver
        if payload.status in [DocStatus.PUBLISHED.value, "approved"]:
            plan.approved_by = current_user.id
            plan.approved_at = dt.datetime.utcnow()
        plan.status = payload.status

    plan.updated_at = dt.datetime.utcnow()
    plan.content = {
        "instructions": plan.instructions,
        "warning_signs": plan.warning_signs,
        "version": plan.version
    }

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="recovery_plan_updated",
        target_type="recovery_plan",
        target_id=plan.id,
        new_values={"status": plan.status}
    )
    db.commit()
    db.refresh(plan)

    plan_out = RecoveryPlanOut.model_validate(plan)
    plan_out.is_approved = (plan.status in [DocStatus.PUBLISHED.value, "approved"]) and bool(plan.approved_by)
    if plan.approved_by:
        approver = db.get(User, plan.approved_by)
        if approver:
            plan_out.approver_name = approver.name
            plan_out.approver_role = approver.role
    return plan_out

@router.post("/api/patients/{pid}/recovery-plan/publish", response_model=RecoveryPlanOut)
def publish_plan_endpoint(
    pid: str,
    payload: Optional[RecoveryPlanUpdate] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    instructions = payload.instructions if payload else None
    warning_signs = payload.warning_signs if payload else None

    plan = publish_recovery_plan(
        patient_id=pid,
        db=db,
        user=current_user,
        instructions=instructions,
        warning_signs=warning_signs
    )
    plan_out = RecoveryPlanOut.model_validate(plan)
    plan_out.is_approved = True
    plan_out.approver_name = current_user.name
    plan_out.approver_role = current_user.role
    return plan_out
