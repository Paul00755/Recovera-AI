import datetime as dt
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.database import get_db
from app.core.constants import Role, DocStatus
from app.core.audit import record_audit
from app.models.recovery_plan import RecoveryPlan
from app.models.patient import Patient
from app.models.user import User
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
        # First query for latest published plan
        plan = db.scalar(
            select(RecoveryPlan)
            .where(RecoveryPlan.patient_id == pid, RecoveryPlan.status == DocStatus.PUBLISHED.value)
            .order_by(RecoveryPlan.version.desc(), RecoveryPlan.created_at.desc())
        )
        if not plan:
            # Fallback to any latest plan with instructions and auto-publish so it doesn't vanish on login
            plan = db.scalar(
                select(RecoveryPlan)
                .where(RecoveryPlan.patient_id == pid)
                .order_by(RecoveryPlan.version.desc(), RecoveryPlan.created_at.desc())
            )
            if plan and plan.instructions:
                plan.status = DocStatus.PUBLISHED.value
                db.commit()
                db.refresh(plan)
            else:
                raise HTTPException(
                    status_code=404,
                    detail="Your recovery plan is currently being prepared or updated by the clinical team. Please check back shortly."
                )
    else:
        plan = db.scalar(
            select(RecoveryPlan)
            .where(RecoveryPlan.patient_id == pid)
            .order_by(RecoveryPlan.created_at.desc())
        )

    if not plan:
        raise HTTPException(status_code=404, detail="No recovery plan found for this patient.")

    return plan

@router.post("/api/patients/{pid}/recovery-plan/generate", response_model=RecoveryPlanOut, status_code=status.HTTP_201_CREATED)
def generate_recovery_plan(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    return generate_plan_from_approved_documents(patient_id=pid, db=db, user_id=current_user.id, force=True)

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
        # Nurse cannot set status to published
        if payload.status == DocStatus.PUBLISHED.value and current_user.role == Role.NURSE.value:
            raise HTTPException(
                status_code=403,
                detail="Forbidden: Nurse cannot publish recovery plans. Doctor approval required."
            )
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
    return plan

@router.post("/api/patients/{pid}/recovery-plan/publish", response_model=RecoveryPlanOut)
def publish_plan_endpoint(
    pid: str,
    payload: Optional[RecoveryPlanUpdate] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    instructions = payload.instructions if payload else None
    warning_signs = payload.warning_signs if payload else None

    return publish_recovery_plan(
        patient_id=pid,
        db=db,
        user=current_user,
        instructions=instructions,
        warning_signs=warning_signs
    )
