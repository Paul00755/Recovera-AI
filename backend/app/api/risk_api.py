"""
Recovera AI – Explainable Risk & Analytics API (Phase 12)

Endpoints:
  GET   /api/patients/{pid}/risk             – Get latest clinical risk classification
  POST  /api/patients/{pid}/risk/evaluate    – Force re-evaluation of risk score
  POST  /api/patients/{pid}/risk/override    – Clinician override with documented rationale
  GET   /api/patients/{pid}/analytics/trends – Longitudinal adherence and pain trajectory
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.constants import Role, RiskLevel
from app.core.audit import record_audit
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.user import User
from app.models.risk_history import RiskClassificationHistory
from app.api.deps import get_current_user, require_role
from app.services.risk_service import (
    evaluate_patient_risk,
    override_patient_risk,
    get_patient_analytics_trends
)

router = APIRouter(tags=["Clinical Risk & Analytics"])


class RiskOverrideRequest(BaseModel):
    level: str = Field(..., description="green, yellow, red, or insufficient_data")
    override_reason: str = Field(..., min_length=5, description="Documented clinical rationale for manual override")


def _risk_to_dict(r: RiskClassificationHistory, db: Session) -> dict:
    from app.models.checkin import DailyCheckIn
    import datetime as dt

    doctor_name = None
    if r.overridden_by:
        doc = db.get(User, r.overridden_by)
        if doc:
            doctor_name = doc.name
    lvl_cap = r.level.capitalize() if r.level else "Green"
    lvl_lower = str(r.level).lower() if r.level else "green"
    reasons_count = len(r.explainability_reasons or [])
    if lvl_lower == "red":
        computed_score = min(96, 72 + (reasons_count * 4))
    elif lvl_lower == "yellow":
        computed_score = min(68, 38 + (reasons_count * 5))
    elif lvl_lower == "green":
        computed_score = max(5, min(25, 10 + (reasons_count * 2)))
    else:
        computed_score = 0

    risk_score = getattr(r, "risk_score", None)
    if risk_score is None or risk_score == 0:
        risk_score = computed_score

    today_checkin = db.scalar(
        select(DailyCheckIn.id).where(
            DailyCheckIn.patient_id == r.patient_id,
            DailyCheckIn.checkin_date == dt.date.today()
        ).limit(1)
    )

    return {
        "id": r.id,
        "patient_id": r.patient_id,
        "level": r.level,
        "risk_level": lvl_cap,
        "risk_score": risk_score,
        "explainability_reasons": r.explainability_reasons or [],
        "insufficient_data": r.insufficient_data,
        "data_window_days": r.data_window_days,
        "is_override": r.is_override,
        "overridden": r.is_override,
        "overridden_by": r.overridden_by,
        "overridden_by_name": doctor_name,
        "override_reason": r.override_reason,
        "evaluated_at": r.evaluated_at.isoformat() if r.evaluated_at else None,
        "evaluation_timestamp": r.evaluated_at.isoformat() if r.evaluated_at else None,
        "has_today_checkin": bool(today_checkin),
    }


def _verify_patient_access(patient: Patient, current_user: User, db: Session):
    if current_user.role == Role.PATIENT.value:
        if patient.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied.")
    elif current_user.role == Role.DOCTOR.value:
        if patient.assigned_doctor_id != current_user.id:
            assignment = db.scalar(
                select(PatientDoctorAssignment).where(
                    PatientDoctorAssignment.patient_id == patient.id,
                    PatientDoctorAssignment.doctor_id == current_user.id
                )
            )
            if not assignment:
                raise HTTPException(status_code=403, detail="Doctor can only access clinical data for assigned patients.")


@router.get("/api/patients/{patient_id}/risk")
def get_patient_risk(
    patient_id: str,
    fresh: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = db.get(Patient, patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")
    _verify_patient_access(patient, current_user, db)

    latest = None
    if not fresh:
        latest = db.scalar(
            select(RiskClassificationHistory)
            .where(RiskClassificationHistory.patient_id == patient_id)
            .order_by(RiskClassificationHistory.evaluated_at.desc())
        )

    if not latest:
        # Evaluate on the fly
        latest = evaluate_patient_risk(db, patient_id=patient_id, window_days=7, force=fresh)

    return _risk_to_dict(latest, db)


@router.post("/api/patients/{patient_id}/risk/evaluate")
def reevaluate_patient_risk(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value, Role.PATIENT.value)),
):
    patient = db.get(Patient, patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")
    _verify_patient_access(patient, current_user, db)

    record = evaluate_patient_risk(db, patient_id=patient_id, window_days=7, force=True)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="risk_evaluated",
        target_type="patient",
        target_id=patient_id,
        details={"level": record.level, "reasons": record.explainability_reasons}
    )
    return _risk_to_dict(record, db)


@router.post("/api/patients/{patient_id}/risk/override")
def override_risk(
    patient_id: str,
    req: RiskOverrideRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value)),
):
    patient = db.get(Patient, patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")
    _verify_patient_access(patient, current_user, db)

    try:
        record = override_patient_risk(
            db=db,
            patient_id=patient_id,
            level=req.level,
            override_reason=req.override_reason,
            doctor_id=current_user.id,
            window_days=7
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="risk_overridden",
        target_type="patient",
        target_id=patient_id,
        details={
            "override_level": req.level,
            "override_reason": req.override_reason,
            "overridden_by": current_user.id
        }
    )
    return _risk_to_dict(record, db)


@router.get("/api/patients/{patient_id}/analytics/trends")
def get_analytics_trends(
    patient_id: str,
    days: int = 14,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = db.get(Patient, patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")
    _verify_patient_access(patient, current_user, db)

    days_clamped = max(3, min(60, days))
    trends = get_patient_analytics_trends(db, patient_id=patient_id, days=days_clamped)
    return trends
