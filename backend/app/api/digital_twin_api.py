"""
Recovera AI 2.0 – Patient Recovery Digital Twin API
Provides endpoints for digital twin state tracking, longitudinal trajectory corridor,
deviation reviews, and deep learning benchmark metrics.
"""
from typing import Optional, List, Any, Dict
import os
import json
import uuid
import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, desc, func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.constants import Role
from app.models.patient import Patient
from app.models.user import User
from app.models.digital_twin import (
    DigitalTwinState,
    DigitalTwinStateHistory,
    RecoveryTrajectoryMilestone,
    DigitalTwinPrediction,
    DigitalTwinDeviationAlert
)
from app.api.deps import get_current_user
from app.services.digital_twin_service import (
    compute_digital_twin_state,
    get_or_create_trajectory_milestones
)

router = APIRouter(prefix="/api/digital-twin", tags=["Digital Twin"])


class AlertReviewRequest(BaseModel):
    review_action: str = Field(..., description="acknowledged, dismissed, or escalated")
    review_notes: Optional[str] = Field(None, description="Clinical notes or rationale")


def _verify_patient_access(patient_id: str, current_user: User, db: Session):
    patient = db.get(Patient, str(patient_id))
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # If patient, verify identity
    if current_user.role == Role.PATIENT.value and patient.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied to this patient profile")
    return patient


@router.get("/patients/{patient_id}/state")
def get_patient_digital_twin_state(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve current digital twin state and component decomposition."""
    patient = _verify_patient_access(patient_id, current_user, db)
    
    state = db.scalar(
        select(DigitalTwinState).where(DigitalTwinState.patient_id == str(patient_id))
    )
    if not state:
        # Compute dynamically on first request
        state = compute_digital_twin_state(db, str(patient_id))

    # Check for active alerts
    alerts = db.scalars(
        select(DigitalTwinDeviationAlert)
        .where(
            DigitalTwinDeviationAlert.patient_id == str(patient_id),
            DigitalTwinDeviationAlert.status == "pending"
        )
        .order_by(desc(DigitalTwinDeviationAlert.created_at))
    ).all()

    return {
        "patient_id": str(patient_id),
        "patient_name": patient.user.name if patient.user else patient.patient_code,
        "care_status": patient.care_status,
        "deviation_score": round(state.composite_deviation, 4),
        "deviation_tier": state.deviation_tier,
        "component_scores": {
            "symptom_deviation_st": round(state.symptom_deviation_score, 4),
            "adherence_penalty_at": round(state.adherence_deviation_score, 4),
            "consistency_penalty_tt": round(state.trajectory_slope_score, 4),
            "fever_vital_flag_ft": round(state.followup_deviation_score, 4),
            "sentiment_penalty_qt": round(state.uncertainty_score, 4),
        },
        "recovery_score": round(state.current_recovery_score, 1),
        "weights": {
            "symptom_weight": 0.35,
            "adherence_weight": 0.25,
            "consistency_weight": 0.20,
            "vital_flag_weight": 0.10,
            "sentiment_weight": 0.10
        },
        "formula": "D_t = 0.35*S_t + 0.25*A_t + 0.20*T_t + 0.10*F_t + 0.10*Q_t",
        "historical_trajectory": state.observed_trajectory_json or [],
        "last_calculated_at": state.computed_at.isoformat() if state.computed_at else None,
        "pending_alerts_count": len(alerts)
    }


@router.post("/patients/{patient_id}/recompute")
def recompute_patient_state(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Force re-computation of digital twin state and corridor trajectory."""
    patient = _verify_patient_access(patient_id, current_user, db)
    updated_state = compute_digital_twin_state(db, str(patient_id))
    return {
        "status": "success",
        "deviation_score": round(updated_state.composite_deviation, 4),
        "deviation_tier": updated_state.deviation_tier,
        "recovery_score": round(updated_state.current_recovery_score, 1),
        "last_calculated_at": updated_state.computed_at.isoformat() if updated_state.computed_at else None
    }


@router.get("/patients/{patient_id}/trajectory")
def get_patient_trajectory_corridor(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns 14-day expected clinical corridor [Y_min, Y_max], daily milestones,
    actual patient telemetry observations, and 24h-48h forecast with 95% confidence intervals.
    """
    patient = _verify_patient_access(patient_id, current_user, db)
    
    # 1. Fetch or create milestones
    milestones = get_or_create_trajectory_milestones(db, str(patient_id))
    
    # 2. Fetch current state & observations
    state = db.scalar(
        select(DigitalTwinState).where(DigitalTwinState.patient_id == str(patient_id))
    )
    if not state:
        state = compute_digital_twin_state(db, str(patient_id))
    
    history_points = state.observed_trajectory_json or []

    # 3. Compute 24h/48h calibrated forecast with 95% confidence bounds
    today_d = dt.date.today()
    base_date = (
        milestones[0].target_date
        if (milestones and milestones[0].target_date)
        else today_d
    )
    current_day = max(1, min(14, (today_d - base_date).days + 1))
    last_val = state.current_recovery_score

    forecasts = []
    # Project day + 1 and day + 2 from current recovery day
    for offset in [1, 2]:
        f_day = min(14, current_day + offset)
        ms = next((m for m in milestones if m.day_number == f_day), None)
        target = (100.0 - ((ms.expected_pain_min + ms.expected_pain_max) / 2.0 * 10.0)) if ms else min(100.0, last_val + 2.0)
        # Point prediction moves smoothly toward corridor target
        point_pred = round(last_val * 0.7 + target * 0.3, 1)
        # Calibrated residual standard error sigma = 4.2
        sigma = 4.2 * (1.0 + 0.2 * offset)
        lower_95 = round(max(0.0, point_pred - 1.96 * sigma), 1)
        upper_95 = round(min(100.0, point_pred + 1.96 * sigma), 1)
        
        forecasts.append({
            "horizon_hours": offset * 24,
            "day_number": f_day,
            "point_forecast": point_pred,
            "confidence_lower_95": lower_95,
            "confidence_upper_95": upper_95,
            "model_architecture": "Clinical Corridor Projection",
            "disclaimer": "Experimental decision-support forecast. Does not constitute autonomous medical judgment."
        })

    milestone_list = []
    for m in milestones:
        c_min = round(max(0.0, 100.0 - m.expected_pain_max * 10.0), 1)
        c_max = round(min(100.0, 100.0 - m.expected_pain_min * 10.0), 1)
        t_score = round((c_min + c_max) / 2.0, 1)
        milestone_list.append({
            "day_number": m.day_number,
            "target_date": m.target_date.isoformat() if m.target_date else None,
            "target_recovery_score": t_score,
            "expected_pain_min": m.expected_pain_min,
            "expected_pain_max": m.expected_pain_max,
            "corridor_score_min": c_min,
            "corridor_score_max": c_max,
            "expected_activity": m.expected_activity,
            "milestone_description": m.clinical_milestone_note or f"Day {m.day_number} Recovery Target"
        })

    return {
        "patient_id": str(patient_id),
        "corridor_milestones": milestone_list,
        "observed_points": history_points,
        "forecasts": forecasts,
        "current_day": current_day,
        "deviation_tier": state.deviation_tier
    }


@router.get("/patients/{patient_id}/alerts")
def get_patient_deviation_alerts(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve all digital twin deviation alerts for patient."""
    _verify_patient_access(patient_id, current_user, db)
    
    alerts = db.scalars(
        select(DigitalTwinDeviationAlert)
        .where(DigitalTwinDeviationAlert.patient_id == str(patient_id))
        .order_by(desc(DigitalTwinDeviationAlert.created_at))
    ).all()

    result = []
    for a in alerts:
        reviewer_name = None
        if a.reviewed_by:
            reviewer = db.get(User, a.reviewed_by)
            if reviewer:
                reviewer_name = reviewer.name

        dev_score = 0.0
        if a.trigger_evidence_json and isinstance(a.trigger_evidence_json, dict):
            dev_score = float(a.trigger_evidence_json.get("composite_deviation", a.trigger_evidence_json.get("deviation_score", 0.0)))

        result.append({
            "id": str(a.id),
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "severity": a.severity,
            "trigger_reason": a.trigger_reason,
            "trigger_evidence": a.trigger_evidence_json or {},
            "deviation_score": round(dev_score, 1),
            "status": a.status,
            "reviewed_by": reviewer_name,
            "clinician_notes": a.clinician_notes,
            "reviewed_at": a.reviewed_at.isoformat() if a.reviewed_at else None
        })
    return result


@router.post("/alerts/{alert_id}/review")
def review_deviation_alert(
    alert_id: str,
    body: AlertReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Clinician review of digital twin deviation flag."""
    if current_user.role not in [Role.DOCTOR.value, Role.NURSE.value, Role.ADMIN.value]:
        raise HTTPException(status_code=403, detail="Only clinical staff can review deviation alerts")
    
    alert = db.get(DigitalTwinDeviationAlert, str(alert_id))
    if not alert:
        raise HTTPException(status_code=404, detail="Deviation alert not found")

    if body.review_action not in ["acknowledged", "dismissed", "escalated"]:
        raise HTTPException(status_code=400, detail="Invalid review action. Must be acknowledged, dismissed, or escalated.")

    alert.status = body.review_action
    alert.reviewed_by = current_user.id
    alert.clinician_notes = body.review_notes
    alert.reviewed_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(alert)

    return {
        "status": "success",
        "alert_id": str(alert.id),
        "review_status": alert.status,
        "reviewed_by": current_user.name,
        "reviewed_at": alert.reviewed_at.isoformat()
    }


@router.get("/benchmark/metrics")
def get_benchmark_metrics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns research benchmark metrics across all 6 model architectures
    (Persistence, Historical Mean, Random Forest, GRU, LSTM, Temporal CNN)
    and synthetic longitudinal cohort metadata.
    """
    json_path = os.path.join(os.path.dirname(__file__), "..", "ml", "benchmark_results.json")
    if not os.path.exists(json_path):
        from app.ml.benchmark_service import run_benchmark_suite
        run_benchmark_suite()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


@router.get("/hospital-cohort-summary")
def get_hospital_digital_twin_cohort_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns REAL hospital-wide post-discharge recovery telemetry:
    - Active monitored patients from database
    - Cohort recovery score distribution (Optimal, Mild, Moderate, Critical)
    - Real medication adherence rates
    - Real clinical deviation alerts
    - Real patient trajectory summaries
    """
    patients = db.scalars(
        select(Patient).order_by(Patient.created_at.desc())
    ).all()

    cohort = []
    total_score = 0.0
    adherence_sum = 0.0
    adherence_count = 0
    tier_counts = {"normal": 0, "mild_deviation": 0, "moderate_deviation": 0, "critical_disruption": 0}

    today_d = dt.date.today()

    for p in patients:
        twin = db.scalar(select(DigitalTwinState).where(DigitalTwinState.patient_id == str(p.id)))
        if not twin:
            twin = compute_digital_twin_state(db, str(p.id))

        doctor_name = "Assigned Specialist"
        doctor_spec = "General Medicine"
        if p.assigned_doctor_id:
            doc = db.get(User, p.assigned_doctor_id)
            if doc:
                doctor_name = doc.name
                doctor_spec = getattr(doc, "specialization", "General Medicine") or "General Medicine"

        # Real post-discharge day (clamped 1-14)
        milestones = get_or_create_trajectory_milestones(db, str(p.id))
        base_date = milestones[0].target_date if (milestones and milestones[0].target_date) else p.created_at.date()
        elapsed = max(1, min(14, (today_d - base_date).days + 1))

        tier_key = twin.deviation_tier if twin.deviation_tier in tier_counts else "normal"
        tier_counts[tier_key] += 1
        total_score += twin.current_recovery_score

        adherence_gap = getattr(twin, "adherence_deviation_score", 0.0) or 0.0
        adherence_val = max(0.0, 100.0 - float(adherence_gap))
        adherence_sum += adherence_val
        adherence_count += 1

        rec_score = getattr(twin, "current_recovery_score", 100.0) or 100.0
        comp_dev = getattr(twin, "composite_deviation", 0.0) or 0.0
        sym_score = getattr(twin, "symptom_deviation_score", 0.0) or 0.0
        mis_score = getattr(twin, "uncertainty_score", 0.0) or 0.0
        acute_flag = bool((getattr(twin, "followup_deviation_score", 0.0) or 0.0) > 0)

        p_name = p.user.name if (p.user and p.user.name) else (p.patient_code or f"Patient {str(p.id)[:8]}")
        p_age = None
        if p.dob:
            p_age = today_d.year - p.dob.year - ((today_d.month, today_d.day) < (p.dob.month, p.dob.day))

        cohort.append({
            "id": str(p.id),
            "patient_code": p.patient_code or f"P-{str(p.id)[:8]}",
            "name": p_name,
            "age": p_age,
            "gender": p.gender,
            "doctor_name": doctor_name,
            "specialization": doctor_spec,
            "elapsed_day": elapsed,
            "recovery_score": round(float(rec_score), 1),
            "composite_deviation": round(float(comp_dev), 1),
            "deviation_tier": twin.deviation_tier or "normal",
            "symptom_score": round(float(sym_score), 1),
            "adherence_score": round(float(adherence_gap), 1),
            "missingness_score": round(float(mis_score), 1),
            "acute_flag": acute_flag
        })

    avg_score = round(total_score / max(1, len(patients)), 1)
    avg_adherence = round(adherence_sum / max(1, adherence_count), 1)

    unreviewed_alerts = db.scalar(
        select(func.count(DigitalTwinDeviationAlert.id)).where(DigitalTwinDeviationAlert.status == "pending")
    ) or 0

    return {
        "total_patients": len(patients),
        "average_recovery_score": avg_score,
        "average_adherence": avg_adherence,
        "unreviewed_alerts": unreviewed_alerts,
        "tier_distribution": tier_counts,
        "patients": cohort
    }
