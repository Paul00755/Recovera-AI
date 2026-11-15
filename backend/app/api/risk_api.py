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


@router.get("/api/patients/high-risk-overview")
def get_high_risk_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value)),
):
    """
    Dedicated clinical risk triage overview for Doctor and Nurse command dashboards.
    Aggregates population risk counters and ranks patients requiring attention.
    """
    import datetime as dt
    from app.models.checkin import DailyCheckIn
    from app.models.appointment import Appointment
    from app.models.digital_twin import DigitalTwinDeviationAlert
    from app.core.constants import AppointmentStatus

    # Query active patients
    query = select(Patient).where(Patient.active == True)
    if current_user.role == Role.DOCTOR.value:
        # If doctor, filter by assigned patients (or all if caseload is open)
        assigned_pids = db.scalars(
            select(PatientDoctorAssignment.patient_id).where(
                PatientDoctorAssignment.doctor_id == current_user.id
            )
        ).all()
        # Also include patients where assigned_doctor_id is current_user.id
        query = query.where(
            (Patient.assigned_doctor_id == current_user.id) | (Patient.id.in_(assigned_pids))
        )
    
    patients = db.scalars(query).all()

    summary_counters = {
        "total_active_patients": len(patients),
        "green_count": 0,
        "yellow_count": 0,
        "red_count": 0,
        "insufficient_data_count": 0,
        "needing_review_count": 0,
        "repeated_missed_meds_count": 0,
        "worsening_symptoms_count": 0,
        "upcoming_missed_followups_count": 0,
    }

    now = dt.datetime.utcnow()
    two_days_ahead = now + dt.timedelta(days=2)
    seven_days_ago = now - dt.timedelta(days=7)

    patient_rows = []

    for p in patients:
        # Latest risk record
        r = db.scalar(
            select(RiskClassificationHistory)
            .where(RiskClassificationHistory.patient_id == p.id)
            .order_by(RiskClassificationHistory.evaluated_at.desc())
        )
        if not r:
            try:
                r = evaluate_patient_risk(db, patient_id=p.id, window_days=7, force=False)
            except Exception:
                r = None

        lvl = (r.level if r else "insufficient_data").lower()
        if lvl == "green":
            summary_counters["green_count"] += 1
        elif lvl == "yellow":
            summary_counters["yellow_count"] += 1
            summary_counters["needing_review_count"] += 1
        elif lvl == "red":
            summary_counters["red_count"] += 1
            summary_counters["needing_review_count"] += 1
        else:
            summary_counters["insufficient_data_count"] += 1

        reasons = r.explainability_reasons if r else ["Telemetry data pending initial ingestion"]

        # Check for repeated missed meds
        has_repeated_missed = any("missed" in reason.lower() and ("consecutive" in reason.lower() or "doses" in reason.lower()) for reason in reasons)
        if has_repeated_missed:
            summary_counters["repeated_missed_meds_count"] += 1

        # Check for worsening symptoms or fever
        has_worsening = any("pain" in reason.lower() or "fever" in reason.lower() or "escalat" in reason.lower() for reason in reasons)
        if has_worsening:
            summary_counters["worsening_symptoms_count"] += 1

        # Check appointments
        appt = db.scalar(
            select(Appointment).where(
                Appointment.patient_id == p.id,
                Appointment.start_time >= seven_days_ago,
                Appointment.start_time <= two_days_ahead
            ).order_by(Appointment.start_time.desc()).limit(1)
        )
        has_appt_alert = False
        if appt:
            if appt.status == AppointmentStatus.MISSED.value:
                has_appt_alert = True
                summary_counters["upcoming_missed_followups_count"] += 1
            elif appt.status in (AppointmentStatus.SCHEDULED.value, AppointmentStatus.CONFIRMED.value) and appt.start_time <= two_days_ahead:
                has_appt_alert = True
                summary_counters["upcoming_missed_followups_count"] += 1

        # Check latest check-in
        latest_ci = db.scalar(
            select(DailyCheckIn)
            .where(DailyCheckIn.patient_id == p.id)
            .order_by(DailyCheckIn.checkin_date.desc())
            .limit(1)
        )
        last_checkin_date = latest_ci.checkin_date.isoformat() if latest_ci else None

        # Determine trend
        trend = "stable"
        if has_worsening or (r and r.level == "red"):
            trend = "deteriorating"
        elif r and r.level == "green":
            trend = "improving"

        # Recommended clinical action
        if lvl == "red":
            rec_action = "Urgent specialist review required. Check vitals, evaluate surgical site / infection markers, and initiate clinical outreach."
        elif lvl == "yellow":
            rec_action = "Nurse telemetry check-in recommended. Review medication adherence and verify symptom stability within 24 hours."
        elif lvl == "green":
            rec_action = "Routine post-discharge recovery pathway on track. Continue standard daily check-in monitoring."
        else:
            rec_action = "Insufficient telemetry. Send patient check-in prompt and verify app engagement."

        # Doctor name
        doc_name = "Unassigned"
        if p.assigned_doctor_id:
            d_user = db.get(User, p.assigned_doctor_id)
            if d_user:
                doc_name = d_user.name

        p_name = p.user.name if p.user else p.patient_code
        age = None
        if p.dob:
            today_d = dt.date.today()
            age = today_d.year - p.dob.year - ((today_d.month, today_d.day) < (p.dob.month, p.dob.day))

        patient_rows.append({
            "patient_id": p.id,
            "patient_code": p.patient_code or f"P-{p.id[:8]}",
            "name": p_name,
            "age": age,
            "gender": p.gender,
            "assigned_doctor": doc_name,
            "risk_level": lvl,
            "risk_score": r.risk_score if (r and hasattr(r, 'risk_score')) else (85 if lvl == "red" else (50 if lvl == "yellow" else 15)),
            "key_contributing_factors": reasons,
            "last_checkin_date": last_checkin_date,
            "trend_indicator": trend,
            "recommended_action": rec_action,
            "is_override": bool(r.is_override if r else False),
            "override_reason": r.override_reason if r else None,
            "has_repeated_missed_meds": has_repeated_missed,
            "has_worsening_symptoms": has_worsening,
            "has_followup_alert": has_appt_alert,
            "evaluated_at": r.evaluated_at.isoformat() if (r and r.evaluated_at) else None
        })

    # Sort: RED first, then YELLOW, then INSUFFICIENT_DATA, then GREEN
    priority_order = {"red": 0, "yellow": 1, "insufficient_data": 2, "green": 3}
    patient_rows.sort(key=lambda x: (priority_order.get(x["risk_level"], 4), -x["risk_score"]))

    return {
        "summary": summary_counters,
        "patients": patient_rows,
        "generated_at": dt.datetime.utcnow().isoformat()
    }

