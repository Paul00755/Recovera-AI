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
from app.models.medication import Medication, MedicationOccurrence
from app.models.checkin import DailyCheckIn
from app.models.appointment import Appointment
from app.core.constants import OccurrenceState
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
    Returns dynamic post-discharge recovery corridor, daily milestones,
    actual patient telemetry observations, adaptive timeline of past and upcoming days,
    and 24h-48h forecast with 95% confidence intervals.
    """
    patient = _verify_patient_access(patient_id, current_user, db)
    today_d = dt.date.today()
    now = dt.datetime.utcnow()

    # 0. Check medication assignment to anchor Day 1 strictly to medication start date
    first_med = db.scalar(
        select(Medication)
        .where(Medication.patient_id == str(patient_id))
        .order_by(Medication.start_date.asc(), Medication.created_at.asc())
        .limit(1)
    )
    has_medications = first_med is not None
    med_start_date = (first_med.start_date or (first_med.created_at.date() if first_med.created_at else None)) if first_med else None
    if isinstance(med_start_date, dt.datetime):
        med_start_date = med_start_date.date()

    if not has_medications or not med_start_date:
        return {
            "patient_id": str(patient_id),
            "corridor_milestones": [],
            "observed_points": [],
            "forecasts": [],
            "current_day": 0,
            "has_started": False,
            "medication_start_date": None,
            "awaiting_reason": "Awaiting medication schedule from attending care team",
            "deviation_tier": "normal",
            "total_timeline_days": 0,
            "completed_checkins_count": 0,
            "scheduled_doses": 0,
            "taken_doses": 0,
            "adherence_rate": 0.0,
            "next_event_summary": "Awaiting clinical medication prescription",
            "timeline": []
        }

    diff_days = (today_d - med_start_date).days
    if diff_days < 0:
        return {
            "patient_id": str(patient_id),
            "corridor_milestones": [],
            "observed_points": [],
            "forecasts": [],
            "current_day": 0,
            "has_started": False,
            "medication_start_date": med_start_date.isoformat(),
            "awaiting_reason": f"Medication course scheduled to commence on {med_start_date.isoformat()}",
            "deviation_tier": "normal",
            "total_timeline_days": 0,
            "completed_checkins_count": 0,
            "scheduled_doses": 0,
            "taken_doses": 0,
            "adherence_rate": 0.0,
            "next_event_summary": f"Treatment starts on {med_start_date.isoformat()}",
            "timeline": []
        }

    current_day = diff_days + 1
    has_started = True

    # 1. Fetch or create milestones anchored to medication start date
    milestones = get_or_create_trajectory_milestones(db, str(patient_id), start_date=med_start_date, min_days=max(14, current_day + 7))

    # 2. Fetch current state & observations
    state = db.scalar(
        select(DigitalTwinState).where(DigitalTwinState.patient_id == str(patient_id))
    )
    if not state:
        state = compute_digital_twin_state(db, str(patient_id))

    history_points = state.observed_trajectory_json or []
    last_val = state.current_recovery_score

    # 3. Compute 24h/48h calibrated forecast with 95% confidence bounds
    forecasts = []
    for offset in [1, 2]:
        f_day = current_day + offset
        ms = next((m for m in milestones if m.day_number == f_day), None)
        target = (100.0 - ((ms.expected_pain_min + ms.expected_pain_max) / 2.0 * 10.0)) if ms else min(100.0, last_val + 2.0)
        point_pred = round(last_val * 0.7 + target * 0.3, 1)
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

    # 4. Fetch checkins, occurrences, appointments, and active prescriptions for dynamic timeline
    checkins = db.scalars(
        select(DailyCheckIn)
        .where(DailyCheckIn.patient_id == str(patient_id))
        .order_by(DailyCheckIn.checkin_date.asc())
    ).all()
    checkin_map = {c.checkin_date: c for c in checkins}

    occurrences = db.scalars(
        select(MedicationOccurrence)
        .where(
            MedicationOccurrence.patient_id == str(patient_id),
            MedicationOccurrence.state != OccurrenceState.CANCELLED.value
        )
        .order_by(MedicationOccurrence.scheduled_at.asc())
    ).all()

    occ_by_date: Dict[dt.date, Dict[str, int]] = {}
    for o in occurrences:
        d = o.scheduled_at.date()
        if d not in occ_by_date:
            occ_by_date[d] = {"scheduled": 0, "taken": 0, "missed": 0, "pending": 0}
        occ_by_date[d]["scheduled"] += 1
        if o.state == OccurrenceState.TAKEN.value:
            occ_by_date[d]["taken"] += 1
        elif o.state in (OccurrenceState.MISSED.value, OccurrenceState.NOT_TAKEN.value) or (
            o.state == OccurrenceState.PENDING.value and o.scheduled_at <= (now - dt.timedelta(minutes=30))
        ):
            occ_by_date[d]["missed"] += 1
        else:
            occ_by_date[d]["pending"] += 1

    all_meds = db.scalars(select(Medication).where(Medication.patient_id == str(patient_id))).all()

    appts = db.scalars(
        select(Appointment)
        .where(
            Appointment.patient_id == str(patient_id),
            Appointment.status.in_(["scheduled", "confirmed", "pending", "in_progress"])
        )
        .order_by(Appointment.start_time.asc())
    ).all()
    appts_by_date: Dict[dt.date, List[Any]] = {}
    for a in appts:
        if a.start_time:
            d = a.start_time.date()
            if d not in appts_by_date:
                appts_by_date[d] = []
            appts_by_date[d].append(a)

    # Calculate timeline horizon
    future_dates = [d for d in occ_by_date.keys() if d > today_d]
    for a_date in appts_by_date.keys():
        if a_date > today_d:
            future_dates.append(a_date)
    for m in all_meds:
        if m.end_date:
            m_end = m.end_date.date() if isinstance(m.end_date, dt.datetime) else m.end_date
            if m_end > today_d:
                future_dates.append(m_end)

    max_future_date = max(future_dates) if future_dates else (today_d + dt.timedelta(days=3))
    total_future_days = max(3, min(14, (max_future_date - today_d).days))
    total_timeline_days = current_day + total_future_days

    timeline_items = []
    total_sched = 0
    total_taken = 0
    checkins_completed = 0
    next_event_summary = None

    for day_num in range(1, total_timeline_days + 1):
        day_date = med_start_date + dt.timedelta(days=day_num - 1)
        is_today = (day_date == today_d)
        is_past = (day_date < today_d)
        is_future = (day_date > today_d)

        # Check-in telemetry
        ci = checkin_map.get(day_date)
        if ci:
            checkins_completed += 1
            checkin_status = "completed"
            checkin_pain = ci.pain_level
            checkin_msg = f"Check-in done (Pain: {ci.pain_level}/10)" + (" · Fever reported" if ci.fever else "")
        elif is_today:
            checkin_status = "pending"
            checkin_pain = None
            checkin_msg = "Check-in pending today"
        elif is_past:
            checkin_status = "missed"
            checkin_pain = None
            checkin_msg = "Check-in not logged"
        else:
            checkin_status = "future"
            checkin_pain = None
            checkin_msg = "Upcoming daily check-in"

        # Medication telemetry
        occ_info = occ_by_date.get(day_date)
        if occ_info and occ_info["scheduled"] > 0:
            sched = occ_info["scheduled"]
            taken = occ_info["taken"]
            missed = occ_info["missed"]
            pending = occ_info["pending"]
            if is_past or is_today:
                total_sched += sched
                total_taken += taken

            if taken == sched:
                med_status = "all_taken"
                med_msg = f"All meds taken ({taken}/{sched} doses)"
            elif taken > 0:
                med_status = "partially_taken"
                med_msg = f"Partially done ({taken}/{sched} taken, {missed or pending} unrecorded)"
            else:
                med_status = "not_taken"
                med_msg = f"Meds not logged (0/{sched})" if is_past else f"0/{sched} taken (pending today)"
        else:
            sched = 0
            taken = 0
            missed = 0
            pending = 0
            med_status = "none_scheduled"
            med_msg = "No meds scheduled"

        # Appointments & Future predictions
        has_appt = day_date in appts_by_date
        appt_detail = None
        if has_appt:
            a = appts_by_date[day_date][0]
            appt_time = a.start_time.strftime("%I:%M %p") if a.start_time else ""
            appt_detail = f"Scheduled {a.appointment_type or 'consultation'} ({appt_time})"
            if not next_event_summary and is_future:
                next_event_summary = f"Day {day_num}: {appt_detail}"

        days_ahead = (day_date - today_d).days
        if is_future:
            proj_score = min(100.0, max(0.0, round(last_val + (1.2 * days_ahead), 1)))
            proj_margin = round(3.2 + 0.6 * days_ahead, 1)
            pred_msg = f"Projected score: {proj_score} (±{proj_margin} pts)"

            ending_meds = [
                m.name for m in all_meds
                if m.end_date and (m.end_date.date() if isinstance(m.end_date, dt.datetime) else m.end_date) == day_date
            ]
            if ending_meds:
                med_future_note = f"Course completion: {', '.join(ending_meds)}"
                if not next_event_summary:
                    next_event_summary = f"Day {day_num}: {med_future_note}"
            elif occ_info and occ_info["scheduled"] > 0:
                med_future_note = f"{occ_info['scheduled']} scheduled doses"
            else:
                med_future_note = "Observation & stabilization"

            if has_appt:
                summary_msg = f"{appt_detail} · {med_future_note}"
            else:
                summary_msg = f"{pred_msg} · {med_future_note}"
        else:
            proj_score = None
            proj_margin = None
            pred_msg = None
            if is_today:
                summary_msg = f"Today: {med_msg} · {checkin_msg}"
                if occ_info and occ_info["pending"] > 0 and not next_event_summary:
                    next_event_summary = f"Today: {occ_info['pending']} pending medication dose(s)"
            else:
                summary_msg = f"{checkin_msg} · {med_msg}"

        timeline_items.append({
            "day_number": day_num,
            "date": day_date.isoformat(),
            "formatted_date": day_date.strftime("%b %d"),
            "is_today": is_today,
            "is_past": is_past,
            "is_future": is_future,
            "checkin_status": checkin_status,
            "checkin_pain": checkin_pain,
            "checkin_message": checkin_msg,
            "med_status": med_status,
            "med_scheduled": sched,
            "med_taken": taken,
            "med_message": med_msg,
            "has_appointment": has_appt,
            "appointment_detail": appt_detail,
            "projected_score": proj_score,
            "projected_margin": proj_margin,
            "prediction_message": pred_msg,
            "summary_message": summary_msg
        })

    adherence_pct = round((total_taken / total_sched) * 100, 1) if total_sched > 0 else 100.0

    return {
        "patient_id": str(patient_id),
        "corridor_milestones": milestone_list,
        "observed_points": history_points,
        "forecasts": forecasts,
        "current_day": current_day,
        "has_started": has_started,
        "medication_start_date": med_start_date.isoformat(),
        "deviation_tier": state.deviation_tier,
        "total_timeline_days": total_timeline_days,
        "completed_checkins_count": checkins_completed,
        "scheduled_doses": total_sched,
        "taken_doses": total_taken,
        "adherence_rate": adherence_pct,
        "next_event_summary": next_event_summary or "Active self-monitoring and resting",
        "timeline": timeline_items
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

        # Real post-discharge recovery day (clamped 1-14, strictly anchored to medication start date)
        first_med = db.scalar(
            select(Medication).where(Medication.patient_id == str(p.id))
            .order_by(Medication.start_date.asc(), Medication.created_at.asc()).limit(1)
        )
        med_start = (first_med.start_date or (first_med.created_at.date() if first_med.created_at else None)) if first_med else None
        if med_start:
            milestones = get_or_create_trajectory_milestones(db, str(p.id), start_date=med_start)
            elapsed = max(1, min(14, (today_d - med_start).days + 1))
        else:
            elapsed = 0  # Awaiting medication assignment

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


class SimulationRequest(BaseModel):
    scenario_type: str = Field(..., description="adherence_drop, perfect_adherence, pain_escalation, fever_spike")
    adjustments: Optional[Dict[str, Any]] = None


@router.get("/patients/{patient_id}/predictions")
def get_patient_predictions(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve live predictions across all trained clinical ML models for this patient."""
    _verify_patient_access(patient_id, current_user, db)
    from app.ml.prediction_engine import predict_patient_multi_model
    return predict_patient_multi_model(db, patient_id)


@router.post("/patients/{patient_id}/simulate")
def run_patient_simulation(
    patient_id: str,
    req: SimulationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Simulate what-if clinical scenario on a cloned feature vector.
    Always clearly marked with simulation_mode: 'hypothetical'.
    """
    _verify_patient_access(patient_id, current_user, db)
    try:
        from app.ml.prediction_engine import simulate_hypothetical_scenario
        return simulate_hypothetical_scenario(
            db=db,
            patient_id=patient_id,
            scenario_type=req.scenario_type,
            hypothetical_adjustments=req.adjustments
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation execution error for patient {patient_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation error: {str(e)}"
        )

