"""
Recovera AI 2.0 — Patient Recovery Digital Twin Service
Deterministic state machine, trajectory comparisons, explainable deviation monitoring (D_t),
and evidence-linked clinician decision support.
"""
import uuid
import datetime as dt
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, desc

from app.models.digital_twin import (
    DigitalTwinState,
    DigitalTwinStateHistory,
    RecoveryTrajectoryMilestone,
    DigitalTwinPrediction,
    DigitalTwinDeviationAlert,
)
from app.models.patient import Patient
from app.models.recovery_plan import RecoveryPlan
from app.models.checkin import DailyCheckIn
from app.models.medication import MedicationOccurrence, Medication
from app.models.appointment import Appointment
from app.core.constants import OccurrenceState

def get_or_create_trajectory_milestones(
    db: Session,
    patient_id: str,
    start_date: Optional[dt.date] = None
) -> List[RecoveryTrajectoryMilestone]:
    """
    Ensures a 14-day clinician-approved expected recovery corridor exists for the patient.
    Defaults to evidence-based post-discharge recovery decline envelopes.
    """
    existing = db.scalars(
        select(RecoveryTrajectoryMilestone)
        .where(RecoveryTrajectoryMilestone.patient_id == patient_id)
        .order_by(RecoveryTrajectoryMilestone.day_number.asc())
    ).all()
    if existing:
        return list(existing)

    patient = db.get(Patient, patient_id)
    base_date = start_date or (patient.created_at.date() if patient and patient.created_at else dt.date.today())

    # Evidence-based recovery curve: gradual expected pain reduction over 14 days
    curve_archetype = [
        # (day, min_pain, max_pain, activity, note)
        (1, 5.0, 8.0, "rest", "Immediate post-discharge acute recovery; bed rest & analgesia"),
        (2, 5.0, 7.5, "rest", "Wound stabilization; initial mobilization"),
        (3, 4.0, 7.0, "light", "Light assisted walking; vital signs monitoring"),
        (4, 4.0, 6.5, "light", "Steady mobilization; monitor surgical dressing"),
        (5, 3.0, 6.0, "light", "Transition to oral analgesics; independent short walks"),
        (6, 3.0, 5.5, "light", "Consistent mobility without sudden pain spikes"),
        (7, 2.0, 5.0, "normal", "1-Week post-op milestone; clinical review recommended"),
        (8, 2.0, 4.5, "normal", "Routine household activity with rest intervals"),
        (9, 1.0, 4.0, "normal", "Improving joint flexibility / systemic stability"),
        (10, 1.0, 3.5, "normal", "Dressing removal or suture evaluation interval"),
        (11, 0.5, 3.0, "normal", "Independent ambulation; residual mild discomfort"),
        (12, 0.0, 2.5, "normal", "Near baseline physical capacity"),
        (13, 0.0, 2.0, "active", "Light exercise & stretching resumption"),
        (14, 0.0, 2.0, "active", "Formal recovery completion assessment milestone"),
    ]

    milestones = []
    for day, p_min, p_max, act, note in curve_archetype:
        m = RecoveryTrajectoryMilestone(
            id=str(uuid.uuid4()),
            patient_id=patient_id,
            day_number=day,
            target_date=base_date + dt.timedelta(days=day - 1),
            expected_pain_min=p_min,
            expected_pain_max=p_max,
            expected_activity=act,
            clinical_milestone_note=note,
            created_at=dt.datetime.utcnow(),
        )
        db.add(m)
        milestones.append(m)

    db.commit()
    return milestones


def compute_digital_twin_state(
    db: Session,
    patient_id: str,
    event_source: str = "telemetry_update",
    event_id: Optional[str] = None
) -> DigitalTwinState:
    """
    Deterministically computes the Patient Recovery Digital Twin state vector.
    Calculates S_t, A_t, F_t, T_t, Q_t and composite deviation D_t.
    """
    now = dt.datetime.utcnow()
    today_d = now.date()
    milestones = get_or_create_trajectory_milestones(db, patient_id)

    # Recovery pathway base date (Day 1) is anchored to milestones[0].target_date
    patient = db.get(Patient, patient_id)
    base_date = (
        milestones[0].target_date
        if (milestones and milestones[0].target_date)
        else (patient.created_at.date() if patient and patient.created_at else today_d)
    )
    end_date = (
        milestones[-1].target_date
        if (milestones and milestones[-1].target_date)
        else (base_date + dt.timedelta(days=13))
    )

    # Elapsed days in recovery pathway up to today (clamped between 1 and 14)
    days_elapsed = max(1, min(14, (today_d - base_date).days + 1))

    # 1. Fetch Plan Version
    latest_plan = db.scalar(
        select(RecoveryPlan)
        .where(RecoveryPlan.patient_id == patient_id)
        .order_by(RecoveryPlan.version.desc())
        .limit(1)
    )
    plan_ver = latest_plan.version if latest_plan else 1

    # 2. Fetch Check-ins for the patient's recovery trajectory
    checkins = db.scalars(
        select(DailyCheckIn)
        .where(
            DailyCheckIn.patient_id == patient_id,
            DailyCheckIn.checkin_date >= base_date,
            DailyCheckIn.checkin_date <= end_date
        )
        .order_by(DailyCheckIn.checkin_date.asc())
    ).all()
    checkin_map = {c.checkin_date: c for c in checkins}

    # 3. Fetch Medication Adherence for elapsed window (base_date to now)
    occurrences = db.scalars(
        select(MedicationOccurrence)
        .where(
            MedicationOccurrence.patient_id == patient_id,
            MedicationOccurrence.scheduled_at >= dt.datetime.combine(base_date, dt.time.min),
            MedicationOccurrence.scheduled_at <= now,
            MedicationOccurrence.state != OccurrenceState.CANCELLED.value
        )
    ).all()

    total_sched = len(occurrences)
    total_taken = sum(1 for o in occurrences if o.state == OccurrenceState.TAKEN.value)
    adherence_pct = (total_taken / total_sched * 100.0) if total_sched > 0 else 100.0

    # ----------------------------------------------------
    # Calculate 5 Deviation Dimensions
    # ----------------------------------------------------
    # S_t: Symptom Deviation (Exceeding Expected Milestone Range)
    symptom_divergences = []
    consecutive_rising_days = 0
    prev_pain = None
    acute_symptom_flags = []

    for m in milestones:
        # Only evaluate elapsed milestones up to today
        if m.target_date and m.target_date > today_d:
            continue

        c = checkin_map.get(m.target_date)
        if c and c.pain_level is not None:
            # Check if pain exceeds expected max
            if c.pain_level > m.expected_pain_max:
                excess = c.pain_level - m.expected_pain_max
                symptom_divergences.append(min(100.0, excess * 25.0))
            else:
                symptom_divergences.append(0.0)

            if c.fever:
                acute_symptom_flags.append(f"Fever reported on Day {m.day_number} ({c.checkin_date})")
            if c.cough:
                acute_symptom_flags.append(f"Respiratory cough reported on Day {m.day_number} ({c.checkin_date})")

            # Check consecutive rising pain (trend velocity)
            if prev_pain is not None:
                if c.pain_level > prev_pain:
                    consecutive_rising_days += 1
                elif c.pain_level < prev_pain:
                    consecutive_rising_days = max(0, consecutive_rising_days - 1)
            prev_pain = c.pain_level

    s_score = round((sum(symptom_divergences) / len(symptom_divergences)), 1) if symptom_divergences else 0.0

    # A_t: Adherence Deviation
    a_score = max(0.0, min(100.0, round(100.0 - adherence_pct, 1)))

    # F_t: Acute Vitals & Care Protocol Disruption (Fever / Acute Symptoms + Missed Appointments)
    appts = db.scalars(
        select(Appointment).where(
            Appointment.patient_id == patient_id,
            Appointment.start_time <= now
        )
    ).all()
    missed_appts = sum(1 for a in appts if a.status == "missed")
    acute_vital_penalty = 50.0 if acute_symptom_flags else 0.0
    f_score = min(100.0, round(acute_vital_penalty + (missed_appts * 35.0), 1))

    # T_t: Trajectory Velocity / Escalation Slope
    # Rising pain across consecutive submissions is a leading indicator of infection/complication
    t_score = min(100.0, round(consecutive_rising_days * 35.0, 1))

    # Q_t: Data Quality & Missingness (Check-in Completeness over elapsed recovery window)
    submitted_count = sum(1 for m in milestones if m.target_date and m.target_date <= today_d and m.target_date in checkin_map)
    missing_checkins = max(0, days_elapsed - submitted_count)
    q_score = round(min(100.0, (missing_checkins / max(1, days_elapsed)) * 100.0), 1)

    # Composite Deviation D_t
    # Weights: Symptom (35%), Adherence (25%), Trend (20%), Acute/Follow-up (10%), Missingness (10%)
    d_composite = round((0.35 * s_score) + (0.25 * a_score) + (0.20 * t_score) + (0.10 * f_score) + (0.10 * q_score), 1)

    # Assign Deviation Tier
    if d_composite >= 70.0 or (s_score >= 60.0 and t_score >= 60.0):
        tier = "critical_deviation"
    elif d_composite >= 45.0:
        tier = "moderate_deviation"
    elif d_composite >= 22.0:
        tier = "mild_deviation"
    else:
        tier = "normal"

    overall_recovery_score = max(0.0, min(100.0, round(100.0 - d_composite, 1)))

    # 4. Trajectory Serialization for interactive SVG visualizer
    expected_payload = [
        {
            "day": m.day_number,
            "date": m.target_date.isoformat() if m.target_date else None,
            "expected_min": m.expected_pain_min,
            "expected_max": m.expected_pain_max,
            "activity": m.expected_activity,
            "milestone_note": m.clinical_milestone_note,
        }
        for m in milestones
    ]

    observed_payload = []
    for m in milestones:
        curr_d = m.target_date
        c = checkin_map.get(curr_d) if curr_d else None
        is_future = (curr_d > today_d) if curr_d else False
        if c:
            observed_payload.append({
                "day": m.day_number,
                "date": curr_d.isoformat() if curr_d else None,
                "pain_level": c.pain_level,
                "pain_location": c.pain_location or "General",
                "fever": c.fever,
                "cough": c.cough,
                "sleep_hours": c.sleep_hours,
                "activity_level": c.activity_level,
                "submitted": True,
                "is_future": False,
            })
        else:
            observed_payload.append({
                "day": m.day_number,
                "date": curr_d.isoformat() if curr_d else None,
                "pain_level": None,
                "fever": False,
                "cough": False,
                "sleep_hours": None,
                "activity_level": None,
                "submitted": False,
                "is_future": is_future,
            })

    # Latest model forecast (if evaluated)
    latest_pred = db.scalar(
        select(DigitalTwinPrediction)
        .where(DigitalTwinPrediction.patient_id == patient_id)
        .order_by(DigitalTwinPrediction.created_at.desc())
        .limit(1)
    )
    forecast_dict = None
    if latest_pred:
        forecast_dict = {
            "model_name": latest_pred.model_name,
            "prediction_target": latest_pred.prediction_target,
            "predicted_value": latest_pred.predicted_value,
            "uncertainty_lower": latest_pred.uncertainty_lower,
            "uncertainty_upper": latest_pred.uncertainty_upper,
            "horizon_hours": latest_pred.prediction_horizon_hours,
            "limitations_note": latest_pred.limitations_note,
            "created_at": latest_pred.created_at.isoformat(),
        }

    # 5. Save or Update DigitalTwinState
    twin = db.scalar(select(DigitalTwinState).where(DigitalTwinState.patient_id == patient_id))
    if not twin:
        twin = DigitalTwinState(
            id=str(uuid.uuid4()),
            patient_id=patient_id,
            plan_version=plan_ver,
            observation_cutoff=now,
            computed_at=now,
            symptom_deviation_score=round(s_score, 1),
            adherence_deviation_score=round(a_score, 1),
            followup_deviation_score=round(f_score, 1),
            trajectory_slope_score=round(t_score, 1),
            uncertainty_score=round(q_score, 1),
            composite_deviation=d_composite,
            deviation_tier=tier,
            current_recovery_score=overall_recovery_score,
            expected_trajectory_json=expected_payload,
            observed_trajectory_json=observed_payload,
            model_forecast_json=forecast_dict,
            updater_version="v2.0-deterministic",
        )
        db.add(twin)
    else:
        twin.plan_version = plan_ver
        twin.observation_cutoff = now
        twin.computed_at = now
        twin.symptom_deviation_score = round(s_score, 1)
        twin.adherence_deviation_score = round(a_score, 1)
        twin.followup_deviation_score = round(f_score, 1)
        twin.trajectory_slope_score = round(t_score, 1)
        twin.uncertainty_score = round(q_score, 1)
        twin.composite_deviation = d_composite
        twin.deviation_tier = tier
        twin.current_recovery_score = overall_recovery_score
        twin.expected_trajectory_json = expected_payload
        twin.observed_trajectory_json = observed_payload
        twin.model_forecast_json = forecast_dict

    # 6. Record State History for Audit Trail
    hist = DigitalTwinStateHistory(
        id=str(uuid.uuid4()),
        patient_id=patient_id,
        event_type=event_source,
        source_event_id=event_id,
        plan_version=plan_ver,
        composite_deviation=d_composite,
        state_snapshot_json={
            "d_composite": d_composite,
            "tier": tier,
            "s_score": round(s_score, 1),
            "a_score": round(a_score, 1),
            "t_score": round(t_score, 1),
            "q_score": round(q_score, 1),
            "recovery_score": overall_recovery_score,
        },
        recorded_at=now,
    )
    db.add(hist)

    # 7. Generate Explainable Deviation Alert if deviation is moderate or critical
    if tier in ("moderate_deviation", "critical_deviation"):
        active_pending = db.scalar(
            select(DigitalTwinDeviationAlert).where(
                DigitalTwinDeviationAlert.patient_id == patient_id,
                DigitalTwinDeviationAlert.status == "pending"
            ).limit(1)
        )
        if not active_pending:
            triggers = []
            if s_score >= 35.0:
                triggers.append(f"Reported pain levels exceeding expected recovery corridor (S_t = {round(s_score)}%)")
            if t_score >= 35.0:
                triggers.append(f"Consecutive days of rising pain detected (T_t = {round(t_score)}%)")
            if a_score >= 40.0:
                triggers.append(f"Sub-optimal medication adherence: {round(adherence_pct)}% (A_t = {round(a_score)}%)")
            if acute_symptom_flags:
                triggers.extend(acute_symptom_flags)
            if not triggers:
                triggers.append(f"Composite recovery deviation elevated (D_t = {d_composite})")

            alert = DigitalTwinDeviationAlert(
                id=str(uuid.uuid4()),
                patient_id=patient_id,
                severity="severe" if tier == "critical_deviation" else "moderate",
                trigger_reason=" • ".join(triggers),
                trigger_evidence_json={
                    "d_composite": d_composite,
                    "tier": tier,
                    "s_score": round(s_score, 1),
                    "a_score": round(a_score, 1),
                    "t_score": round(t_score, 1),
                    "q_score": round(q_score, 1),
                    "consecutive_rising_days": consecutive_rising_days,
                    "recent_checkin_count": len(checkins),
                    "adherence_pct": round(adherence_pct, 1),
                },
                rule_or_model_version="D_t-rule-v2.0",
                status="pending",
                created_at=now,
            )
            db.add(alert)

    db.commit()
    db.refresh(twin)
    return twin
