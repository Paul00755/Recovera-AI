"""
Recovera AI – Explainable Clinical Risk Engine Service (Phase 12)

Computes automated risk classification (Green, Yellow, Red, Insufficient Data)
with granular clinical explainability reasons, supports manual clinician override
with mandatory clinical rationale, and aggregates longitudinal analytics trends.
"""
import datetime as dt
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, func

from app.core.config import settings
from app.core.constants import RiskLevel, OccurrenceState, AppointmentStatus, SMSCategory
from app.models.risk_history import RiskClassificationHistory
from app.models.medication import Medication, MedicationOccurrence
from app.models.recovery_plan import RecoveryPlan
from app.models.checkin import DailyCheckIn
from app.models.appointment import Appointment
from app.models.patient import Patient
from app.models.user import User
from app.services.notification_service import create_in_app_notification, send_sms_job


def evaluate_patient_risk(
    db: Session,
    patient_id: str,
    window_days: int = 7,
    force: bool = False
) -> RiskClassificationHistory:
    """
    Evaluates patient risk over `window_days` using adherence %, symptom check-ins,
    consecutive missed doses, and appointment attendance.
    """
    patient = db.get(Patient, patient_id)
    if not patient:
        raise ValueError(f"Patient {patient_id} not found.")

    # If force is False, check if there is an active clinician override in the last 24h
    if not force:
        latest_override = db.scalar(
            select(RiskClassificationHistory)
            .where(
                RiskClassificationHistory.patient_id == patient_id,
                RiskClassificationHistory.is_override == True
            )
            .order_by(RiskClassificationHistory.evaluated_at.desc())
        )
        if latest_override:
            # Overrides remain active until explicitly cleared or re-evaluated with force=True
            return latest_override

    now = dt.datetime.utcnow()
    window_start = now - dt.timedelta(days=window_days)
    grace_cutoff = now - dt.timedelta(minutes=30)

    # 1. Medication Occurrences in Window (excluding cancelled/voided doses and unrecorded doses of discontinued medications)
    active_meds_subquery = select(Medication.id).where(
        Medication.patient_id == patient_id,
        Medication.status == "active"
    )
    occurrences = db.scalars(
        select(MedicationOccurrence)
        .where(
            MedicationOccurrence.patient_id == patient_id,
            MedicationOccurrence.scheduled_at >= window_start,
            MedicationOccurrence.scheduled_at <= now,
            MedicationOccurrence.state != OccurrenceState.CANCELLED.value,
            or_(
                MedicationOccurrence.state == OccurrenceState.TAKEN.value,
                MedicationOccurrence.medication_id.in_(active_meds_subquery)
            )
        )
        .order_by(MedicationOccurrence.scheduled_at.desc())
    ).all()

    def is_missed_or_unlogged(o: MedicationOccurrence) -> bool:
        if o.state in (OccurrenceState.MISSED.value, OccurrenceState.NOT_TAKEN.value):
            return True
        if o.state == OccurrenceState.PENDING.value and o.scheduled_at <= grace_cutoff:
            return True
        return False

    total_occurrences = len(occurrences)
    taken_count = sum(1 for o in occurrences if o.state == OccurrenceState.TAKEN.value)
    missed_count = sum(1 for o in occurrences if is_missed_or_unlogged(o))

    # Consecutive missed doses from most recent
    consecutive_missed = 0
    for o in occurrences:
        if is_missed_or_unlogged(o):
            consecutive_missed += 1
        elif o.state == OccurrenceState.TAKEN.value:
            break

    adherence_pct = round((taken_count / total_occurrences) * 100, 1) if total_occurrences > 0 else None

    # 2. Daily Symptom Check-ins in Window
    checkins = db.scalars(
        select(DailyCheckIn)
        .where(
            DailyCheckIn.patient_id == patient_id,
            DailyCheckIn.checkin_date >= window_start.date()
        )
        .order_by(DailyCheckIn.checkin_date.desc())
    ).all()

    checkin_count = len(checkins)
    max_pain = max((c.pain_level for c in checkins), default=None)
    latest_checkin = checkins[0] if checkins else None
    has_fever = any(c.fever for c in checkins)
    has_cough = any(c.cough for c in checkins)

    # 3. Missed Appointments in Window
    missed_appointments = db.scalar(
        select(func.count(Appointment.id)).where(
            Appointment.patient_id == patient_id,
            Appointment.status == AppointmentStatus.MISSED.value,
            Appointment.start_time >= window_start
        )
    ) or 0

    # ----------------------------------------------------
    # 4. Multi-Pillar Clinical Risk Stratification
    # ----------------------------------------------------
    reasons: List[str] = []
    level: str
    insufficient_data: bool = False

    # PILLAR 1: Medication Intake & Adherence Telemetry
    med_red: List[str] = []
    med_yellow: List[str] = []
    if total_occurrences > 0:
        if adherence_pct is not None and adherence_pct < 50.0:
            med_red.append(f"Critical medication adherence: {adherence_pct}% (below 50% safety threshold)")
        if consecutive_missed >= 3:
            med_red.append(f"High risk: {consecutive_missed} consecutive unlogged or missed medication doses detected")
        if not med_red:
            if adherence_pct is not None and adherence_pct < 75.0:
                med_yellow.append(f"Sub-optimal medication adherence: {adherence_pct}% (expected: >= 75%)")
            elif missed_count >= 1:
                med_yellow.append(f"{missed_count} unlogged or missed medication dose(s) in the past week")

    # PILLAR 2: Clinical Appointments
    appt_red: List[str] = []
    appt_yellow: List[str] = []
    if missed_appointments >= 2:
        appt_red.append(f"Patient has {missed_appointments} missed clinical follow-up appointments")
    elif missed_appointments == 1:
        if (max_pain is not None and max_pain >= 6) or has_fever:
            appt_red.append("Patient missed clinical follow-up appointment while experiencing elevated pain or fever")
        else:
            appt_yellow.append("Patient missed 1 scheduled clinical follow-up appointment")

    # PILLAR 3: Daily Check-in Compliance (Frequency & Missing Telemetry)
    checkin_comp_red: List[str] = []
    checkin_comp_yellow: List[str] = []
    has_active_meds = db.scalar(
        select(func.count(Medication.id)).where(Medication.patient_id == patient_id, Medication.status == "active")
    ) or 0
    has_active_plan = db.scalar(
        select(func.count(RecoveryPlan.id)).where(RecoveryPlan.patient_id == patient_id, RecoveryPlan.status.in_(["published", "active"]))
    ) or 0
    is_active_monitoring = (patient.care_status != "idle") and (has_active_meds > 0 or has_active_plan > 0)

    checkin_dates_set = {c.checkin_date for c in checkins}
    today_date = dt.date.today()

    # Determine when patient monitoring actually began to avoid falsely flagging new patients
    first_med_date = db.scalar(select(func.min(Medication.start_date)).where(Medication.patient_id == patient_id))
    created_date = patient.created_at.date() if patient.created_at else today_date
    protocol_start_date = min(first_med_date, created_date) if first_med_date else created_date
    days_active = max(0, (today_date - protocol_start_date).days)

    if is_active_monitoring and days_active > 1:
        consecutive_missing_checkins = 0
        max_lookback = min(window_days, days_active)
        for day_i in range(1, max_lookback + 1):
            d = today_date - dt.timedelta(days=day_i)
            if d < protocol_start_date:
                break
            if d not in checkin_dates_set:
                consecutive_missing_checkins += 1
            else:
                break

        # If patient has already checked in today, daily check-in is complete for today
        if today_date not in checkin_dates_set:
            # Only escalate to RED if persistent blackout (>= 4 days) AND concurrent symptoms or missed meds
            if consecutive_missing_checkins >= 4 and (has_fever or (max_pain is not None and max_pain >= 6) or missed_count >= 2):
                checkin_comp_red.append(
                    f"Critical telemetry blackout: No daily symptom check-ins submitted for {consecutive_missing_checkins} consecutive days alongside elevated risk symptoms"
                )
            elif consecutive_missing_checkins >= 2:
                checkin_comp_yellow.append(
                    f"Inconsistent daily monitoring: {consecutive_missing_checkins} day(s) missing daily telemetry check-ins"
                )

    # PILLAR 4: Daily Check-in Reports (Symptoms, Pain, Vitals & Notes)
    symptom_red: List[str] = []
    symptom_yellow: List[str] = []

    if checkins:
        if max_pain is not None:
            loc = f" at {latest_checkin.pain_location}" if latest_checkin and latest_checkin.pain_location else ""
            if max_pain >= 8:
                symptom_red.append(f"Severe pain reported: {max_pain}/10 on visual analog scale{loc}")
            elif max_pain >= 5:
                symptom_yellow.append(f"Elevated pain reported: {max_pain}/10{loc}")

        if latest_checkin and latest_checkin.pain_location:
            locs = [p.strip() for p in latest_checkin.pain_location.split(",") if p.strip()]
            if len(locs) >= 2:
                symptom_yellow.append(f"Multiple anatomical areas of discomfort reported: {', '.join(locs)}")

        if len(checkins) >= 3:
            recent_pains = [c.pain_level for c in checkins[:3] if c.pain_level is not None]
            if len(recent_pains) >= 3:
                if recent_pains[0] >= 6 and (recent_pains[0] - recent_pains[-1]) >= 3:
                    symptom_red.append(
                        f"Deteriorating pain trajectory: Pain escalated from {recent_pains[-1]}/10 to {recent_pains[0]}/10"
                    )

        max_temp = max((c.temperature for c in checkins if getattr(c, 'temperature', None) is not None), default=None)
        if has_fever or (max_temp and max_temp >= 100.4):
            if (max_temp and max_temp >= 101.5) or (max_pain is not None and max_pain >= 6):
                symptom_red.append("Clinical red flag: Persistent/high fever reported concurrently with elevated pain")
            else:
                symptom_yellow.append(f"Fever / elevated body temperature ({max_temp or '100.4+'}°F) reported during check-in")

        if has_cough:
            symptom_yellow.append("Respiratory cough symptom reported in daily check-in")

        min_sleep = min((c.sleep_hours for c in checkins if c.sleep_hours is not None), default=None)
        if min_sleep is not None and min_sleep < 4.0:
            symptom_yellow.append(f"Severe sleep deprivation reported ({min_sleep} hours)")

        has_restricted_mobility = any(c.activity_level in ("low", "rest", "bed_rest") for c in checkins[:2])
        if has_restricted_mobility and max_pain and max_pain >= 6:
            symptom_yellow.append("Severely restricted mobility and bed rest reported due to acute discomfort")

        CRITICAL_NOTE_KEYWORDS = [
            "bleeding", "pus", "discharge", "infection", "wound open", "wound dehiscence",
            "chest pain", "shortness of breath", "dyspnea", "fainting", "syncope", "cannot breathe"
        ]
        MODERATE_NOTE_KEYWORDS = [
            "swelling", "edema", "numbness", "tingling", "dizziness", "nausea", "vomiting", "redness", "rash"
        ]

        found_critical_kws = set()
        found_mod_kws = set()
        for c in checkins:
            notes_lower = (c.notes or "").lower()
            for kw in CRITICAL_NOTE_KEYWORDS:
                if kw in notes_lower:
                    found_critical_kws.add(kw)
            for kw in MODERATE_NOTE_KEYWORDS:
                if kw in notes_lower:
                    found_mod_kws.add(kw)

        if found_critical_kws:
            symptom_red.append(f"Clinical warning flag in patient notes: '{', '.join(sorted(found_critical_kws))}' reported")
        elif found_mod_kws:
            symptom_yellow.append(f"Clinical symptom observation in notes: '{', '.join(sorted(found_mod_kws))}' reported")

    # Factor in Active Clinical Disease / Condition
    if getattr(patient, "active_condition", None):
        cond = patient.active_condition
        stage = getattr(patient, "condition_stage", "active") or "active"
        if stage in ("acute", "prior_state"):
            symptom_yellow.append(f"Diagnosed clinical condition: {cond} (Acute Phase)")
        elif stage == "recovering":
            symptom_yellow.append(f"Clinical recovery in progress: {cond}")

    # ----------------------------------------------------
    # SYNTHESIS & RISK CLASSIFICATION
    # ----------------------------------------------------
    all_red_triggers = med_red + appt_red + checkin_comp_red + symptom_red
    all_yellow_triggers = med_yellow + appt_yellow + checkin_comp_yellow + symptom_yellow

    if total_occurrences == 0 and checkin_count == 0 and not is_active_monitoring:
        level = RiskLevel.INSUFFICIENT_DATA.value
        insufficient_data = True
        reasons.append("Insufficient clinical data: No medication occurrences or symptom check-ins recorded in the 7-day window.")
    elif all_red_triggers:
        level = RiskLevel.RED.value
        reasons.extend(all_red_triggers)
        if all_yellow_triggers:
            reasons.extend(all_yellow_triggers[:2])
    elif len(all_yellow_triggers) >= 4 and len({
        ('med' if med_yellow else None),
        ('appt' if appt_yellow else None),
        ('comp' if checkin_comp_yellow else None),
        ('symp' if symptom_yellow else None)
    } - {None}) >= 3 and ((max_pain is not None and max_pain >= 6) or (adherence_pct is not None and adherence_pct < 75.0)):
        level = RiskLevel.RED.value
        reasons.append("Multi-pillar risk escalation: Concurrently experiencing unlogged medication, missing check-ins, and elevated symptom alerts")
        reasons.extend(all_yellow_triggers)
    elif all_yellow_triggers:
        level = RiskLevel.YELLOW.value
        reasons.extend(all_yellow_triggers)
    else:
        level = RiskLevel.GREEN.value
        if total_occurrences > 0:
            reasons.append(f"Optimal medication adherence: {adherence_pct or 100}%")
        else:
            reasons.append("No active medications missed")
        if checkin_count > 0:
            reasons.append(f"Daily symptom check-in compliance maintained ({checkin_count} reports)")
            if max_pain is not None:
                reasons.append(f"Pain well-controlled: Max {max_pain}/10")
            if not has_fever and not has_cough:
                reasons.append("Stable vitals: No fever or respiratory symptoms reported")
        else:
            reasons.append("Clinical condition stable")
        reasons.append("Follow-up appointment attendance compliant")

    # 5. Persist Evaluation in RiskClassificationHistory
    record = RiskClassificationHistory(
        patient_id=patient_id,
        level=level,
        explainability_reasons=reasons,
        insufficient_data=insufficient_data,
        data_window_days=window_days,
        is_override=False,
        evaluated_at=now
    )
    db.add(record)
    db.commit()

    # If categorized as RED, notify the assigned doctor via in-app alert + SMS Gateway
    if level == RiskLevel.RED.value and patient.assigned_doctor_id:
        try:
            doctor = db.get(User, patient.assigned_doctor_id)
            if doctor and doctor.active:
                patient_user = db.get(User, patient.user_id) if patient.user_id else None
                patient_name = patient_user.name if patient_user else patient.patient_code
                primary_reason = reasons[0] if reasons else "Elevated risk symptoms reported"

                # 1. In-app notification to doctor
                create_in_app_notification(
                    db=db,
                    user_id=doctor.id,
                    title=f"Critical Alert: Patient {patient_name} Classified RED",
                    message=f"Patient {patient_name} ({patient.patient_code}) has been classified as RED (High Risk). Reason: {primary_reason}. Immediate review recommended.",
                    category="alert",
                    link=f"#/doctor/patients/{patient.id}"
                )

                # 2. SMS Gateway alert to doctor
                if doctor.phone:
                    send_sms_job(
                        db=db,
                        phone=doctor.phone,
                        message_text=f"RecoveraAI Urgent: Patient {patient_name} ({patient.patient_code}) is categorized as RED (High Risk). Reason: {primary_reason}. Review chart: {settings.FRONTEND_URL.rstrip('/')}/#/doctor/patients/{patient.id}",
                        category=SMSCategory.TRACKER_FOLLOWUP.value,
                        patient_id=patient.id,
                        idempotency_key=f"red_alert_{patient.id}_{now.strftime('%Y%m%d_%H')}"
                    )
        except Exception as e:
            import logging
            logging.getLogger("recovera.risk").warning("Failed to dispatch RED alert to doctor: %s", e)

    return record


def override_patient_risk(
    db: Session,
    patient_id: str,
    level: str,
    override_reason: str,
    doctor_id: str,
    window_days: int = 7
) -> RiskClassificationHistory:
    """Clinician manual override with mandatory documented clinical rationale."""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise ValueError(f"Patient {patient_id} not found.")

    valid_levels = {r.value for r in RiskLevel}
    if level not in valid_levels:
        raise ValueError(f"Invalid risk level '{level}'. Must be one of: {', '.join(valid_levels)}.")

    doctor = db.get(User, doctor_id)
    doctor_name = doctor.name if doctor else "Attending Clinician"

    now = dt.datetime.utcnow()
    reasons = [
        f"[Clinician Override] Manually adjusted to {level.upper()} by Dr. {doctor_name}.",
        f"Clinical Rationale: {override_reason.strip()}"
    ]

    record = RiskClassificationHistory(
        patient_id=patient_id,
        level=level,
        explainability_reasons=reasons,
        insufficient_data=False,
        data_window_days=window_days,
        is_override=True,
        overridden_by=doctor_id,
        override_reason=override_reason.strip(),
        evaluated_at=now
    )
    db.add(record)
    db.commit()
    return record


def get_patient_analytics_trends(
    db: Session,
    patient_id: str,
    days: int = 14
) -> Dict[str, Any]:
    """Generates daily time-series data for medication adherence and pain/symptom trajectory."""
    now = dt.datetime.utcnow()
    start_date = (now - dt.timedelta(days=days - 1)).date()

    # 1. Fetch occurrences in past `days`
    active_meds_sub = select(Medication.id).where(
        Medication.patient_id == patient_id,
        Medication.status == "active"
    )
    occurrences = db.scalars(
        select(MedicationOccurrence)
        .where(
            MedicationOccurrence.patient_id == patient_id,
            MedicationOccurrence.scheduled_at >= dt.datetime.combine(start_date, dt.time.min),
            MedicationOccurrence.scheduled_at <= now,
            MedicationOccurrence.state != OccurrenceState.CANCELLED.value,
            or_(
                MedicationOccurrence.state == OccurrenceState.TAKEN.value,
                MedicationOccurrence.medication_id.in_(active_meds_sub)
            )
        )
    ).all()

    # Group occurrences by date
    occ_by_date: Dict[str, Dict[str, int]] = {}
    for i in range(days):
        d_str = (start_date + dt.timedelta(days=i)).isoformat()
        occ_by_date[d_str] = {"scheduled": 0, "taken": 0, "missed": 0}

    for o in occurrences:
        d_str = o.scheduled_at.date().isoformat()
        if d_str in occ_by_date:
            occ_by_date[d_str]["scheduled"] += 1
            if o.state == OccurrenceState.TAKEN.value:
                occ_by_date[d_str]["taken"] += 1
            elif o.state in (OccurrenceState.MISSED.value, OccurrenceState.NOT_TAKEN.value) or (o.state == OccurrenceState.PENDING.value and o.scheduled_at <= (now - dt.timedelta(minutes=30))):
                occ_by_date[d_str]["missed"] += 1

    adherence_trend = []
    total_sched = 0
    total_taken = 0
    for d_str, counts in occ_by_date.items():
        s = counts["scheduled"]
        t = counts["taken"]
        total_sched += s
        total_taken += t
        # Only compute percentage if doses were actually scheduled on this day
        rate = round((t / s) * 100, 1) if s > 0 else None
        adherence_trend.append({
            "date": d_str,
            "scheduled": s,
            "taken": t,
            "missed": counts["missed"],
            "rate": rate if rate is not None else 0.0,
            "has_schedule": s > 0
        })

    # 2. Fetch daily check-ins in past `days`
    checkins = db.scalars(
        select(DailyCheckIn)
        .where(
            DailyCheckIn.patient_id == patient_id,
            DailyCheckIn.checkin_date >= start_date,
        )
        .order_by(DailyCheckIn.checkin_date.asc())
    ).all()

    checkin_by_date = {c.checkin_date.isoformat(): c for c in checkins}
    pain_trend = []
    pain_values = []
    for i in range(days):
        d_str = (start_date + dt.timedelta(days=i)).isoformat()
        c = checkin_by_date.get(d_str)
        if c:
            pain_values.append(c.pain_level)
            pain_trend.append({
                "date": d_str,
                "pain_level": c.pain_level,
                "pain_location": c.pain_location or "General",
                "fever": c.fever,
                "cough": c.cough,
                "sleep_hours": c.sleep_hours,
                "activity_level": c.activity_level,
                "submitted": True
            })
        else:
            pain_trend.append({
                "date": d_str,
                "pain_level": None,
                "pain_location": None,
                "fever": False,
                "cough": False,
                "sleep_hours": None,
                "activity_level": None,
                "submitted": False
            })

    overall_adherence = round((total_taken / total_sched) * 100, 1) if total_sched > 0 else 0.0
    average_pain = round(sum(pain_values) / len(pain_values), 1) if pain_values else 0.0

    adherence_trajectory = [
        {
            "date": item["date"],
            "percentage": item.get("rate", 0.0) if item.get("has_schedule") else None,
            "rate": item.get("rate", 0.0) if item.get("has_schedule") else None,
            "scheduled": item["scheduled"],
            "taken": item["taken"],
            "missed": item["missed"],
            "has_schedule": item.get("has_schedule", False)
        }
        for item in adherence_trend
    ]

    symptom_trajectory = [
        {
            "date": item["date"],
            "pain_level": item["pain_level"],
            "pain_location": item.get("pain_location"),
            "has_fever": item.get("fever", False),
            "fever": item.get("fever", False),
            "cough": item.get("cough", False),
            "sleep_hours": item.get("sleep_hours"),
            "activity_level": item.get("activity_level"),
            "submitted": item.get("submitted", False)
        }
        for item in pain_trend
    ]

    return {
        "patient_id": patient_id,
        "window_days": days,
        "adherence_trend": adherence_trend,
        "adherence_trajectory": adherence_trajectory,
        "pain_trend": pain_trend,
        "symptom_trajectory": symptom_trajectory,
        "summary": {
            "overall_adherence_pct": overall_adherence,
            "adherence_percentage_14d": overall_adherence,
            "total_doses_scheduled": total_sched,
            "total_scheduled_doses_14d": total_sched,
            "total_doses_taken": total_taken,
            "taken_doses_14d": total_taken,
            "checkin_count": len(checkins),
            "checkins_count_14d": len(checkins),
            "average_pain_level": average_pain,
            "average_pain_14d": average_pain,
            "max_pain_level": max(pain_values) if pain_values else 0,
        }
    }
