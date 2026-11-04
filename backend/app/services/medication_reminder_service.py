import logging
import datetime as dt
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from app.core.constants import OccurrenceState, SMSCategory, SMSJobStatus, AppointmentStatus
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence, MedicationAdherence
from app.models.patient import Patient
from app.models.sms_job import SMSJob
from app.models.checkin import DailyCheckIn
from app.models.appointment import Appointment
from app.services.sms_gateway_service import queue_and_send_sms
from app.services.notification_service import create_in_app_notification

logger = logging.getLogger("recovera.medication_reminders")


def process_due_medication_reminders(db: Session) -> Dict[str, Any]:
    """
    Evaluates all active medication occurrences and processes the two-step stacked reminder workflow:
    - Message 1: T - 30 minutes before scheduled intake (combining multiple pills for the same meal).
    - Message 2: T (exact meal time / follow-up) if any scheduled doses remain unrecorded in the tracker.
    Notifies both via SMS Gateway and In-App notifications.
    """
    now = dt.datetime.utcnow()
    dose_reminders_sent = 0
    tracker_followups_sent = 0

    # 1. Evaluate Message 1 (T - 30 mins dose reminder)
    # Target occurrences scheduled between now - 120 mins and now + 30 mins
    window_start = now - dt.timedelta(minutes=120)
    window_end = now + dt.timedelta(minutes=30)

    pending_occurrences = db.scalars(
        select(MedicationOccurrence)
        .where(
            MedicationOccurrence.state == OccurrenceState.PENDING.value,
            MedicationOccurrence.scheduled_at >= window_start,
            MedicationOccurrence.scheduled_at <= window_end
        )
    ).all()

    # Group pending occurrences by (patient_id, scheduled_at_timestamp)
    grouped_reminders: Dict[str, List[MedicationOccurrence]] = {}
    for occ in pending_occurrences:
        patient = db.get(Patient, occ.patient_id)
        if not patient or getattr(patient, "care_status", "active") == "idle":
            continue

        med = db.get(Medication, occ.medication_id)
        if not med or med.status == "discontinued":
            continue

        group_id = f"{occ.patient_id}_{occ.scheduled_at.strftime('%Y%m%d_%H%M')}"
        if group_id not in grouped_reminders:
            grouped_reminders[group_id] = []
        grouped_reminders[group_id].append(occ)

    for group_id, occ_list in grouped_reminders.items():
        first_occ = occ_list[0]
        patient = db.get(Patient, first_occ.patient_id)
        if not patient:
            continue

        dose_key = f"stacked_med_dose_{group_id}"
        existing_dose_sms = db.scalar(
            select(SMSJob).where(
                SMSJob.idempotency_key.in_([dose_key] + [f"med_dose_{o.id}" for o in occ_list])
            )
        )

        if not existing_dose_sms:
            # Determine meal label
            hour = first_occ.scheduled_at.hour
            sched = db.get(MedicationSchedule, first_occ.schedule_id) if first_occ.schedule_id else None
            slot_label = (sched.label if sched and sched.label else (
                "Breakfast" if hour < 11 else "Lunch" if hour < 16 else "Dinner"
            ))

            # Build list of medicines with doses and food relations
            med_items = []
            for occ in occ_list:
                m = db.get(Medication, occ.medication_id)
                s = db.get(MedicationSchedule, occ.schedule_id) if occ.schedule_id else None
                m_name = m.name if m else "prescribed medication"
                d_str = s.dose if s and s.dose else (m.dose or "")
                f_rel = (s.food_relation.replace("_", " ") if s and s.food_relation else "as instructed")
                med_items.append(f"{m_name} {d_str} ({f_rel})".strip())

            stacked_meds_text = ", ".join(med_items)
            msg_text = (
                f"Medication reminder: Please take your scheduled {slot_label} medicine{'s' if len(med_items) > 1 else ''} — "
                f"{stacked_meds_text}. Open Recovera for your schedule."
            )

            # Send persistent In-App notification
            if patient.user_id:
                create_in_app_notification(
                    db=db,
                    user_id=patient.user_id,
                    title=f"Medication Reminder: {slot_label}",
                    message=f"Scheduled intake: {stacked_meds_text}.",
                    category="medication",
                    link="#/patient/medications"
                )

            # Send SMS Gateway notification
            if patient.sms_opt_in and patient.phone:
                queue_and_send_sms(
                    db=db,
                    phone=patient.phone,
                    message_text=msg_text,
                    category=SMSCategory.MEDICATION_REMINDER.value,
                    patient_id=patient.id,
                    idempotency_key=dose_key,
                    scheduled_for=now
                )

            db.commit()
            dose_reminders_sent += 1

    # 2. Evaluate Message 2 (Tracker follow-up at exact meal time / T + 0 to T + 6 hours)
    followup_window_start = now - dt.timedelta(hours=6)
    followup_window_end = now  # exact meal time or later

    followup_candidates = db.scalars(
        select(MedicationOccurrence)
        .where(
            MedicationOccurrence.state == OccurrenceState.PENDING.value,
            MedicationOccurrence.scheduled_at >= followup_window_start,
            MedicationOccurrence.scheduled_at <= followup_window_end
        )
    ).all()

    grouped_followups: Dict[str, List[MedicationOccurrence]] = {}
    for occ in followup_candidates:
        adh = db.scalar(select(MedicationAdherence).where(MedicationAdherence.occurrence_id == occ.id))
        if adh:
            continue

        patient = db.get(Patient, occ.patient_id)
        if not patient or getattr(patient, "care_status", "active") == "idle":
            continue

        med = db.get(Medication, occ.medication_id)
        if not med or med.status == "discontinued":
            continue

        group_id = f"{occ.patient_id}_{occ.scheduled_at.strftime('%Y%m%d_%H%M')}"
        if group_id not in grouped_followups:
            grouped_followups[group_id] = []
        grouped_followups[group_id].append(occ)

    for group_id, occ_list in grouped_followups.items():
        first_occ = occ_list[0]
        patient = db.get(Patient, first_occ.patient_id)
        if not patient:
            continue

        # Verify that Message 1 was sent for this group
        prior_dose_sms = db.scalar(
            select(SMSJob).where(
                SMSJob.idempotency_key.in_([f"stacked_med_dose_{group_id}"] + [f"med_dose_{o.id}" for o in occ_list]),
                SMSJob.state.in_([SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value, SMSJobStatus.QUEUED.value])
            )
        )
        if not prior_dose_sms:
            continue

        # Check if follow-up Message 2 was already sent
        track_key = f"stacked_med_track_{group_id}"
        existing_track = db.scalar(
            select(SMSJob).where(
                SMSJob.idempotency_key.in_([track_key] + [f"med_track_{o.id}" for o in occ_list])
            )
        )

        if not existing_track:
            hour = first_occ.scheduled_at.hour
            sched = db.get(MedicationSchedule, first_occ.schedule_id) if first_occ.schedule_id else None
            slot_label = (sched.label if sched and sched.label else (
                "Breakfast" if hour < 11 else "Lunch" if hour < 16 else "Dinner"
            ))

            pending_names = []
            for occ in occ_list:
                m = db.get(Medication, occ.medication_id)
                if m and m.name:
                    pending_names.append(m.name)

            meds_str = ", ".join(pending_names) if pending_names else "prescribed medicines"
            msg_text = (
                f"We hope you have taken your {slot_label} medicines ({meds_str}). "
                f"Please open Recovera and fill in the medicine intake status tracker."
            )

            # Send persistent In-App follow-up
            if patient.user_id:
                create_in_app_notification(
                    db=db,
                    user_id=patient.user_id,
                    title=f"Intake Tracker Follow-Up: {slot_label}",
                    message=f"Please record your intake for {meds_str} in your medication tracker.",
                    category="medication",
                    link="#/patient/medications"
                )

            # Send SMS Gateway follow-up
            if patient.sms_opt_in and patient.phone:
                queue_and_send_sms(
                    db=db,
                    phone=patient.phone,
                    message_text=msg_text,
                    category=SMSCategory.TRACKER_FOLLOWUP.value,
                    patient_id=patient.id,
                    idempotency_key=track_key,
                    scheduled_for=now
                )

            db.commit()
            tracker_followups_sent += 1

    # 3. Suggestion 3: Missed Medication Escalation (T + 2 hours post-scheduled intake)
    missed_escalations_sent = process_missed_medication_escalations(db, now)

    # 4. Suggestion 2: Evening Daily Check-In Reminders
    checkin_reminders_sent = process_evening_checkin_reminders(db, now)

    # 5. Suggestion 5: Upcoming Appointment Reminders (24h and 2h prior)
    appt_reminders_sent = process_upcoming_appointment_reminders(db, now)

    return {
        "dose_reminders_sent": dose_reminders_sent,
        "tracker_followups_sent": tracker_followups_sent,
        "missed_escalations_sent": missed_escalations_sent,
        "checkin_reminders_sent": checkin_reminders_sent,
        "appointment_reminders_sent": appt_reminders_sent,
        "evaluated_at": now.isoformat()
    }


def process_missed_medication_escalations(db: Session, now: Optional[dt.datetime] = None) -> int:
    """
    Suggestion 3: Evaluates overdue medications (T + 2 hours / 120 mins).
    Alerts nursing pool via In-App alert and dispatches warning SMS to patient.
    """
    if now is None:
        now = dt.datetime.utcnow()

    from app.services.notification_service import notify_staff_and_patient_missed_medication

    escalations_sent = 0
    # Window: occurrences scheduled between now - 12 hours and now - 2 hours
    window_start = now - dt.timedelta(hours=12)
    window_end = now - dt.timedelta(hours=2)

    overdue_occurrences = db.scalars(
        select(MedicationOccurrence)
        .where(
            MedicationOccurrence.state == OccurrenceState.PENDING.value,
            MedicationOccurrence.scheduled_at >= window_start,
            MedicationOccurrence.scheduled_at <= window_end
        )
    ).all()

    for occ in overdue_occurrences:
        # Verify no adherence was logged
        adh = db.scalar(select(MedicationAdherence).where(MedicationAdherence.occurrence_id == occ.id))
        if adh:
            continue

        esc_key = f"med_escalation_{occ.id}"
        existing = db.scalar(select(SMSJob).where(SMSJob.idempotency_key == esc_key))
        if existing:
            continue

        patient = db.get(Patient, occ.patient_id)
        if not patient or getattr(patient, "care_status", "active") == "idle":
            continue

        med = db.get(Medication, occ.medication_id)
        if not med or med.status == "discontinued":
            continue

        sched = db.get(MedicationSchedule, occ.schedule_id) if occ.schedule_id else None
        dose_str = sched.dose if sched and sched.dose else (med.dose or "")
        time_str = occ.scheduled_at.strftime("%I:%M %p")

        try:
            notify_staff_and_patient_missed_medication(
                db=db,
                patient=patient,
                med_name=med.name,
                dose_str=dose_str,
                sched_time_str=time_str
            )
            # Create a tracking marker SMSJob to ensure strict idempotency
            marker = SMSJob(
                patient_id=patient.id,
                recipient_phone=patient.phone or "N/A",
                category=SMSCategory.MEDICATION_REMINDER.value,
                message_text=f"Escalation logged for {med.name}",
                idempotency_key=esc_key,
                state=SMSJobStatus.SENT.value,
                scheduled_for=now,
                created_at=now
            )
            db.add(marker)
            db.flush()
            escalations_sent += 1
        except Exception as e:
            logger.warning("Failed to process missed medication escalation for occ %s: %s", occ.id, e)

    return escalations_sent


def process_evening_checkin_reminders(db: Session, now: Optional[dt.datetime] = None) -> int:
    """
    Suggestion 2: Evaluates active patients who have not completed daily check-in for today.
    Dispatches In-App check-in alert and SMS Gateway prompt.
    """
    if now is None:
        now = dt.datetime.utcnow()

    from app.services.notification_service import notify_patient_evening_checkin_reminder

    today = now.date()
    reminders_sent = 0

    # Retrieve all active patients
    active_patients = db.scalars(
        select(Patient).where(Patient.active == True)
    ).all()

    for patient in active_patients:
        # Check care status & active treatment eligibility
        if getattr(patient, "care_status", "active") == "idle":
            continue

        from sqlalchemy import func
        from app.models.recovery_plan import RecoveryPlan
        has_meds = db.scalar(select(func.count(Medication.id)).where(Medication.patient_id == patient.id, Medication.status == "active")) or 0
        has_plan = db.scalar(select(func.count(RecoveryPlan.id)).where(RecoveryPlan.patient_id == patient.id, RecoveryPlan.status.in_(["published", "active"]))) or 0
        if has_meds == 0 and has_plan == 0:
            continue

        checkin_key = f"evening_checkin_{patient.id}_{today.isoformat()}"
        existing = db.scalar(select(SMSJob).where(SMSJob.idempotency_key == checkin_key))
        if existing:
            continue

        # Check if patient already completed a check-in for today
        done_checkin = db.scalar(
            select(DailyCheckIn).where(
                DailyCheckIn.patient_id == patient.id,
                DailyCheckIn.checkin_date == today
            )
        )
        if done_checkin:
            continue

        try:
            notify_patient_evening_checkin_reminder(db=db, patient=patient, idempotency_key=checkin_key)
            reminders_sent += 1
        except Exception as e:
            logger.warning("Failed to send checkin reminder to patient %s: %s", patient.id, e)

    return reminders_sent


def process_upcoming_appointment_reminders(db: Session, now: Optional[dt.datetime] = None) -> int:
    """
    Suggestion 5: Evaluates upcoming appointments and dispatches reminders at 24 hours and 2 hours prior.
    """
    if now is None:
        now = dt.datetime.utcnow()

    from app.services.notification_service import notify_appointment_upcoming_reminder

    reminders_sent = 0

    # 1. Check 24-Hour Reminder Window (scheduled between now + 23h and now + 25h)
    start_24h = now + dt.timedelta(hours=23)
    end_24h = now + dt.timedelta(hours=25)

    appts_24h = db.scalars(
        select(Appointment).where(
            Appointment.status == AppointmentStatus.SCHEDULED.value,
            Appointment.start_time >= start_24h,
            Appointment.start_time <= end_24h
        )
    ).all()

    for appt in appts_24h:
        key_24h = f"appt_remind_24h_{appt.id}"
        existing = db.scalar(select(SMSJob).where(SMSJob.idempotency_key == key_24h))
        if not existing or existing.state == SMSJobStatus.QUEUED.value:
            try:
                notify_appointment_upcoming_reminder(db=db, appointment=appt, hours_before=24, idempotency_key=key_24h)
                reminders_sent += 1
            except Exception as e:
                logger.warning("Failed 24h appointment reminder for %s: %s", appt.id, e)

    # 2. Check 2-Hour Reminder Window (scheduled between now + 90 min and now + 150 min)
    start_2h = now + dt.timedelta(minutes=90)
    end_2h = now + dt.timedelta(minutes=150)

    appts_2h = db.scalars(
        select(Appointment).where(
            Appointment.status == AppointmentStatus.SCHEDULED.value,
            Appointment.start_time >= start_2h,
            Appointment.start_time <= end_2h
        )
    ).all()

    for appt in appts_2h:
        key_2h = f"appt_remind_2h_{appt.id}"
        existing = db.scalar(select(SMSJob).where(SMSJob.idempotency_key == key_2h))
        if not existing or existing.state == SMSJobStatus.QUEUED.value:
            try:
                notify_appointment_upcoming_reminder(db=db, appointment=appt, hours_before=2, idempotency_key=key_2h)
                reminders_sent += 1
            except Exception as e:
                logger.warning("Failed 2h appointment reminder for %s: %s", appt.id, e)

    return reminders_sent


def cancel_pending_reminders_for_occurrence(db: Session, occurrence_id: str) -> int:
    """Cancels queued SMS reminder jobs associated with an occurrence when it is cancelled or discontinued."""
    jobs = db.scalars(
        select(SMSJob).where(
            SMSJob.idempotency_key.like(f"%{occurrence_id}%"),
            SMSJob.state == SMSJobStatus.QUEUED.value
        )
    ).all()

    count = 0
    for job in jobs:
        job.state = SMSJobStatus.CANCELLED.value
        count += 1
    if count:
        db.commit()
    return count
