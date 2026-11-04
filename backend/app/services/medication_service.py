import uuid
import datetime as dt
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence, MedicationAdherence
from app.models.patient import Patient
from app.models.user import User
from app.schemas.medication import MedicationCreateIn, MedicationUpdateIn, AdherenceRecordIn
from app.core.constants import OccurrenceState
from app.core.audit import record_audit

def create_medication_and_schedule(
    patient_id: str,
    payload: MedicationCreateIn,
    db: Session,
    created_by: str
) -> Medication:
    """
    Creates a medication and generates all scheduled occurrences across the date range.
    """
    if payload.end_date < payload.start_date:
        raise HTTPException(status_code=400, detail="end_date cannot be earlier than start_date.")

    medication = Medication(
        patient_id=patient_id,
        name=payload.name,
        generic_name=payload.generic_name,
        strength=payload.strength,
        dose=payload.dose,
        unit=payload.unit,
        route=payload.route,
        start_date=payload.start_date,
        end_date=payload.end_date,
        instructions=payload.instructions or "",
        status="active",
        version=1,
        source_document_id=payload.source_document_id,
        created_by=created_by,
        created_at=dt.datetime.utcnow()
    )
    db.add(medication)
    db.flush()

    schedules: List[MedicationSchedule] = []
    for s_in in payload.schedules:
        sched = MedicationSchedule(
            medication_id=medication.id,
            patient_id=patient_id,
            label=s_in.label,
            time_hhmm=s_in.time_hhmm,
            dose=s_in.dose or payload.dose,
            food_relation=s_in.food_relation
        )
        db.add(sched)
        schedules.append(sched)

    db.flush()

    # Generate occurrences for each day and schedule
    generate_occurrences_for_schedules(
        medication=medication,
        schedules=schedules,
        start_date=payload.start_date,
        end_date=payload.end_date,
        db=db
    )

    record_audit(
        db=db,
        actor_id=created_by,
        action="medication_created",
        target_type="medication",
        target_id=medication.id,
        new_values={"name": medication.name, "start_date": str(medication.start_date), "end_date": str(medication.end_date)}
    )
    db.commit()
    db.refresh(medication)
    return medication

def generate_occurrences_for_schedules(
    medication: Medication,
    schedules: List[MedicationSchedule],
    start_date: dt.date,
    end_date: dt.date,
    db: Session
) -> int:
    """
    Generates dated occurrence records for all days in [start_date, end_date].
    Uses idempotency checks to avoid duplicate occurrences.
    """
    total_days = (end_date - start_date).days + 1
    count = 0

    for day_offset in range(total_days):
        current_day = start_date + dt.timedelta(days=day_offset)
        for sched in schedules:
            try:
                hour_str, min_str = sched.time_hhmm.split(":")
                scheduled_dt = dt.datetime.combine(current_day, dt.time(int(hour_str), int(min_str)))
            except Exception:
                scheduled_dt = dt.datetime.combine(current_day, dt.time(8, 0))

            # Idempotency check: avoid duplicate occurrence for schedule at this time
            existing = db.scalar(
                select(MedicationOccurrence).where(
                    MedicationOccurrence.schedule_id == sched.id,
                    MedicationOccurrence.scheduled_at == scheduled_dt
                )
            )
            if not existing:
                occ = MedicationOccurrence(
                    medication_id=medication.id,
                    schedule_id=sched.id,
                    patient_id=medication.patient_id,
                    scheduled_at=scheduled_dt,
                    state=OccurrenceState.PENDING.value,
                    snooze_count=0
                )
                db.add(occ)
                count += 1

    db.flush()
    return count

def update_medication(
    medication_id: str,
    payload: MedicationUpdateIn,
    db: Session,
    updated_by: str
) -> Medication:
    """
    Updates medication details while strictly preserving past occurrences & adherence records.
    Only future unrecorded occurrences are modified.
    """
    med = db.get(Medication, medication_id)
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found.")

    med.version += 1

    if payload.name is not None:
        med.name = payload.name
    if payload.generic_name is not None:
        med.generic_name = payload.generic_name
    if payload.strength is not None:
        med.strength = payload.strength
    if payload.dose is not None:
        med.dose = payload.dose
    if payload.unit is not None:
        med.unit = payload.unit
    if payload.route is not None:
        med.route = payload.route
    if payload.instructions is not None:
        med.instructions = payload.instructions

    now = dt.datetime.utcnow()

    # If date range or schedules change, update future occurrences
    if payload.start_date is not None:
        med.start_date = payload.start_date
    if payload.end_date is not None:
        med.end_date = payload.end_date

    if payload.schedules is not None:
        # 1. Cancel future pending occurrences that have no adherence recorded
        future_pending = db.scalars(
            select(MedicationOccurrence).where(
                MedicationOccurrence.medication_id == med.id,
                MedicationOccurrence.scheduled_at >= now,
                MedicationOccurrence.state == OccurrenceState.PENDING.value
            )
        ).all()

        for occ in future_pending:
            # Check if adherence was logged
            adh = db.scalar(select(MedicationAdherence).where(MedicationAdherence.occurrence_id == occ.id))
            if not adh:
                occ.state = OccurrenceState.CANCELLED.value
                occ.finalized_at = now

        # 2. Add new schedules
        new_scheds = []
        for s_in in payload.schedules:
            ns = MedicationSchedule(
                medication_id=med.id,
                patient_id=med.patient_id,
                label=s_in.label,
                time_hhmm=s_in.time_hhmm,
                dose=s_in.dose or med.dose,
                food_relation=s_in.food_relation
            )
            db.add(ns)
            new_scheds.append(ns)
        db.flush()

        # 3. Generate future occurrences from today/tomorrow onwards up to end_date
        today = now.date()
        start_for_new = max(med.start_date, today)
        if med.end_date >= start_for_new:
            generate_occurrences_for_schedules(
                medication=med,
                schedules=new_scheds,
                start_date=start_for_new,
                end_date=med.end_date,
                db=db
            )
    elif payload.end_date is not None:
        # Existing schedules, potentially extended end_date
        existing_scheds = db.scalars(
            select(MedicationSchedule).where(MedicationSchedule.medication_id == med.id)
        ).all()
        today = now.date()
        start_for_new = max(med.start_date, today)
        if med.end_date >= start_for_new and existing_scheds:
            generate_occurrences_for_schedules(
                medication=med,
                schedules=existing_scheds,
                start_date=start_for_new,
                end_date=med.end_date,
                db=db
            )

    record_audit(
        db=db,
        actor_id=updated_by,
        action="medication_updated",
        target_type="medication",
        target_id=med.id,
        new_values={"version": med.version}
    )
    db.commit()
    db.refresh(med)
    return med

def discontinue_medication(
    medication_id: str,
    db: Session,
    actor_id: str,
    reason: Optional[str] = None
) -> Medication:
    """
    Discontinues medication. Cancels all pending occurrences that have no recorded
    adherence, while preserving all past intake history completely intact.
    """
    med = db.get(Medication, medication_id)
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found.")

    med.status = "discontinued"
    now = dt.datetime.utcnow()

    # Cancel all pending occurrences without adherence
    pending_occurrences = db.scalars(
        select(MedicationOccurrence).where(
            MedicationOccurrence.medication_id == med.id,
            MedicationOccurrence.state == OccurrenceState.PENDING.value
        )
    ).all()

    cancelled_count = 0
    from app.services.medication_reminder_service import cancel_pending_reminders_for_occurrence
    for occ in pending_occurrences:
        adh = db.scalar(select(MedicationAdherence).where(MedicationAdherence.occurrence_id == occ.id))
        if not adh:
            occ.state = OccurrenceState.CANCELLED.value
            occ.finalized_at = now
            cancelled_count += 1
            cancel_pending_reminders_for_occurrence(db, occ.id)

    record_audit(
        db=db,
        actor_id=actor_id,
        action="medication_discontinued",
        target_type="medication",
        target_id=med.id,
        new_values={"status": "discontinued", "cancelled_future_occurrences": cancelled_count, "reason": reason}
    )
    db.commit()
    db.refresh(med)
    return med

def record_occurrence_adherence(
    occurrence_id: str,
    payload: AdherenceRecordIn,
    db: Session,
    user_id: str
) -> MedicationAdherence:
    """
    Records adherence for an occurrence (taken / not_taken with optional reason).
    Enforces that past records are updated idempotently.
    """
    occ = db.get(MedicationOccurrence, occurrence_id)
    if not occ:
        raise HTTPException(status_code=404, detail="Medication occurrence not found.")

    now = dt.datetime.utcnow()
    occ.state = payload.state
    occ.finalized_at = now

    adh = db.scalar(
        select(MedicationAdherence).where(MedicationAdherence.occurrence_id == occ.id)
    )

    if adh:
        adh.state = payload.state
        adh.reason = payload.reason
        adh.recorded_by = user_id
        adh.recorded_at = now
    else:
        adh = MedicationAdherence(
            occurrence_id=occ.id,
            state=payload.state,
            reason=payload.reason,
            recorded_by=user_id,
            recorded_at=now
        )
        db.add(adh)

    record_audit(
        db=db,
        actor_id=user_id,
        action="medication_adherence_recorded",
        target_type="medication_occurrence",
        target_id=occ.id,
        new_values={"state": payload.state, "reason": payload.reason}
    )
    db.commit()
    db.refresh(adh)
    return adh
