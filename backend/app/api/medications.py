import datetime as dt
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_

from app.core.database import get_db
from app.core.constants import Role, OccurrenceState
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence, MedicationAdherence
from app.models.patient import Patient
from app.models.user import User
from app.schemas.medication import (
    MedicationCreateIn,
    MedicationUpdateIn,
    MedicationOut,
    MedicationOccurrenceOut,
    AdherenceRecordIn,
    AdherenceRecordOut,
    MedicationScheduleOut,
    InteractionCheckIn,
    InteractionCheckOut
)
from app.services.medication_service import (
    create_medication_and_schedule,
    update_medication,
    discontinue_medication,
    record_occurrence_adherence
)
from app.services.interaction_service import check_drug_interactions
from app.models.patient import MedicalHistory
from app.core.audit import record_audit
from app.api.deps import get_current_user, require_role, enforce_patient_isolation

import logging

logger = logging.getLogger("recovera.medications")

router = APIRouter(tags=["Medications & Adherence Tracker"])


@router.get("/api/medications/library")
def get_medication_library(
    current_user: User = Depends(get_current_user)
):
    """
    Returns the comprehensive medication library including system defaults and newly ingested medicines.
    """
    from app.services.handwriting_ml_service import COMMON_MEDICATIONS
    return COMMON_MEDICATIONS


@router.post("/api/medications/library")
def add_to_medication_library(
    payload: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """
    Allows clinical staff to register a newly prescribed or discovered medicine in the library.
    Guarantees strict case-insensitive deduplication.
    """
    from app.services.handwriting_ml_service import register_medicine_in_library
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Medication name is required.")
    
    added = register_medicine_in_library(
        name=name,
        generic=payload.get("generic") or payload.get("generic_name") or payload.get("classification"),
        typical_doses=[payload.get("dose")] if payload.get("dose") else payload.get("typical_doses"),
        forms=payload.get("forms") or ["Tablet"]
    )
    return {"success": True, "added": added, "name": name}


def _get_patient_clinical_context(pid: str, db: Session):
    # Active meds
    active_meds = db.scalars(
        select(Medication).where(
            Medication.patient_id == pid,
            Medication.status == "active"
        )
    ).all()
    raw_names = [m.name for m in active_meds] + [m.generic_name for m in active_meds if m.generic_name]
    med_names = list(dict.fromkeys(raw_names))

    # Conditions & allergies
    conditions = []
    allergies = []
    med_hist = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == pid))
    if med_hist:
        if med_hist.conditions:
            conditions.extend(med_hist.conditions if isinstance(med_hist.conditions, list) else [med_hist.conditions])
        if med_hist.notes:
            allergies.append(med_hist.notes)

    patient = db.get(Patient, pid)
    if patient and patient.history:
        if isinstance(patient.history, dict):
            if "conditions" in patient.history:
                conditions.extend(patient.history["conditions"])
            if "allergies" in patient.history:
                allergies.extend(patient.history["allergies"])
        elif isinstance(patient.history, list):
            conditions.extend(patient.history)

    return med_names, conditions, allergies

def _format_medication(med: Medication, db: Session) -> MedicationOut:
    schedules = db.scalars(
        select(MedicationSchedule).where(MedicationSchedule.medication_id == med.id)
    ).all()
    sched_outs = [
        MedicationScheduleOut(
            id=s.id,
            medication_id=s.medication_id,
            label=s.label,
            time_hhmm=s.time_hhmm,
            dose=s.dose,
            food_relation=s.food_relation
        )
        for s in schedules
    ]
    return MedicationOut(
        id=med.id,
        patient_id=med.patient_id,
        name=med.name,
        generic_name=med.generic_name,
        strength=med.strength,
        dose=med.dose,
        unit=med.unit,
        route=med.route,
        start_date=med.start_date,
        end_date=med.end_date,
        instructions=med.instructions or "",
        status=med.status,
        version=med.version,
        source_document_id=med.source_document_id,
        created_by=med.created_by,
        created_at=med.created_at,
        schedules=sched_outs
    )

@router.post("/api/medications/check-interactions", response_model=InteractionCheckOut)
def check_interactions_endpoint(
    payload: InteractionCheckIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, payload.patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    med_names, conditions, allergies = _get_patient_clinical_context(payload.patient_id, db)

    res = check_drug_interactions(
        candidate_medication=payload.medication_name,
        candidate_dose=payload.dose,
        active_medications=med_names,
        conditions=conditions,
        allergies=allergies
    )
    return res

@router.get("/api/patients/{pid}/medications", response_model=List[MedicationOut])
def list_patient_medications(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)

    meds = db.scalars(
        select(Medication).where(Medication.patient_id == pid).order_by(Medication.created_at.desc())
    ).all()

    return [_format_medication(m, db) for m in meds]


@router.get("/api/patients/{pid}/medications/export")
def export_patient_medications(
    pid: str,
    format: str = Query("csv", regex="^(csv|json)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Exports official patient medication prescriptions including mandatory duration,
    intake type (before/after food), dosage/count, 1-1-1 period notation, and descriptions.
    """
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)

    meds = db.scalars(
        select(Medication).where(Medication.patient_id == pid).order_by(Medication.created_at.desc())
    ).all()

    formatted_meds = [_format_medication(m, db) for m in meds]

    export_rows = []
    for med in formatted_meds:
        duration_days = (med.end_date - med.start_date).days + 1 if med.end_date and med.start_date else 0

        has_b = 0
        has_l = 0
        has_d = 0
        intake_types = set()

        for s in (med.schedules or []):
            lbl = (s.label or "").lower()
            time = s.time_hhmm or ""
            f_rel = (s.food_relation or "after_food").replace("_", " ").title()
            intake_types.add(f_rel)
            if "breakfast" in lbl or "morning" in lbl or time == "08:00":
                has_b = 1
            elif "lunch" in lbl or "afternoon" in lbl or "noon" in lbl or time in ("13:00", "14:00"):
                has_l = 1
            elif "dinner" in lbl or "night" in lbl or "evening" in lbl or time in ("20:00", "21:00"):
                has_d = 1

        period_code = f"{has_b}-{has_l}-{has_d}"
        active_meals = []
        if has_b:
            active_meals.append("Breakfast")
        if has_l:
            active_meals.append("Lunch")
        if has_d:
            active_meals.append("Dinner")
        timing_label = " & ".join(active_meals) if active_meals else "As Needed"
        intake_type_str = ", ".join(sorted(intake_types)) if intake_types else "After Food"

        export_rows.append({
            "patient_name": patient.name,
            "patient_code": patient.patient_code,
            "medication_name": med.name,
            "generic_name": med.generic_name or "",
            "dosage": med.dose or med.strength or "1 tablet",
            "duration_days": duration_days,
            "intake_type": intake_type_str,
            "period_notation": period_code,
            "timing_schedule": f"{period_code} ({timing_label})",
            "instructions": med.instructions or "",
            "start_date": str(med.start_date),
            "end_date": str(med.end_date),
            "status": med.status
        })

    if format == "json":
        return {
            "patient_id": pid,
            "patient_name": patient.name,
            "patient_code": patient.patient_code,
            "export_date": dt.date.today().isoformat(),
            "medications": export_rows
        }

    import io
    import csv
    from fastapi.responses import Response

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Patient Name", "Patient Code", "Medication Name", "Generic Name",
        "Dosage / Tablet Count", "Duration (Days)", "Intake Type (Before/After Food)",
        "Period Notation (1-1-1)", "Timing Schedule", "Special Instructions / Description",
        "Start Date", "End Date", "Status"
    ])
    for r in export_rows:
        writer.writerow([
            r["patient_name"], r["patient_code"], r["medication_name"], r["generic_name"],
            r["dosage"], r["duration_days"], r["intake_type"],
            r["period_notation"], r["timing_schedule"], r["instructions"],
            r["start_date"], r["end_date"], r["status"]
        ])

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=prescriptions_{patient.patient_code}.csv"}
    )


@router.post("/api/patients/{pid}/medications", response_model=MedicationOut, status_code=status.HTTP_201_CREATED)
def add_patient_medication(
    pid: str,
    payload: MedicationCreateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    # ADE & Cross-Reactivity Interceptor Check
    med_names, conditions, allergies = _get_patient_clinical_context(pid, db)
    interaction_res = check_drug_interactions(
        candidate_medication=payload.name,
        candidate_dose=payload.dose,
        active_medications=med_names,
        conditions=conditions,
        allergies=allergies
    )

    if interaction_res["override_required"]:
        if not payload.clinical_override_rationale or len(payload.clinical_override_rationale.strip()) < 8:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "Adverse drug event interceptor: Clinical override rationale is required before prescribing.",
                    "max_severity": interaction_res["max_severity"],
                    "interactions": interaction_res["interactions"]
                }
            )
        # Log clinician override to audit log
        record_audit(
            db=db,
            actor_id=current_user.id,
            action="medication_interaction_override",
            target_type="medication",
            target_id=pid,
            details={
                "candidate_medication": payload.name,
                "max_severity": interaction_res["max_severity"],
                "clinical_override_rationale": payload.clinical_override_rationale,
                "interactions": interaction_res["interactions"]
            }
        )

    med = create_medication_and_schedule(
        patient_id=pid,
        payload=payload,
        db=db,
        created_by=current_user.id
    )

    # Suggestion 4: Alert patient of new prescription
    try:
        from app.services.notification_service import notify_patient_medication_prescribed
        notify_patient_medication_prescribed(
            db=db,
            patient=patient,
            doctor=current_user,
            med_name=payload.name,
            dose=payload.dose or payload.strength or "",
            action="prescribed"
        )
    except Exception as e:
        import logging
        logging.getLogger("recovera.medications").warning("Failed to notify patient of prescribed med: %s", e)

    return _format_medication(med, db)

@router.put("/api/medications/{mid}", response_model=MedicationOut)
def edit_medication(
    mid: str,
    payload: MedicationUpdateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    med = db.get(Medication, mid)
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found.")

    updated_med = update_medication(
        medication_id=mid,
        payload=payload,
        db=db,
        updated_by=current_user.id
    )

    # Alert patient of updated prescription
    try:
        from app.services.notification_service import notify_patient_medication_prescribed
        patient_rec = db.get(Patient, med.patient_id)
        if patient_rec:
            notify_patient_medication_prescribed(
                db=db,
                patient=patient_rec,
                doctor=current_user,
                med_name=updated_med.name,
                dose=updated_med.dose or updated_med.strength or "",
                action="updated"
            )
    except Exception as e:
        import logging
        logging.getLogger("recovera.medications").warning("Failed to notify patient of updated med: %s", e)

    return _format_medication(updated_med, db)

@router.post("/api/medications/{mid}/discontinue", response_model=MedicationOut)
def discontinue_medication_endpoint(
    mid: str,
    reason: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    med = db.get(Medication, mid)
    if not med:
        raise HTTPException(status_code=404, detail="Medication not found.")

    disc_med = discontinue_medication(
        medication_id=mid,
        db=db,
        actor_id=current_user.id,
        reason=reason
    )

    # Suggestion 4: Alert patient of discontinued medication
    try:
        from app.models.patient import Patient
        from app.services.notification_service import notify_patient_medication_prescribed
        patient_rec = db.get(Patient, med.patient_id)
        if patient_rec:
            notify_patient_medication_prescribed(
                db=db,
                patient=patient_rec,
                doctor=current_user,
                med_name=med.name,
                dose=med.dose or med.strength or "",
                action="discontinued"
            )
    except Exception as e:
        import logging
        logging.getLogger("recovera.medications").warning("Failed to notify patient of discontinued med: %s", e)

    return _format_medication(disc_med, db)

@router.get("/api/patients/{pid}/medication-occurrences", response_model=List[MedicationOccurrenceOut])
def list_medication_occurrences(
    pid: str,
    date: Optional[str] = Query(None, description="Filter date in YYYY-MM-DD format"),
    start_date: Optional[str] = Query(None, description="Filter range start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="Filter range end date YYYY-MM-DD"),
    state: Optional[str] = Query(None),
    missed_only: bool = Query(False, description="Filter only missed or overdue unlogged occurrences"),
    limit: int = Query(500, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)

    query = select(MedicationOccurrence).where(MedicationOccurrence.patient_id == pid)

    # Exclude cancelled occurrences unless explicitly requested by state query param
    if not state:
        query = query.where(MedicationOccurrence.state != OccurrenceState.CANCELLED.value)
    else:
        query = query.where(MedicationOccurrence.state == state)

    # Strictly exclude unrecorded occurrences of discontinued medications
    active_med_subquery = select(Medication.id).where(
        Medication.patient_id == pid,
        Medication.status == "active"
    )
    query = query.where(
        or_(
            MedicationOccurrence.state == OccurrenceState.TAKEN.value,
            MedicationOccurrence.medication_id.in_(active_med_subquery)
        )
    )

    if missed_only:
        now = dt.datetime.utcnow()
        # Occurrences in past 7 days up to now that are UNVERIFIED / UNLOGGED:
        # Scheduled more than 30 mins ago, still in PENDING state, and without adherence record.
        cutoff = now - dt.timedelta(minutes=30)
        window_start = now - dt.timedelta(days=7)
        query = query.where(
            MedicationOccurrence.scheduled_at >= window_start,
            MedicationOccurrence.scheduled_at <= now,
            MedicationOccurrence.state == OccurrenceState.PENDING.value,
            MedicationOccurrence.scheduled_at <= cutoff,
            MedicationOccurrence.id.not_in(select(MedicationAdherence.occurrence_id))
        )
    else:
        if date:
            try:
                target_date = dt.date.fromisoformat(date)
                start_dt = dt.datetime.combine(target_date, dt.time.min)
                end_dt = dt.datetime.combine(target_date, dt.time.max)
                query = query.where(MedicationOccurrence.scheduled_at >= start_dt, MedicationOccurrence.scheduled_at <= end_dt)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")
        elif start_date and end_date:
            try:
                s_date = dt.date.fromisoformat(start_date)
                e_date = dt.date.fromisoformat(end_date)
                start_dt = dt.datetime.combine(s_date, dt.time.min)
                end_dt = dt.datetime.combine(e_date, dt.time.max)
                query = query.where(MedicationOccurrence.scheduled_at >= start_dt, MedicationOccurrence.scheduled_at <= end_dt)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date or end_date format. Expected YYYY-MM-DD.")

    query = query.order_by(MedicationOccurrence.scheduled_at.asc()).limit(limit)
    occurrences = db.scalars(query).all()

    # Pre-fetch medications, schedules, and adherences
    med_ids = list({occ.medication_id for occ in occurrences})
    sched_ids = list({occ.schedule_id for occ in occurrences})
    occ_ids = [occ.id for occ in occurrences]

    med_map = {}
    if med_ids:
        meds = db.scalars(select(Medication).where(Medication.id.in_(med_ids))).all()
        med_map = {m.id: m for m in meds}

    sched_map = {}
    if sched_ids:
        scheds = db.scalars(select(MedicationSchedule).where(MedicationSchedule.id.in_(sched_ids))).all()
        sched_map = {s.id: s for s in scheds}

    adh_map = {}
    if occ_ids:
        adherences = db.scalars(select(MedicationAdherence).where(MedicationAdherence.occurrence_id.in_(occ_ids))).all()
        adh_map = {a.occurrence_id: a for a in adherences}

    result = []
    for occ in occurrences:
        med = med_map.get(occ.medication_id)
        sched = sched_map.get(occ.schedule_id)
        adh = adh_map.get(occ.id)

        adh_dict = None
        if adh:
            adh_dict = {
                "id": adh.id,
                "state": adh.state,
                "reason": adh.reason,
                "recorded_at": adh.recorded_at.isoformat()
            }

        result.append(
            MedicationOccurrenceOut(
                id=occ.id,
                medication_id=occ.medication_id,
                schedule_id=occ.schedule_id,
                patient_id=occ.patient_id,
                medicine_name=med.name if med else "Unknown Medication",
                dose=sched.dose if sched else (med.dose if med else None),
                label=sched.label if sched else None,
                food_relation=sched.food_relation if sched else None,
                scheduled_at=occ.scheduled_at,
                state=occ.state,
                snooze_count=occ.snooze_count,
                snoozed_until=occ.snoozed_until,
                finalized_at=occ.finalized_at,
                start_date=med.start_date if med else None,
                end_date=med.end_date if med else None,
                adherence=adh_dict
            )
        )

    return result

@router.post("/api/medication-occurrences/{occ_id}/adherence", response_model=AdherenceRecordOut)
def record_adherence_endpoint(
    occ_id: str,
    payload: AdherenceRecordIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    occ = db.get(MedicationOccurrence, occ_id)
    if not occ:
        raise HTTPException(status_code=404, detail="Medication occurrence not found.")

    # Isolation check
    enforce_patient_isolation(patient_id=occ.patient_id, current_user=current_user, db=db)

    adh = record_occurrence_adherence(
        occurrence_id=occ_id,
        payload=payload,
        db=db,
        user_id=current_user.id
    )

    # Immediately re-evaluate clinical risk so emergency/adherence status adapts in real-time
    if occ.patient_id:
        try:
            from app.services.risk_service import evaluate_patient_risk
            evaluate_patient_risk(db, patient_id=occ.patient_id, window_days=7, force=True)
            db.commit()
        except Exception as e:
            logger.warning(f"Auto-reevaluating patient risk failed: {e}")

    return AdherenceRecordOut(
        occurrence_id=adh.occurrence_id,
        state=adh.state,
        reason=adh.reason,
        recorded_at=adh.recorded_at
    )
