"""
Recovera AI – Daily Symptom Check-ins API (Phase 11)

Endpoints:
  POST  /api/checkins                     – Submit or update daily symptom check-in
  GET   /api/checkins                     – List check-ins for logged-in patient
  GET   /api/patients/{pid}/checkins      – List check-ins for a specific patient (clinician access)
"""
import datetime as dt
from typing import Optional, List, Union, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.constants import Role
from app.core.audit import record_audit
from app.models.checkin import DailyCheckIn
from app.models.patient import Patient
from app.models.user import User
from app.api.deps import get_current_user, require_role

router = APIRouter(tags=["Check-ins"])


class DailyCheckInRequest(BaseModel):
    patient_id: Optional[str] = None
    checkin_date: Optional[dt.date] = None
    pain_level: int = Field(..., ge=0, le=10, description="Pain level on a 0 to 10 visual scale")
    pain_location: Optional[Union[str, List[str]]] = None
    sleep_hours: Optional[float] = Field(None, ge=0, le=24)
    activity_level: Optional[str] = "normal"  # rest, light, normal, active
    fever: bool = False
    cough: bool = False
    notes: Optional[str] = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            loc = values.get("pain_location") or values.get("anatomical_location") or values.get("pain_locations")
            if isinstance(loc, list):
                values["pain_location"] = ", ".join(str(x) for x in loc if x)
            elif loc is not None:
                values["pain_location"] = str(loc)
        return values


def _checkin_to_dict(c: DailyCheckIn) -> dict:
    return {
        "id": c.id,
        "patient_id": c.patient_id,
        "checkin_date": c.checkin_date.isoformat(),
        "pain_level": c.pain_level,
        "pain_score": c.pain_level,
        "pain_location": c.pain_location,
        "anatomical_location": c.pain_location,
        "sleep_hours": c.sleep_hours,
        "activity_level": c.activity_level,
        "fever": c.fever,
        "cough": c.cough,
        "notes": c.notes or "",
        "submitted_at": c.submitted_at.isoformat() if c.submitted_at else None,
    }


@router.get("/api/checkins/eligibility")
def get_checkin_eligibility(
    patient_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from sqlalchemy import func
    from app.models.medication import Medication
    from app.models.recovery_plan import RecoveryPlan

    target_pid = patient_id
    if current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if not patient:
            raise HTTPException(status_code=404, detail="Patient profile not found.")
        target_pid = patient.id
    else:
        if not target_pid:
            raise HTTPException(status_code=422, detail="patient_id required for staff.")
        patient = db.get(Patient, target_pid)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found.")

    earliest_med_date = db.scalar(
        select(func.min(Medication.start_date)).where(Medication.patient_id == patient.id)
    )
    if not earliest_med_date:
        earliest_created = db.scalar(
            select(func.min(Medication.created_at)).where(Medication.patient_id == patient.id)
        )
        if earliest_created:
            earliest_med_date = earliest_created.date()

    treatment_start_date = earliest_med_date or (patient.created_at.date() if patient.created_at else dt.date.today())

    care_status = getattr(patient, "care_status", "active") or "active"
    if care_status == "idle":
        return {
            "eligible": False,
            "reason": "care_idle",
            "message": "Your recovery care status is currently set to Idle. Routine daily check-ins are paused.",
            "care_status": "idle",
            "earliest_medication_date": earliest_med_date.isoformat() if earliest_med_date else None,
            "treatment_start_date": treatment_start_date.isoformat()
        }

    has_meds = db.scalar(select(func.count(Medication.id)).where(Medication.patient_id == patient.id, Medication.status == "active")) or 0
    has_plan = db.scalar(select(func.count(RecoveryPlan.id)).where(RecoveryPlan.patient_id == patient.id, RecoveryPlan.status.in_(["published", "active"]))) or 0

    if has_meds == 0 and has_plan == 0:
        return {
            "eligible": False,
            "reason": "no_active_care",
            "message": "Daily check-in is paused: No active medications or recovery plan requiring daily monitoring.",
            "care_status": "active",
            "earliest_medication_date": earliest_med_date.isoformat() if earliest_med_date else None,
            "treatment_start_date": treatment_start_date.isoformat()
        }

    return {
        "eligible": True,
        "care_status": "active",
        "has_medications": bool(has_meds),
        "has_recovery_plan": bool(has_plan),
        "earliest_medication_date": earliest_med_date.isoformat() if earliest_med_date else None,
        "treatment_start_date": treatment_start_date.isoformat(),
        "message": "Patient is actively undergoing care and eligible for daily check-in."
    }


@router.post("/api/checkins", status_code=status.HTTP_201_CREATED)
def submit_checkin(
    req: DailyCheckInRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Resolve target patient
    target_patient_id = req.patient_id
    if current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if not patient:
            raise HTTPException(status_code=404, detail="Patient profile not found for this account.")
        target_patient_id = patient.id
    else:
        if not target_patient_id:
            raise HTTPException(status_code=422, detail="patient_id is required when submitted by clinical staff.")
        patient = db.get(Patient, target_patient_id)
        if not patient or not patient.active:
            raise HTTPException(status_code=404, detail="Patient not found or inactive.")
        if current_user.role == Role.DOCTOR.value and patient.assigned_doctor_id != current_user.id:
            from app.models.patient import PatientDoctorAssignment
            assignment = db.scalar(
                select(PatientDoctorAssignment).where(
                    PatientDoctorAssignment.patient_id == patient.id,
                    PatientDoctorAssignment.doctor_id == current_user.id
                )
            )
            if not assignment:
                raise HTTPException(status_code=403, detail="Doctor may only submit check-ins for assigned patients.")

    # Enforce active care status & active treatment requirement
    care_status = getattr(patient, "care_status", "active") or "active"
    if care_status == "idle":
        raise HTTPException(status_code=403, detail="Daily check-in is paused because care status is set to Idle.")

    from sqlalchemy import func
    from app.models.medication import Medication
    from app.models.recovery_plan import RecoveryPlan
    has_meds = db.scalar(select(func.count(Medication.id)).where(Medication.patient_id == patient.id, Medication.status == "active")) or 0
    has_plan = db.scalar(select(func.count(RecoveryPlan.id)).where(RecoveryPlan.patient_id == patient.id, RecoveryPlan.status.in_(["published", "active"]))) or 0
    if has_meds == 0 and has_plan == 0:
        raise HTTPException(status_code=400, detail="Daily check-in is not required: Patient has no active medications or recovery plan.")

    target_date = req.checkin_date or dt.date.today()
    today = dt.date.today()
    if target_date > today:
        raise HTTPException(status_code=400, detail="Cannot submit check-ins for future calendar dates.")

    # Retrospective reporting cutoff: cannot log missed check-ins older than 7 calendar days
    if (today - target_date).days > 7:
        raise HTTPException(
            status_code=400,
            detail="Cannot report missed daily check-ins older than 7 days from today to prevent retrospective recall bias."
        )

    # Pre-treatment boundary: check-ins cannot be logged prior to the patient's first medication added date
    earliest_med = db.scalar(
        select(func.min(Medication.start_date)).where(Medication.patient_id == target_patient_id)
    )
    if not earliest_med:
        earliest_created = db.scalar(
            select(func.min(Medication.created_at)).where(Medication.patient_id == target_patient_id)
        )
        if earliest_created:
            earliest_med = earliest_created.date()

    if earliest_med and target_date < earliest_med:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot submit daily check-in prior to medication start date ({earliest_med.isoformat()}). Daily check-in tracking begins when medications are prescribed."
        )

    now = dt.datetime.utcnow()

    # Upsert check-in for this patient on this date
    existing = db.scalar(
        select(DailyCheckIn).where(
            DailyCheckIn.patient_id == target_patient_id,
            DailyCheckIn.checkin_date == target_date
        )
    )

    if existing:
        # Enforce calendar-cycle restriction:
        # Only today's check-in can be edited or updated.
        # Once that calendar day passes, past submissions are strictly locked.
        if target_date < today:
            raise HTTPException(
                status_code=403,
                detail="Past daily check-ins are locked once that calendar day has passed. Only today's check-in can be edited or updated."
            )

        existing.pain_level = req.pain_level
        existing.pain_location = req.pain_location
        existing.sleep_hours = req.sleep_hours
        existing.activity_level = req.activity_level
        existing.fever = req.fever
        existing.cough = req.cough
        existing.notes = req.notes or ""
        existing.submitted_at = now
        checkin_obj = existing
        action = "checkin_updated"
    else:
        # New check-in (either for today, or submitting a missed day retroactively)
        checkin_obj = DailyCheckIn(
            patient_id=target_patient_id,
            checkin_date=target_date,
            pain_level=req.pain_level,
            pain_location=req.pain_location,
            sleep_hours=req.sleep_hours,
            activity_level=req.activity_level,
            fever=req.fever,
            cough=req.cough,
            notes=req.notes or "",
            submitted_at=now
        )
        db.add(checkin_obj)
        action = "checkin_submitted"

    db.flush()

    # Suggestion 1: Trigger High-Risk / Critical Daily Check-in Alert (Pain >= 7 or Fever)
    if req.pain_level >= 7 or req.fever:
        try:
            from app.services.notification_service import notify_clinical_staff_critical_checkin
            patient_rec = db.get(Patient, target_patient_id)
            if patient_rec:
                notify_clinical_staff_critical_checkin(db=db, patient=patient_rec, checkin=checkin_obj)
        except Exception as e:
            import logging
            logging.getLogger("recovera.checkins").warning("Failed to dispatch critical checkin alert: %s", e)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action=action,
        target_type="daily_checkin",
        target_id=checkin_obj.id,
        details={
            "patient_id": target_patient_id,
            "checkin_date": target_date.isoformat(),
            "pain_level": req.pain_level,
            "fever": req.fever,
            "cough": req.cough
        }
    )
    db.commit()

    # Automatically re-evaluate clinical risk tier so emergency status reflects symptoms & vitals
    try:
        from app.services.risk_service import evaluate_patient_risk
        evaluate_patient_risk(db, patient_id=target_patient_id, window_days=7, force=True)
        db.commit()
    except Exception as e:
        import logging
        logging.getLogger("recovera.checkins").warning("Failed to auto-evaluate patient risk on checkin: %s", e)

    # Recovera AI 2.0: Update Digital Twin state on new check-in telemetry
    try:
        from app.services.digital_twin_service import compute_digital_twin_state
        compute_digital_twin_state(db, patient_id=target_patient_id, event_source="checkin_submission", event_id=checkin_obj.id)
        db.commit()
    except Exception as e:
        import logging
        logging.getLogger("recovera.checkins").warning("Failed to update digital twin state on checkin: %s", e)

    return _checkin_to_dict(checkin_obj)


@router.get("/api/checkins")
def get_my_checkins(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if not patient:
            return []
        items = db.scalars(
            select(DailyCheckIn)
            .where(DailyCheckIn.patient_id == patient.id)
            .order_by(DailyCheckIn.checkin_date.desc())
        ).all()
        return [_checkin_to_dict(c) for c in items]
    else:
        # For staff querying general check-ins
        items = db.scalars(
            select(DailyCheckIn).order_by(DailyCheckIn.submitted_at.desc()).limit(50)
        ).all()
        return [_checkin_to_dict(c) for c in items]


@router.get("/api/patients/{patient_id}/checkins")
@router.get("/api/patients/{patient_id}/check-ins")
def get_patient_checkins(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    patient = db.get(Patient, patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")

    if current_user.role == Role.PATIENT.value:
        if patient.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied.")
    elif current_user.role == Role.DOCTOR.value:
        if patient.assigned_doctor_id != current_user.id:
            from app.models.patient import PatientDoctorAssignment
            assignment = db.scalar(
                select(PatientDoctorAssignment).where(
                    PatientDoctorAssignment.patient_id == patient.id,
                    PatientDoctorAssignment.doctor_id == current_user.id
                )
            )
            if not assignment:
                raise HTTPException(status_code=403, detail="Doctor can only view check-ins for assigned patients.")

    items = db.scalars(
        select(DailyCheckIn)
        .where(DailyCheckIn.patient_id == patient_id)
        .order_by(DailyCheckIn.checkin_date.desc())
    ).all()
    return [_checkin_to_dict(c) for c in items]
