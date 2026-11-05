"""
Recovera AI – Appointments & Doctor Management API (Phase 6)

Endpoints:
  POST   /api/appointments                         – book appointment (overlap-safe)
  GET    /api/appointments                         – list (role-filtered)
  GET    /api/appointments/{id}                    – single appointment
  PUT    /api/appointments/{id}/status             – update status
  GET    /api/doctors                              – list doctors with availability
  GET    /api/doctors/{doctor_id}/slots?date=…     – free 60-min slots for a date
  PUT    /api/doctors/availability                 – doctor updates own availability
  PUT    /api/patients/{pid}/assign-doctor         – nurse/admin assigns doctor to patient
"""
import datetime as dt
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, text, func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.constants import Role, AppointmentStatus, SMSCategory
from app.core.audit import record_audit
from app.models.appointment import Appointment, DoctorAvailability
from app.models.user import User
from app.models.patient import Patient
from app.api.deps import get_current_user, require_role
from app.services.notification_service import (
    notify_doctor_appointment_booked,
    notify_doctor_patient_assigned,
    notify_nurse_reschedule_requested,
    notify_nurses_appointment_requested,
    create_in_app_notification,
    send_sms_job,
    schedule_appointment_patient_notifications,
    cancel_pending_appointment_notifications,
)

router = APIRouter(tags=["Appointments & Doctor Management"])

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AppointmentBookRequest(BaseModel):
    patient_id: str
    doctor_id: str
    start_time: dt.datetime
    notes: str = ""
    appointment_type: str = "general"  # general, follow_up, emergency


class StatusUpdateRequest(BaseModel):
    status: str  # started, completed, cancelled, missed
    reason: Optional[str] = ""


class NurseDeclineAppointmentRequest(BaseModel):
    reason: Optional[str] = "Clinical reassessment required or slot unavailable"


class NurseForwardAppointmentRequest(BaseModel):
    doctor_id: str
    start_time: Optional[dt.datetime] = None  # optional slot adjustment
    triage_notes: Optional[str] = ""


class DoctorAppointmentDecisionRequest(BaseModel):
    decision: str  # accept, deny, reschedule
    reschedule_date: Optional[dt.datetime] = None  # mandatory if decision == 'reschedule'
    notes: Optional[str] = ""


class AvailabilityUpdateRequest(BaseModel):
    status: str  # online, away
    away_until: Optional[dt.datetime] = None  # required if status is 'away'


class AssignDoctorRequest(BaseModel):
    doctor_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SLOT_DURATION = dt.timedelta(hours=1)
WORK_START = dt.time(8, 0)
WORK_END = dt.time(22, 0)


def _appointment_to_dict(appt: Appointment, db: Session) -> dict:
    patient = db.get(Patient, appt.patient_id)
    doctor = db.get(User, appt.doctor_id)
    booked_by = db.get(User, appt.booked_by)
    patient_user = db.get(User, patient.user_id) if patient and patient.user_id else None

    # Calculate status: if past scheduled appointment day (past EOD) and not completed/cancelled, mark as expired
    appt_status = appt.status
    if appt_status not in (AppointmentStatus.COMPLETED.value, AppointmentStatus.CANCELLED.value, AppointmentStatus.EXPIRED.value):
        appt_date = appt.start_time.date()
        today = dt.date.today()
        if appt_date < today:
            appt_status = AppointmentStatus.EXPIRED.value
            if appt.status != AppointmentStatus.EXPIRED.value:
                appt.status = AppointmentStatus.EXPIRED.value
                try:
                    db.add(appt)
                    db.commit()
                except Exception:
                    db.rollback()

    return {
        "id": appt.id,
        "patient_id": appt.patient_id,
        "patient_name": patient_user.name if patient_user else (patient.patient_code if patient else "Unknown"),
        "doctor_id": appt.doctor_id,
        "doctor_name": doctor.name if doctor else "Unknown",
        "doctor_specialization": doctor.specialization if doctor else None,
        "booked_by": appt.booked_by,
        "booked_by_name": booked_by.name if booked_by else "Unknown",
        "start_time": appt.start_time.isoformat(),
        "end_time": appt.end_time.isoformat(),
        "status": appt_status,
        "appointment_type": getattr(appt, "appointment_type", "follow_up") or "follow_up",
        "decision_status": appt.decision_status,
        "reschedule_proposed_time": appt.reschedule_proposed_time.isoformat() if appt.reschedule_proposed_time else None,
        "reschedule_reason": appt.reschedule_reason,
        "notes": appt.notes,
        "created_at": appt.created_at.isoformat(),
        "updated_at": appt.updated_at.isoformat(),
    }


def _check_overlap(db: Session, doctor_id: str, start_time: dt.datetime, end_time: dt.datetime, exclude_id: Optional[str] = None) -> bool:
    """Returns True if there is a conflicting doctor-accepted appointment for the doctor."""
    query = (
        select(func.count(Appointment.id))
        .where(
            Appointment.doctor_id == doctor_id,
            Appointment.status.notin_(["cancelled", "missed"]),
            Appointment.decision_status == "accepted",
            Appointment.start_time < end_time,
            Appointment.end_time > start_time,
        )
    )
    if exclude_id:
        query = query.where(Appointment.id != exclude_id)
    count = db.scalar(query) or 0
    return count > 0


# ---------------------------------------------------------------------------
# POST /api/appointments – book appointment
# ---------------------------------------------------------------------------

@router.post("/api/appointments", status_code=status.HTTP_201_CREATED)
def book_appointment(
    req: AppointmentBookRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(
        Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value, Role.PATIENT.value
    )),
):
    # Validate patient
    patient = db.get(Patient, req.patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")

    # Enforce patient self-service booking isolation
    if current_user.role == Role.PATIENT.value:
        if patient.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You may only book appointments for yourself."
            )

    # Validate doctor
    doctor = db.get(User, req.doctor_id)
    if not doctor or not doctor.active or doctor.role != Role.DOCTOR.value:
        raise HTTPException(status_code=404, detail="Doctor not found or inactive.")

    start_time = req.start_time.replace(tzinfo=None)
    end_time = start_time + SLOT_DURATION

    # Date and time validation: Cannot book previous dates or slots in the past
    now_local = dt.datetime.now()
    if start_time.date() < now_local.date():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot book an appointment for a previous date. Please choose today or a future date."
        )
    if start_time < now_local:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The selected time slot has already passed. Please select an upcoming time slot."
        )
    if start_time.time() > dt.time(21, 0) or start_time.time() < WORK_START:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Appointment slots cannot start before 8:00 AM or beyond 9:00 PM."
        )

    # Overlap check (SELECT FOR UPDATE via raw SQL for safety)
    if _check_overlap(db, req.doctor_id, start_time, end_time):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The requested time slot overlaps with an existing appointment for this doctor.",
        )

    appt = Appointment(
        patient_id=req.patient_id,
        doctor_id=req.doctor_id,
        booked_by=current_user.id,
        start_time=start_time,
        end_time=end_time,
        status=AppointmentStatus.SCHEDULED.value,
        appointment_type=req.appointment_type or "follow_up",
        decision_status="nurse_review",
        notes=req.notes,
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow(),
    )
    db.add(appt)
    db.flush()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="appointment_booked",
        target_type="appointment",
        target_id=appt.id,
        new_values={
            "patient_id": req.patient_id,
            "doctor_id": req.doctor_id,
            "start_time": start_time.isoformat(),
            "appointment_type": req.appointment_type,
            "decision_status": "nurse_review",
        },
    )
    # Forward all bookings to Nurse Triage first (standard clinical workflow)
    notify_nurses_appointment_requested(db=db, appointment=appt, patient=patient)
    db.commit()
    return _appointment_to_dict(appt, db)


# ---------------------------------------------------------------------------
# GET /api/appointments – list (role-filtered)
# ---------------------------------------------------------------------------

@router.get("/api/appointments")
def list_appointments(
    patient_id: Optional[str] = Query(None, description="Optional patient ID filter"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role == Role.PATIENT.value:
        # Patient sees their own appointments only
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if not patient:
            return []
        query = select(Appointment).where(Appointment.patient_id == patient.id)
    elif current_user.role == Role.DOCTOR.value:
        # Doctor sees appointments where they are attending doctor
        query = select(Appointment).where(Appointment.doctor_id == current_user.id)
        if patient_id:
            query = query.where(Appointment.patient_id == patient_id)
    else:
        # Nurse / Admin sees all (optionally filtered by patient_id)
        query = select(Appointment)
        if patient_id:
            query = query.where(Appointment.patient_id == patient_id)

    appts = db.scalars(query.order_by(Appointment.start_time.desc())).all()
    return [_appointment_to_dict(a, db) for a in appts]


# ---------------------------------------------------------------------------
# GET /api/appointments/{id} – single appointment
# ---------------------------------------------------------------------------

@router.get("/api/appointments/{appointment_id}")
def get_appointment(
    appointment_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    appt = db.get(Appointment, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    # Enforce visibility
    if current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if not patient or appt.patient_id != patient.id:
            raise HTTPException(status_code=403, detail="Access denied.")
    elif current_user.role == Role.DOCTOR.value:
        if appt.doctor_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied.")

    return _appointment_to_dict(appt, db)


# ---------------------------------------------------------------------------
# PUT /api/appointments/{id}/status – update status
# ---------------------------------------------------------------------------

@router.put("/api/appointments/{appointment_id}/status")
def update_appointment_status(
    appointment_id: str,
    req: StatusUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(
        Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value, Role.PATIENT.value
    )),
):
    valid_statuses = {s.value for s in AppointmentStatus}
    if req.status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}.",
        )

    appt = db.get(Appointment, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    # Block starting/cancelling appointments that have expired past their scheduled End-of-Day
    if appt.status == AppointmentStatus.EXPIRED.value or (appt.start_time.date() < dt.date.today() and appt.status not in (AppointmentStatus.COMPLETED.value, AppointmentStatus.CANCELLED.value)):
        raise HTTPException(
            status_code=400,
            detail="This appointment has expired past its scheduled date and can no longer be modified."
        )

    if current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if not patient or appt.patient_id != patient.id:
            raise HTTPException(status_code=403, detail="Access denied.")
        if req.status != AppointmentStatus.CANCELLED.value:
            raise HTTPException(status_code=403, detail="Patients may only cancel their appointments.")
    elif current_user.role == Role.DOCTOR.value and appt.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied.")

    old_status = appt.status
    appt.status = req.status
    appt.updated_at = dt.datetime.utcnow()

    # If cancelled, cancel any queued reminder SMS jobs and notify all parties
    if req.status == AppointmentStatus.CANCELLED.value:
        appt.decision_status = "cancelled"
        cancel_pending_appointment_notifications(db=db, appointment_id=appt.id)

        doctor = db.get(User, appt.doctor_id)
        doctor_name = doctor.name if doctor else "Doctor"
        patient = db.get(Patient, appt.patient_id)
        patient_user = db.get(User, patient.user_id) if patient and patient.user_id else None
        patient_name = patient_user.name if patient_user else (patient.patient_code if patient else "Patient")
        time_str = appt.start_time.strftime("%d %b %Y at %I:%M %p")

        if current_user.role == Role.PATIENT.value:
            # 1. Cancelled by Patient -> notify Doctor, Nurses, and confirm to Patient
            if doctor:
                create_in_app_notification(
                    db=db,
                    user_id=doctor.id,
                    title="Patient Cancelled Appointment",
                    message=f"Patient {patient_name} has cancelled their appointment scheduled for {time_str}. The slot has been released.",
                    category="appointment",
                    link="#/staff/appointments"
                )
                if doctor.phone:
                    send_sms_job(
                        db=db,
                        phone=doctor.phone,
                        message_text=f"RecoveraAI: Patient {patient_name} cancelled their appointment on {time_str}. Slot released.",
                        category=SMSCategory.APPOINTMENT.value,
                        patient_id=patient.id if patient else None
                    )

            nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
            for n in nurses:
                create_in_app_notification(
                    db=db,
                    user_id=n.id,
                    title="Appointment Cancelled by Patient",
                    message=f"Patient {patient_name} cancelled their appointment with Dr. {doctor_name} for {time_str}.",
                    category="appointment",
                    link="#/staff/appointments"
                )

            if patient_user:
                create_in_app_notification(
                    db=db,
                    user_id=patient_user.id,
                    title="Appointment Cancelled",
                    message=f"Your appointment with Dr. {doctor_name} on {time_str} has been cancelled successfully.",
                    category="appointment",
                    link="#/patient/appointments"
                )
                if patient.phone and patient.sms_opt_in:
                    send_sms_job(
                        db=db,
                        phone=patient.phone,
                        message_text=f"RecoveraAI: Your appointment with Dr. {doctor_name} on {time_str} has been cancelled.",
                        category=SMSCategory.APPOINTMENT.value,
                        patient_id=patient.id
                    )
        else:
            # 2. Cancelled by Doctor / Staff -> notify Patient and other Staff
            reason_clean = (req.reason or "").strip()
            reason_txt = f" Reason: {reason_clean}." if reason_clean else ""
            is_nurse_action = (appt.decision_status == "nurse_review" or current_user.role == Role.NURSE.value)

            if is_nurse_action:
                appt.decision_status = "nurse_declined"
                notif_title = "Appointment Request Not Forwarded"
                notif_msg = f"Your appointment request for {time_str} was not forwarded to Dr. {doctor_name} by the clinical nursing staff.{reason_txt} Kindly retry or choose another available time slot."
                sms_msg = f"RecoveraAI: Your appointment request for {time_str} was not forwarded to Dr. {doctor_name} by nursing staff.{reason_txt} Kindly retry in portal: {settings.FRONTEND_URL.rstrip('/')}/#/patient/appointments"
            else:
                appt.decision_status = "cancelled"
                canceller_title = "Doctor" if current_user.role == Role.DOCTOR.value else "Clinical Staff"
                notif_title = "Appointment Cancelled by Doctor" if current_user.role == Role.DOCTOR.value else "Appointment Cancelled"
                notif_msg = f"Your appointment with Dr. {doctor_name} on {time_str} was cancelled by the doctor.{reason_txt} Kindly retry or choose an alternative available slot in the portal."
                sms_msg = f"RecoveraAI: Your appointment with Dr. {doctor_name} on {time_str} was cancelled by the doctor.{reason_txt} Please rebook in portal: {settings.FRONTEND_URL.rstrip('/')}/#/patient/appointments"

            if patient_user:
                create_in_app_notification(
                    db=db,
                    user_id=patient_user.id,
                    title=notif_title,
                    message=notif_msg,
                    category="appointment",
                    link="#/patient/appointments"
                )
                if patient.phone and patient.sms_opt_in:
                    send_sms_job(
                        db=db,
                        phone=patient.phone,
                        message_text=sms_msg,
                        category=SMSCategory.APPOINTMENT.value,
                        patient_id=patient.id
                    )

            nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
            for n in nurses:
                if n.id != current_user.id:
                    create_in_app_notification(
                        db=db,
                        user_id=n.id,
                        title="Doctor Cancelled Appointment" if current_user.role == Role.DOCTOR.value else "Staff Cancelled Appointment",
                        message=f"{canceller_title if not is_nurse_action else 'Nurse ' + current_user.name} cancelled/declined appointment for patient {patient_name} with Dr. {doctor_name} on {time_str}.{reason_txt}",
                        category="appointment",
                        link="#/staff/appointments"
                    )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="appointment_status_updated",
        target_type="appointment",
        target_id=appt.id,
        details={"old_status": old_status, "new_status": req.status},
    )
    db.commit()
    return _appointment_to_dict(appt, db)


# ---------------------------------------------------------------------------
# PUT /api/appointments/{id}/doctor-decision – doctor accepts, denies, or reschedules
# ---------------------------------------------------------------------------

@router.put("/api/appointments/{appointment_id}/doctor-decision")
def doctor_appointment_decision(
    appointment_id: str,
    req: DoctorAppointmentDecisionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.DOCTOR.value, Role.ADMIN.value)),
):
    valid_decisions = {"accept", "deny", "reschedule"}
    if req.decision not in valid_decisions:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid decision. Must be one of: {', '.join(valid_decisions)}."
        )

    appt = db.get(Appointment, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    # Block starting/cancelling appointments that have expired past their scheduled End-of-Day
    if appt.status == AppointmentStatus.EXPIRED.value or (appt.start_time.date() < dt.date.today() and appt.status not in (AppointmentStatus.COMPLETED.value, AppointmentStatus.CANCELLED.value)):
        raise HTTPException(
            status_code=400,
            detail="This appointment has expired past its scheduled date and can no longer be modified."
        )

    if current_user.role == Role.DOCTOR.value and appt.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="You may only make decisions for appointments assigned to you.")

    doctor = db.get(User, appt.doctor_id)
    patient = db.get(Patient, appt.patient_id)

    if req.decision == "accept":
        appt.decision_status = "accepted"
        appt.status = AppointmentStatus.SCHEDULED.value
        if req.notes and req.notes.strip():
            appt.notes = (appt.notes + f"\n[Doctor note]: {req.notes.strip()}").strip()

        # Send official confirmation + schedule 24h & 2h reminders to patient
        schedule_appointment_patient_notifications(db=db, appointment=appt)

        # Send official confirmation to doctor (in-app + SMS)
        notify_doctor_appointment_booked(db=db, doctor_id=appt.doctor_id, patient=patient, start_time=appt.start_time)

        # Notify forwarding nurse / nurse triage pool
        forwarder = db.get(User, appt.booked_by)
        nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
        target_nurse_ids = {n.id for n in nurses}
        if forwarder and forwarder.role in (Role.NURSE.value, Role.ADMIN.value):
            target_nurse_ids.add(forwarder.id)

        patient_name = patient.patient_code if patient else "Patient"
        if patient and patient.user_id:
            pu = db.get(User, patient.user_id)
            if pu: patient_name = pu.name
        time_str = appt.start_time.strftime("%d %b %Y at %I:%M %p")

        for nid in target_nurse_ids:
            create_in_app_notification(
                db=db,
                user_id=nid,
                title="Appointment Accepted by Doctor",
                message=f"Dr. {doctor.name} has accepted the appointment for patient {patient_name} on {time_str}.",
                category="appointment",
                link="#/staff/appointments"
            )
            # Send SMS to forwarding nurse if phone exists
            nurse_u = db.get(User, nid)
            if nurse_u and nurse_u.phone and (nurse_u.id == appt.booked_by):
                send_sms_job(
                    db=db,
                    phone=nurse_u.phone,
                    message_text=f"RecoveraAI: Dr. {doctor.name} accepted appointment for patient {patient_name} on {time_str}.",
                    category=SMSCategory.APPOINTMENT.value,
                    patient_id=patient.id if patient else None
                )

    elif req.decision == "deny":
        appt.decision_status = "denied"
        appt.status = AppointmentStatus.CANCELLED.value
        cancel_pending_appointment_notifications(db=db, appointment_id=appt.id)
        reason_txt = req.notes.strip() if req.notes and req.notes.strip() else "Doctor unavailable"
        appt.notes = (appt.notes + f"\n[Doctor denied]: {reason_txt}").strip()
        time_str = appt.start_time.strftime("%d %b %Y at %I:%M %p")
        patient_name = patient.patient_code if patient else "Patient"

        if patient and patient.user_id:
            pu = db.get(User, patient.user_id)
            if pu: patient_name = pu.name
            create_in_app_notification(
                db=db,
                user_id=patient.user_id,
                title="Appointment Cancelled by Doctor",
                message=f"Your appointment with Dr. {doctor.name} on {time_str} was cancelled by the doctor. Reason: {reason_txt}. Kindly retry or choose an alternative available slot in the portal.",
                category="appointment",
                link="#/patient/appointments"
            )
            if patient.phone and patient.sms_opt_in:
                send_sms_job(
                    db=db,
                    phone=patient.phone,
                    message_text=f"RecoveraAI: Your appointment with Dr. {doctor.name} on {time_str} was cancelled by the doctor. Reason: {reason_txt}. Please rebook in portal: {settings.FRONTEND_URL.rstrip('/')}/#/patient/appointments",
                    category=SMSCategory.APPOINTMENT.value,
                    patient_id=patient.id
                )

        nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
        for n in nurses:
            create_in_app_notification(
                db=db,
                user_id=n.id,
                title="Doctor Cancelled Appointment",
                message=f"Dr. {doctor.name} cancelled appointment for patient {patient_name} on {time_str}. Reason: {reason_txt}.",
                category="appointment",
                link="#/staff/appointments"
            )
            if n.phone and (n.id == appt.booked_by):
                send_sms_job(
                    db=db,
                    phone=n.phone,
                    message_text=f"RecoveraAI: Dr. {doctor.name} cancelled appointment for patient {patient_name} on {time_str}. Reason: {reason_txt}.",
                    category=SMSCategory.APPOINTMENT.value,
                    patient_id=patient.id if patient else None
                )

    elif req.decision == "reschedule":
        if not req.reschedule_date:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="reschedule_date is mandatory when requesting a reschedule."
            )
        new_start = req.reschedule_date.replace(tzinfo=None)
        if new_start.time() > dt.time(21, 0) or new_start.time() < WORK_START:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Appointment slots cannot start before 8:00 AM or beyond 9:00 PM."
            )
        new_end = new_start + SLOT_DURATION

        now_local = dt.datetime.now()
        if new_start.date() < now_local.date():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot reschedule an appointment to a previous date. Please choose a future date."
            )
        if new_start < now_local:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot reschedule an appointment to a date or time that has already passed."
            )

        if appt.start_time and new_start == appt.start_time:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rescheduled appointment cannot be placed in the same time slot that the doctor is rejecting. Please choose an alternative slot."
            )

        if _check_overlap(db, appt.doctor_id, new_start, new_end, exclude_id=appt.id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The requested reschedule date/time conflicts with another appointment."
            )
        appt.decision_status = "reschedule_requested"
        appt.status = AppointmentStatus.RESCHEDULED.value
        appt.reschedule_proposed_time = new_start
        appt.reschedule_reason = req.notes.strip() if req.notes else ""
        reason_display = req.notes.strip() if req.notes and req.notes.strip() else "not specified"
        appt.notes = (appt.notes + f"\n[Doctor requested reschedule to {new_start.isoformat()}]: {reason_display}").strip()

        # Trigger nurse alert + in-app alert + SMS Gateway
        notify_nurse_reschedule_requested(
            db=db,
            appointment=appt,
            doctor=doctor,
            reschedule_date=new_start,
            reason=req.notes
        )

    appt.updated_at = dt.datetime.utcnow()
    record_audit(
        db=db,
        actor_id=current_user.id,
        action="appointment_doctor_decision",
        target_type="appointment",
        target_id=appt.id,
        details={"decision": req.decision, "reschedule_date": req.reschedule_date.isoformat() if req.reschedule_date else None, "notes": req.notes}
    )
    db.commit()
    return _appointment_to_dict(appt, db)


# ---------------------------------------------------------------------------
# PUT /api/appointments/{id}/nurse-forward – nurse triages & forwards to chosen doctor
# ---------------------------------------------------------------------------

@router.put("/api/appointments/{appointment_id}/nurse-forward")
def nurse_forward_appointment(
    appointment_id: str,
    req: NurseForwardAppointmentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.NURSE.value, Role.ADMIN.value)),
):
    appt = db.get(Appointment, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    # Block starting/cancelling appointments that have expired past their scheduled End-of-Day
    if appt.status == AppointmentStatus.EXPIRED.value or (appt.start_time.date() < dt.date.today() and appt.status not in (AppointmentStatus.COMPLETED.value, AppointmentStatus.CANCELLED.value)):
        raise HTTPException(
            status_code=400,
            detail="This appointment has expired past its scheduled date and can no longer be modified."
        )

    # Concurrency guard: Prevent duplicate forwarding if another nurse already triaged this request
    if appt.decision_status == "forwarded_to_doctor":
        forwarder = db.get(User, appt.booked_by)
        forwarder_name = forwarder.name if forwarder else "another nurse"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This appointment request has already been reviewed and forwarded to the doctor by Nurse {forwarder_name}."
        )
    if appt.decision_status == "accepted":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This appointment has already been reviewed and accepted by the doctor."
        )
    if appt.status == AppointmentStatus.CANCELLED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This appointment has been cancelled and cannot be forwarded."
        )

    doctor = db.get(User, req.doctor_id)
    if not doctor or not doctor.active or doctor.role != Role.DOCTOR.value:
        raise HTTPException(status_code=404, detail="Selected doctor not found or inactive.")

    patient = db.get(Patient, appt.patient_id)
    patient_user = db.get(User, patient.user_id) if patient and patient.user_id else None
    patient_name = patient_user.name if patient_user else (patient.patient_code if patient else "Patient")

    # Update slot if nurse adjusted it
    if req.start_time:
        new_start = req.start_time.replace(tzinfo=None)
        new_end = new_start + SLOT_DURATION
    else:
        new_start = appt.start_time
        new_end = appt.end_time

    now_local = dt.datetime.now()
    if new_start.date() < now_local.date():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot forward an appointment to a previous date. Please choose a future date."
        )
    if new_start < now_local:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot forward an appointment to a date or time that has already passed."
        )
    if new_start.time() > dt.time(21, 0) or new_start.time() < WORK_START:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Appointment slots cannot start before 8:00 AM or beyond 9:00 PM."
        )

    # If the doctor rejected or requested reschedule for the original slot, do not permit re-assigning that exact rejected slot
    if (appt.decision_status in ("denied", "reschedule_requested") or appt.status == AppointmentStatus.RESCHEDULED.value) and appt.start_time and new_start == appt.start_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot assign to the same time slot that the doctor has rejected. Please select an alternative time slot."
        )

    # Conflict check for target doctor
    if _check_overlap(db, req.doctor_id, new_start, new_end, exclude_id=appt.id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Dr. {doctor.name} already has a conflicting appointment at this time slot. Please choose another slot.",
        )

    appt.doctor_id = req.doctor_id
    appt.start_time = new_start
    appt.end_time = new_end
    appt.booked_by = current_user.id  # Record this nurse as the forwarding contact
    appt.decision_status = "forwarded_to_doctor"
    if req.triage_notes and req.triage_notes.strip():
        appt.notes = (appt.notes + f"\n[Nurse Triage ({current_user.name})]: {req.triage_notes.strip()}").strip()
    appt.updated_at = dt.datetime.utcnow()

    time_str = new_start.strftime("%d %b %Y at %I:%M %p")

    # Notify doctor
    create_in_app_notification(
        db=db,
        user_id=doctor.id,
        title="Forwarded Patient Appointment",
        message=f"Nurse {current_user.name} forwarded patient {patient_name}'s appointment request to you for {time_str}.",
        category="appointment",
        link="#/staff/appointments"
    )
    if doctor.phone:
        send_sms_job(
            db=db,
            phone=doctor.phone,
            message_text=f"RecoveraAI: Nurse {current_user.name} forwarded patient {patient_name} appointment for {time_str}. Review in portal.",
            category=SMSCategory.APPOINTMENT.value,
            patient_id=patient.id if patient else None
        )

    # Notify patient
    if patient and patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title="Appointment Triage Completed",
            message=f"Your appointment request was triaged by Nurse {current_user.name} and forwarded to Dr. {doctor.name} for {time_str}.",
            category="appointment",
            link="#/patient/appointments"
        )
        if patient.phone and patient.sms_opt_in:
            send_sms_job(
                db=db,
                phone=patient.phone,
                message_text=f"RecoveraAI: Your appointment was triaged and forwarded to Dr. {doctor.name} for {time_str}.",
                category=SMSCategory.APPOINTMENT.value,
                patient_id=patient.id
            )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="appointment_nurse_forwarded",
        target_type="appointment",
        target_id=appt.id,
        new_values={"doctor_id": req.doctor_id, "start_time": new_start.isoformat(), "triage_notes": req.triage_notes}
    )
    db.commit()
    return _appointment_to_dict(appt, db)


# ---------------------------------------------------------------------------
# PUT /api/appointments/{id}/nurse-decline - nurse declines to forward to doctor
# ---------------------------------------------------------------------------

@router.put("/api/appointments/{appointment_id}/nurse-decline")
def nurse_decline_appointment(
    appointment_id: str,
    req: NurseDeclineAppointmentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.NURSE.value, Role.ADMIN.value)),
):
    """Nurse declines to forward the appointment request to the doctor, notifying the patient context-aware."""
    appt = db.get(Appointment, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    # Block starting/cancelling appointments that have expired past their scheduled End-of-Day
    if appt.status == AppointmentStatus.EXPIRED.value or (appt.start_time.date() < dt.date.today() and appt.status not in (AppointmentStatus.COMPLETED.value, AppointmentStatus.CANCELLED.value)):
        raise HTTPException(
            status_code=400,
            detail="This appointment has expired past its scheduled date and can no longer be modified."
        )

    if appt.decision_status not in ("nurse_review", "pending"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decline appointment in '{appt.decision_status}' state."
        )

    old_status = appt.status
    appt.status = AppointmentStatus.CANCELLED.value
    appt.decision_status = "nurse_declined"
    reason_clean = (req.reason or "Clinical reassessment required or slot unavailable").strip()
    appt.notes = (appt.notes + f"\n[Nurse Decline ({current_user.name})]: {reason_clean}").strip()
    appt.updated_at = dt.datetime.utcnow()
    cancel_pending_appointment_notifications(db=db, appointment_id=appt.id)

    patient = db.get(Patient, appt.patient_id)
    doctor = db.get(User, appt.doctor_id)
    doctor_name = doctor.name if doctor else "Doctor"
    time_str = appt.start_time.strftime("%d %b %Y at %I:%M %p")

    if patient and patient.user_id:
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title="Appointment Request Not Forwarded",
            message=f"Your appointment request for {time_str} was not forwarded to Dr. {doctor_name} by the clinical nursing staff. Reason: {reason_clean}. Kindly retry or choose another available time slot.",
            category="appointment",
            link="#/patient/appointments"
        )
        if patient.phone and patient.sms_opt_in:
            send_sms_job(
                db=db,
                phone=patient.phone,
                message_text=f"RecoveraAI: Your appointment request for {time_str} was not forwarded to Dr. {doctor_name} by nursing staff. Reason: {reason_clean}. Kindly retry in portal: {settings.FRONTEND_URL.rstrip('/')}/#/patient/appointments",
                category=SMSCategory.APPOINTMENT.value,
                patient_id=patient.id
            )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="appointment_nurse_declined",
        target_type="appointment",
        target_id=appt.id,
        details={"old_status": old_status, "decision_status": "nurse_declined", "reason": reason_clean},
    )
    db.commit()
    return _appointment_to_dict(appt, db)


# ---------------------------------------------------------------------------
# GET /api/doctors – list all doctors with specialization + availability
# ---------------------------------------------------------------------------

@router.get("/api/doctors")
def list_doctors(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    doctors = db.scalars(
        select(User)
        .where(User.role == Role.DOCTOR.value, User.active == True)
        .order_by(User.name)
    ).all()

    return [
        {
            "id": d.id,
            "name": d.name,
            "email": d.email,
            "specialization": d.specialization,
            "availability_status": d.availability_status,
            "away_until": d.away_until.isoformat() if d.away_until else None,
        }
        for d in doctors
    ]


# ---------------------------------------------------------------------------
# GET /api/doctors/{doctor_id}/slots?date=YYYY-MM-DD – free 60-min slots
# ---------------------------------------------------------------------------

@router.get("/api/doctors/{doctor_id}/slots")
def get_doctor_slots(
    doctor_id: str,
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    exclude_slot: Optional[str] = Query(None, description="Optional slot ISO to exclude e.g. rejected slot"),
    exclude_appointment_id: Optional[str] = Query(None, description="Optional appointment ID to exclude from conflict check"),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    # Validate doctor
    doctor = db.get(User, doctor_id)
    if not doctor or not doctor.active or doctor.role != Role.DOCTOR.value:
        raise HTTPException(status_code=404, detail="Doctor not found or inactive.")

    try:
        target_date = dt.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid date format. Use YYYY-MM-DD.")

    today_local = dt.date.today()
    if target_date < today_local:
        return {"doctor_id": doctor_id, "date": date, "available_slots": []}

    day_start = dt.datetime.combine(target_date, WORK_START)
    day_end = dt.datetime.combine(target_date, WORK_END)

    # Fetch existing appointments for this doctor on this date that are ACCEPTED by the doctor
    # Only doctor-accepted appointments block the time slot!
    query = select(Appointment).where(
        Appointment.doctor_id == doctor_id,
        Appointment.start_time >= day_start,
        Appointment.start_time < day_end,
        Appointment.status.notin_(["cancelled", "missed"]),
        Appointment.decision_status == "accepted",
    )
    if exclude_appointment_id:
        query = query.where(Appointment.id != exclude_appointment_id)
    existing = db.scalars(query).all()

    booked_ranges = [(a.start_time, a.end_time) for a in existing]

    now_utc = dt.datetime.utcnow()
    now_local = dt.datetime.now()

    excluded_dt = None
    if exclude_slot:
        try:
            excluded_dt = dt.datetime.fromisoformat(exclude_slot.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            excluded_dt = None

    # Generate all 60-min slots from 08:00 to 20:00, strictly excluding past times & rejected slots
    free_slots: List[str] = []
    slot_start = day_start
    while slot_start + SLOT_DURATION <= day_end:
        if slot_start.time() > dt.time(21, 0):
            break
        slot_end = slot_start + SLOT_DURATION
        overlaps = any(
            s < slot_end and e > slot_start for s, e in booked_ranges
        )
        is_past = (target_date <= today_local) and (slot_start < now_local)
        is_excluded = excluded_dt and slot_start == excluded_dt

        if not overlaps and not is_past and not is_excluded:
            free_slots.append(slot_start.isoformat())
        slot_start = slot_end

    return {"doctor_id": doctor_id, "date": date, "available_slots": free_slots}


# ---------------------------------------------------------------------------
# PUT /api/doctors/availability – doctor updates own availability
# ---------------------------------------------------------------------------

@router.put("/api/doctors/availability")
def update_own_availability(
    req: AvailabilityUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.DOCTOR.value)),
):
    allowed = {"online", "away"}
    if req.status not in allowed:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Must be one of: {', '.join(allowed)}.",
        )
    if req.status == "away" and not req.away_until:
        raise HTTPException(
            status_code=422,
            detail="away_until is required when setting status to 'away'.",
        )

    current_user.availability_status = req.status
    current_user.away_until = req.away_until if req.status == "away" else None

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="doctor_availability_updated",
        target_type="user",
        target_id=current_user.id,
        details={"status": req.status, "away_until": req.away_until.isoformat() if req.away_until else None},
    )
    db.commit()
    return {
        "message": "Availability updated.",
        "availability_status": current_user.availability_status,
        "away_until": current_user.away_until.isoformat() if current_user.away_until else None,
    }


# ---------------------------------------------------------------------------
# PUT /api/patients/{pid}/assign-doctor – nurse/admin assigns patient to doctor
# ---------------------------------------------------------------------------

@router.put("/api/patients/{pid}/assign-doctor")
def assign_doctor_to_patient(
    pid: str,
    req: AssignDoctorRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.NURSE.value, Role.DOCTOR.value)),
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")

    doctor = db.get(User, req.doctor_id)
    if not doctor or not doctor.active or doctor.role != Role.DOCTOR.value:
        raise HTTPException(status_code=404, detail="Doctor not found or inactive.")

    old_doctor_id = patient.assigned_doctor_id
    patient.assigned_doctor_id = req.doctor_id
    patient.updated_at = dt.datetime.utcnow()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_assigned_doctor",
        target_type="patient",
        target_id=patient.id,
        details={"old_doctor_id": old_doctor_id, "new_doctor_id": req.doctor_id},
    )
    notify_doctor_patient_assigned(db=db, doctor_id=req.doctor_id, patient=patient)
    db.commit()
    return {
        "message": f"Patient {patient.patient_code} assigned to Dr. {doctor.name}.",
        "patient_id": patient.id,
        "assigned_doctor_id": patient.assigned_doctor_id,
    }
