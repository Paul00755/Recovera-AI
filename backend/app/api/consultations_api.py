"""
Recovera AI – Consultation API (Phase 6)

Endpoints:
  POST  /api/consultations                       – create consultation request (15 min default)
  GET   /api/consultations                       – list consultations for current user
  GET   /api/consultations/active                – active consultation for current user
  PUT   /api/consultations/{id}/start            – doctor starts consultation
  PUT   /api/consultations/{id}/extend           – doctor extends by 5/10/15 min
  PUT   /api/consultations/{id}/end              – end consultation
  POST  /api/consultations/{id}/messages         – send chat message
  GET   /api/consultations/{id}/messages         – get chat messages
"""
import uuid
import json
import datetime as dt
from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.constants import Role, ConsultationStatus, ConsultationMode
from app.core.audit import record_audit
from app.models.consultation import Consultation, ConsultationMessage, ConsultationSignal, ConsultationParticipant
from app.models.user import User
from app.models.patient import Patient
from app.api.deps import get_current_user, require_role
from app.services.notification_service import (
    notify_doctor_consultation_requested,
    notify_patient_consultation_opened,
    create_in_app_notification,
)

router = APIRouter(tags=["Consultations"])

# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ConsultationCreateRequest(BaseModel):
    patient_id: str
    doctor_id: str
    mode: str = "video"  # audio, video, chat
    notes: str = ""


class ExtendRequest(BaseModel):
    minutes: int  # must be 5, 10, or 15


class MessageRequest(BaseModel):
    message: str


class SignalRequest(BaseModel):
    signal_type: Optional[str] = None  # "offer", "answer", "ice-candidate", "call-ended", "live-caption"
    signal_data: Optional[str] = None  # JSON-encoded payload or string
    type: Optional[str] = None
    payload: Optional[Any] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any):
        if isinstance(data, dict):
            st = data.get("signal_type") or data.get("type") or "signal"
            sd = data.get("signal_data")
            if sd is None and "payload" in data:
                p = data["payload"]
                sd = json.dumps(p) if not isinstance(p, str) else p
            return {
                "signal_type": st,
                "signal_data": sd or "",
                "type": st,
                "payload": data.get("payload")
            }
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_EXPIRY_MINUTES = 15


def _format_utc(d: Optional[dt.datetime]) -> Optional[str]:
    """Serializes datetime to ISO-8601 string explicitly with 'Z' UTC suffix."""
    if not d:
        return None
    iso = d.isoformat()
    return iso + "Z" if not iso.endswith("Z") else iso


def _room_id() -> str:
    return f"room_{uuid.uuid4().hex}"


def _consultation_to_dict(c: Consultation, db: Session) -> dict:
    patient = db.get(Patient, c.patient_id)
    doctor = db.get(User, c.doctor_id)
    patient_user = db.get(User, patient.user_id) if patient and patient.user_id else None
    
    # Check active participants
    participants = db.scalars(
        select(ConsultationParticipant).where(
            ConsultationParticipant.consultation_id == c.id,
            ConsultationParticipant.left_at.is_(None)
        )
    ).all()
    roles = {p.role for p in participants}
    both_joined = ("doctor" in roles and "patient" in roles) or len(participants) >= 2

    return {
        "id": c.id,
        "room_id": c.room_id,
        "patient_id": c.patient_id,
        "patient_user_id": patient.user_id if patient else None,
        "patient_name": patient_user.name if patient_user else (patient.patient_code if patient else "Unknown"),
        "doctor_id": c.doctor_id,
        "doctor_name": doctor.name if doctor else "Unknown",
        "initiated_by": c.initiated_by,
        "mode": c.mode,
        "status": c.status,
        "both_joined": both_joined,
        "active_participants_count": len(participants),
        "started_at": _format_utc(c.started_at),
        "expires_at": _format_utc(c.expires_at),
        "ended_at": _format_utc(c.ended_at),
        "created_at": _format_utc(c.created_at),
    }


def _visible_to_user(c: Consultation, current_user: User, db: Session) -> bool:
    """Return True if the current user is allowed to see this consultation."""
    if current_user.role in (Role.ADMIN.value, Role.NURSE.value, Role.DOCTOR.value):
        return True
    if current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        return patient is not None and c.patient_id == patient.id
    return False


def _check_and_expire_consultations(db: Session) -> None:
    """Enforces server-side 15-minute auto-termination for consultations past their expiration time."""
    now = dt.datetime.utcnow()
    expired = db.scalars(
        select(Consultation).where(
            Consultation.status == ConsultationStatus.ACTIVE.value,
            Consultation.expires_at.is_not(None),
            Consultation.expires_at < now,
        )
    ).all()
    for c in expired:
        c.status = ConsultationStatus.COMPLETED.value
        c.ended_at = c.expires_at or now
        doc = db.get(User, c.doctor_id)
        if doc and doc.availability_status == "busy":
            doc.availability_status = "online"
        record_audit(
            db=db,
            actor_id=c.doctor_id,
            action="consultation_auto_terminated",
            target_type="consultation",
            target_id=c.id,
            details={"reason": "15_min_countdown_expired", "ended_at": _format_utc(c.ended_at)}
        )
    if expired:
        db.commit()


# ---------------------------------------------------------------------------
# POST /api/consultations – create consultation request
# ---------------------------------------------------------------------------

@router.post("/api/consultations", status_code=status.HTTP_201_CREATED)
def create_consultation(
    req: ConsultationCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(
        Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value, Role.PATIENT.value
    )),
):
    _check_and_expire_consultations(db)

    # Validate mode
    valid_modes = {m.value for m in ConsultationMode}
    if req.mode not in valid_modes:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode. Must be one of: {', '.join(valid_modes)}.",
        )

    patient = db.get(Patient, req.patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")

    doctor = db.get(User, req.doctor_id)
    if not doctor or not doctor.active or doctor.role != Role.DOCTOR.value:
        raise HTTPException(status_code=404, detail="Doctor not found or inactive.")

    if current_user.role == Role.PATIENT.value:
        if patient.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Patients may only request consultations for themselves.")

    if current_user.role == Role.DOCTOR.value:
        if req.doctor_id != current_user.id:
            raise HTTPException(
                status_code=403,
                detail="Doctors may only request consultations where they are the attending doctor."
            )
        if patient.assigned_doctor_id != current_user.id:
            from app.models.patient import PatientDoctorAssignment
            is_assigned = db.scalar(
                select(PatientDoctorAssignment).where(
                    PatientDoctorAssignment.patient_id == patient.id,
                    PatientDoctorAssignment.doctor_id == current_user.id,
                    PatientDoctorAssignment.is_active == True
                )
            )
            if not is_assigned:
                raise HTTPException(
                    status_code=403,
                    detail="Doctors may only request consultations with patients assigned to their care."
                )

    now = dt.datetime.utcnow()
    consultation = Consultation(
        room_id=_room_id(),
        patient_id=req.patient_id,
        doctor_id=req.doctor_id,
        initiated_by=current_user.id,
        mode=req.mode,
        status=ConsultationStatus.WAITING.value,
        started_at=None,
        expires_at=now + dt.timedelta(minutes=DEFAULT_EXPIRY_MINUTES),
        created_at=now,
    )
    db.add(consultation)
    db.flush()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="consultation_created",
        target_type="consultation",
        target_id=consultation.id,
        new_values={
            "patient_id": req.patient_id,
            "doctor_id": req.doctor_id,
            "mode": req.mode,
        },
    )

    # Role-based notification routing
    # When doctor calls/books a consultation with patient -> notify patient
    if current_user.role == Role.DOCTOR.value:
        notify_patient_consultation_opened(
            db=db,
            patient=patient,
            doctor=doctor,
            room_id=consultation.room_id,
            mode=req.mode
        )
    elif current_user.role in (Role.NURSE.value, Role.ADMIN.value):
        # When nurse acts as bridge connecting patient and doctor:
        # Alert BOTH the patient and the doctor immediately!
        notify_patient_consultation_opened(
            db=db,
            patient=patient,
            doctor=doctor,
            room_id=consultation.room_id,
            mode=req.mode
        )
        initiator_name = f"Nurse {current_user.name}" if current_user.role == Role.NURSE.value else f"Staff {current_user.name}"
        notify_doctor_consultation_requested(
            db=db,
            doctor_id=req.doctor_id,
            patient=patient,
            mode=req.mode,
            room_id=consultation.room_id,
            initiator_name=initiator_name
        )
    else:
        # When patient requests consultation -> notify the doctor
        notify_doctor_consultation_requested(
            db=db,
            doctor_id=req.doctor_id,
            patient=patient,
            mode=req.mode,
            room_id=consultation.room_id
        )

    db.commit()
    return _consultation_to_dict(consultation, db)


# ---------------------------------------------------------------------------
# GET /api/consultations – list consultations for current user
# ---------------------------------------------------------------------------

@router.get("/api/consultations")
def list_consultations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_and_expire_consultations(db)
    all_consultations = db.scalars(
        select(Consultation).order_by(Consultation.created_at.desc())
    ).all()
    return [
        _consultation_to_dict(c, db)
        for c in all_consultations
        if _visible_to_user(c, current_user, db)
    ]


# ---------------------------------------------------------------------------
# GET /api/consultations/active – get active consultation for current user
# ---------------------------------------------------------------------------

@router.get("/api/consultations/active")
def get_active_consultation(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_and_expire_consultations(db)
    if current_user.role == Role.DOCTOR.value:
        c = db.scalar(
            select(Consultation).where(
                Consultation.doctor_id == current_user.id,
                Consultation.status == ConsultationStatus.ACTIVE.value,
            )
        )
    elif current_user.role == Role.PATIENT.value:
        patient = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if not patient:
            return None
        c = db.scalar(
            select(Consultation).where(
                Consultation.patient_id == patient.id,
                Consultation.status == ConsultationStatus.ACTIVE.value,
            )
        )
    else:
        # Admin / Nurse: Act purely as coordinators/bridges;
        # they do not participate directly in media calls.
        return None

    if not c:
        return None
    return _consultation_to_dict(c, db)


# ---------------------------------------------------------------------------
# PUT /api/consultations/{id}/accept – recipient accepts consultation request
# ---------------------------------------------------------------------------

@router.put("/api/consultations/{consultation_id}/accept")
def accept_consultation(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    if c.status == ConsultationStatus.ACCEPTED.value:
        return _consultation_to_dict(c, db)

    if c.status != ConsultationStatus.WAITING.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot accept consultation with status '{c.status}'.",
        )

    c.status = ConsultationStatus.ACCEPTED.value
    now = dt.datetime.utcnow()

    # Notify the initiator that their consultation request has been accepted
    initiator = db.get(User, c.initiated_by)
    if initiator:
        doc = db.get(User, c.doctor_id)
        pat = db.get(Patient, c.patient_id)
        pat_user = db.get(User, pat.user_id) if pat and pat.user_id else None
        acceptor_name = current_user.name
        create_in_app_notification(
            db=db,
            user_id=initiator.id,
            title="Teleconsultation Request Accepted",
            message=f"{acceptor_name} accepted your teleconsultation request! You can now join the teleconsultation room.",
            category="consultation",
            link=f"#/consultation/{c.room_id}"
        )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="consultation_accepted",
        target_type="consultation",
        target_id=c.id,
        details={"accepted_by": current_user.id, "accepted_by_name": current_user.name}
    )
    db.commit()
    return _consultation_to_dict(c, db)


# ---------------------------------------------------------------------------
# PUT /api/consultations/{id}/reject – recipient declines or initiator cancels
# ---------------------------------------------------------------------------

@router.put("/api/consultations/{consultation_id}/reject")
def reject_consultation(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    if c.status in (ConsultationStatus.REJECTED.value, ConsultationStatus.CANCELLED.value):
        return _consultation_to_dict(c, db)

    if c.status not in (ConsultationStatus.WAITING.value, ConsultationStatus.ACCEPTED.value):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decline consultation with status '{c.status}'.",
        )

    now = dt.datetime.utcnow()
    c.ended_at = now

    if current_user.id == c.initiated_by:
        c.status = ConsultationStatus.CANCELLED.value
    else:
        c.status = ConsultationStatus.REJECTED.value

    # Revert doctor status
    doctor = db.get(User, c.doctor_id)
    if doctor and doctor.availability_status == "busy":
        doctor.availability_status = "online"

    # Notify other party
    other_user_id = c.initiated_by if current_user.id != c.initiated_by else c.doctor_id
    if other_user_id:
        create_in_app_notification(
            db=db,
            user_id=other_user_id,
            title="Teleconsultation Request Declined/Cancelled",
            message=f"The teleconsultation request was {c.status} by {current_user.name}.",
            category="consultation",
            link="#/patient/appointments" if current_user.role != Role.PATIENT.value else "#/staff/consultations"
        )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action=f"consultation_{c.status}",
        target_type="consultation",
        target_id=c.id,
        details={"declined_by": current_user.id, "status": c.status}
    )
    db.commit()
    return _consultation_to_dict(c, db)


# ---------------------------------------------------------------------------
# POST /api/consultations/{id}/join – register active presence in room
# ---------------------------------------------------------------------------

@router.post("/api/consultations/{consultation_id}/join")
def join_consultation(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    if c.status == ConsultationStatus.WAITING.value:
        raise HTTPException(
            status_code=400,
            detail="This teleconsultation request is pending acceptance. Please accept the request first before joining the room."
        )

    if c.status in (ConsultationStatus.COMPLETED.value, ConsultationStatus.CANCELLED.value, ConsultationStatus.REJECTED.value):
        raise HTTPException(
            status_code=400,
            detail=f"This teleconsultation has ended or was {c.status}."
        )

    now = dt.datetime.utcnow()

    # Check for existing active participation record
    part = db.scalar(
        select(ConsultationParticipant).where(
            ConsultationParticipant.consultation_id == c.id,
            ConsultationParticipant.user_id == current_user.id,
            ConsultationParticipant.left_at.is_(None)
        )
    )
    if not part:
        part = ConsultationParticipant(
            consultation_id=c.id,
            user_id=current_user.id,
            role=current_user.role,
            joined_at=now,
        )
        db.add(part)
        db.flush()

    # Check whether both participants are now joined in the room
    all_active = db.scalars(
        select(ConsultationParticipant).where(
            ConsultationParticipant.consultation_id == c.id,
            ConsultationParticipant.left_at.is_(None)
        )
    ).all()
    roles = {p.role for p in all_active}
    both_joined = ("doctor" in roles and "patient" in roles) or len(all_active) >= 2

    # ONLY start the 15-minute consultation countdown when both persons have joined
    if both_joined and c.status == ConsultationStatus.ACCEPTED.value:
        c.status = ConsultationStatus.ACTIVE.value
        c.started_at = now
        c.expires_at = now + dt.timedelta(minutes=DEFAULT_EXPIRY_MINUTES)

        doc = db.get(User, c.doctor_id)
        if doc and doc.availability_status == "online":
            doc.availability_status = "busy"

        record_audit(
            db=db,
            actor_id=current_user.id,
            action="consultation_activated_both_joined",
            target_type="consultation",
            target_id=c.id,
            details={"started_at": _format_utc(now), "expires_at": _format_utc(c.expires_at)}
        )

    db.commit()
    return _consultation_to_dict(c, db)


# ---------------------------------------------------------------------------
# POST /api/consultations/{id}/leave – mark departure from room
# ---------------------------------------------------------------------------

@router.post("/api/consultations/{consultation_id}/leave")
def leave_consultation(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    now = dt.datetime.utcnow()
    part = db.scalar(
        select(ConsultationParticipant).where(
            ConsultationParticipant.consultation_id == c.id,
            ConsultationParticipant.user_id == current_user.id,
            ConsultationParticipant.left_at.is_(None)
        )
    )
    if part:
        part.left_at = now
        db.commit()

    return {"message": "Left consultation."}


# ---------------------------------------------------------------------------
# PUT /api/consultations/{id}/start – doctor/nurse/admin starts consultation
# ---------------------------------------------------------------------------

@router.put("/api/consultations/{consultation_id}/start")
def start_consultation(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(
        Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value
    )),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    if current_user.role == Role.DOCTOR.value and c.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied: not the assigned doctor.")

    # Idempotent: if already active, return consultation without error
    if c.status == ConsultationStatus.ACTIVE.value:
        return _consultation_to_dict(c, db)

    if c.status not in (ConsultationStatus.WAITING.value, ConsultationStatus.ACCEPTED.value):
        raise HTTPException(
            status_code=400,
            detail=f"Consultation cannot be started from status '{c.status}'.",
        )

    now = dt.datetime.utcnow()
    c.status = ConsultationStatus.ACTIVE.value
    c.started_at = now
    c.expires_at = now + dt.timedelta(minutes=DEFAULT_EXPIRY_MINUTES)

    # Set doctor to busy
    doctor = db.get(User, c.doctor_id)
    if doctor and doctor.availability_status == "online":
        doctor.availability_status = "busy"

    # When doctor/staff starts/joins the consultation, immediately alert the patient
    patient = db.get(Patient, c.patient_id)
    if patient and doctor:
        notify_patient_consultation_opened(
            db=db,
            patient=patient,
            doctor=doctor,
            room_id=c.room_id,
            mode=c.mode
        )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="consultation_started",
        target_type="consultation",
        target_id=c.id,
        details={"started_at": _format_utc(now), "expires_at": _format_utc(c.expires_at)},
    )
    db.commit()
    return _consultation_to_dict(c, db)


# ---------------------------------------------------------------------------
# PUT /api/consultations/{id}/extend – doctor/nurse/admin extends by minutes
# ---------------------------------------------------------------------------

@router.put("/api/consultations/{consultation_id}/extend")
def extend_consultation(
    consultation_id: str,
    req: ExtendRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(
        Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value
    )),
):
    if req.minutes not in (5, 10, 15):
        raise HTTPException(
            status_code=422,
            detail="Extension must be 5, 10, or 15 minutes.",
        )

    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if current_user.role == Role.DOCTOR.value and c.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied: not the assigned doctor.")
    if c.status != ConsultationStatus.ACTIVE.value:
        raise HTTPException(
            status_code=400,
            detail="Only active consultations can be extended.",
        )

    base = c.expires_at if c.expires_at else dt.datetime.utcnow()
    new_expires = base + dt.timedelta(minutes=req.minutes)
    c.expires_at = new_expires

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="consultation_extended",
        target_type="consultation",
        target_id=c.id,
        details={"extended_by_minutes": req.minutes, "new_expires_at": _format_utc(new_expires)},
    )
    db.commit()
    return {
        "message": f"Consultation extended by {req.minutes} minutes.",
        "expires_at": _format_utc(new_expires),
        **_consultation_to_dict(c, db),
    }


# ---------------------------------------------------------------------------
# PUT /api/consultations/{id}/end – end consultation
# ---------------------------------------------------------------------------

@router.put("/api/consultations/{consultation_id}/end")
def end_consultation(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")

    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied: not a participant of this consultation.")

    now = dt.datetime.utcnow()

    # If already ended, return gracefully (idempotent for instant hangup from either side)
    if c.status in (ConsultationStatus.COMPLETED.value, ConsultationStatus.CANCELLED.value):
        return _consultation_to_dict(c, db)

    c.status = ConsultationStatus.COMPLETED.value
    c.ended_at = now

    # Revert doctor availability to online
    doctor = db.get(User, c.doctor_id)
    if doctor and doctor.availability_status == "busy":
        doctor.availability_status = "online"

    # Immediately broadcast call-ended signal so remote peer terminates call instantaneously
    hangup_sig = ConsultationSignal(
        consultation_id=c.id,
        sender_id=current_user.id,
        sender_role=current_user.role,
        signal_type="call-ended",
        signal_data=json.dumps({
            "type": "call-ended",
            "ended_by_id": current_user.id,
            "ended_by_name": current_user.name,
            "ended_by_role": current_user.role,
            "reason": "hangup",
            "ended_at": _format_utc(now)
        }),
        created_at=now,
    )
    db.add(hangup_sig)

    # Add a system ledger entry into consultation messages
    end_msg = ConsultationMessage(
        consultation_id=c.id,
        sender_id=current_user.id,
        sender_name="System",
        sender_role="system",
        message=f"📞 Call ended by {current_user.name} ({current_user.role.title()}). Consultation session concluded.",
        created_at=now,
    )
    db.add(end_msg)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="consultation_ended",
        target_type="consultation",
        target_id=c.id,
        details={"ended_at": _format_utc(now), "ended_by_role": current_user.role, "ended_by_name": current_user.name},
    )
    db.commit()
    return _consultation_to_dict(c, db)


# ---------------------------------------------------------------------------
# POST /api/consultations/{id}/messages – send chat message
# ---------------------------------------------------------------------------

@router.post("/api/consultations/{consultation_id}/messages", status_code=status.HTTP_201_CREATED)
def send_message(
    consultation_id: str,
    req: MessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")
    if c.status not in (ConsultationStatus.ACTIVE.value, ConsultationStatus.ACCEPTED.value, ConsultationStatus.WAITING.value):
        raise HTTPException(
            status_code=400,
            detail="Cannot send messages to a completed or cancelled consultation.",
        )
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    msg = ConsultationMessage(
        consultation_id=consultation_id,
        sender_id=current_user.id,
        sender_name=current_user.name,
        sender_role=current_user.role,
        message=req.message.strip(),
        created_at=dt.datetime.utcnow(),
    )
    db.add(msg)
    db.commit()
    return {
        "id": msg.id,
        "consultation_id": msg.consultation_id,
        "sender_id": msg.sender_id,
        "sender_name": msg.sender_name,
        "sender_role": msg.sender_role,
        "message": msg.message,
        "created_at": _format_utc(msg.created_at),
    }


# ---------------------------------------------------------------------------
# GET /api/consultations/{id}/messages – get chat messages
# ---------------------------------------------------------------------------

@router.get("/api/consultations/{consultation_id}/messages")
def get_messages(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    messages = db.scalars(
        select(ConsultationMessage)
        .where(ConsultationMessage.consultation_id == consultation_id)
        .order_by(ConsultationMessage.created_at.asc())
    ).all()

    return [
        {
            "id": m.id,
            "consultation_id": m.consultation_id,
            "sender_id": m.sender_id,
            "sender_name": m.sender_name,
            "sender_role": m.sender_role,
            "message": m.message,
            "created_at": _format_utc(m.created_at),
        }
        for m in messages
    ]


# ---------------------------------------------------------------------------
# GET /api/consultations/room/{room_id} – get consultation details by room ID
# ---------------------------------------------------------------------------

@router.get("/api/consultations/room/{room_id}")
def get_consultation_by_room(
    room_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_and_expire_consultations(db)
    c = db.scalar(select(Consultation).where(Consultation.room_id == room_id))
    if not c:
        raise HTTPException(status_code=404, detail="Consultation room not found.")
    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied to this consultation room.")
    return _consultation_to_dict(c, db)


# ---------------------------------------------------------------------------
# GET /api/consultations/{consultation_id} – get single consultation by ID
# ---------------------------------------------------------------------------

@router.get("/api/consultations/{consultation_id}")
def get_consultation_by_id(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    _check_and_expire_consultations(db)
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")
    return _consultation_to_dict(c, db)


# ---------------------------------------------------------------------------
# POST /api/consultations/{id}/signals – send WebRTC signaling data
# ---------------------------------------------------------------------------

@router.post("/api/consultations/{consultation_id}/signals", status_code=status.HTTP_201_CREATED)
def send_signal(
    consultation_id: str,
    req: SignalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    sig = ConsultationSignal(
        consultation_id=consultation_id,
        sender_id=current_user.id,
        sender_role=current_user.role,
        signal_type=req.signal_type,
        signal_data=req.signal_data,
        created_at=dt.datetime.utcnow(),
    )
    db.add(sig)
    db.commit()
    return {
        "id": sig.id,
        "consultation_id": sig.consultation_id,
        "sender_id": sig.sender_id,
        "sender_role": sig.sender_role,
        "signal_type": sig.signal_type,
        "signal_data": sig.signal_data,
        "created_at": _format_utc(sig.created_at),
    }


# ---------------------------------------------------------------------------
# GET /api/consultations/{id}/signals – get WebRTC signaling data from peer
# ---------------------------------------------------------------------------

@router.get("/api/consultations/{consultation_id}/signals")
def get_signals(
    consultation_id: str,
    since_id: Optional[str] = None,
    since: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    query = select(ConsultationSignal).where(
        ConsultationSignal.consultation_id == consultation_id,
        ConsultationSignal.sender_id != current_user.id
    )
    if since_id:
        ref_sig = db.get(ConsultationSignal, since_id)
        if ref_sig:
            query = query.where(ConsultationSignal.created_at > ref_sig.created_at)
    elif since:
        try:
            cleaned = since.rstrip("Z")
            since_dt = dt.datetime.fromisoformat(cleaned)
            query = query.where(ConsultationSignal.created_at > since_dt)
        except Exception:
            pass

    signals = db.scalars(query.order_by(ConsultationSignal.created_at.asc())).all()
    results = []
    for s in signals:
        payload = s.signal_data
        if s.signal_data and (s.signal_data.startswith("{") or s.signal_data.startswith("[")):
            try:
                payload = json.loads(s.signal_data)
            except Exception:
                payload = s.signal_data
        results.append({
            "id": s.id,
            "consultation_id": s.consultation_id,
            "sender_id": s.sender_id,
            "sender_role": s.sender_role,
            "signal_type": s.signal_type,
            "signal_data": s.signal_data,
            "type": s.signal_type,
            "payload": payload,
            "created_at": _format_utc(s.created_at),
        })
    return results


# ---------------------------------------------------------------------------
# DELETE /api/consultations/{id}/signals – clear WebRTC signaling buffer
# ---------------------------------------------------------------------------

@router.delete("/api/consultations/{consultation_id}/signals")
def clear_signals(
    consultation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    c = db.get(Consultation, consultation_id)
    if not c:
        raise HTTPException(status_code=404, detail="Consultation not found.")
    if not _visible_to_user(c, current_user, db):
        raise HTTPException(status_code=403, detail="Access denied.")

    signals = db.scalars(select(ConsultationSignal).where(ConsultationSignal.consultation_id == consultation_id)).all()
    for s in signals:
        db.delete(s)
    db.commit()
    return {"message": "Signaling buffer cleared.", "count": len(signals)}

