import datetime as dt
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.database import get_db
from app.core.config import settings
from app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    validate_password_length,
)
from app.core.audit import record_audit
from app.core.constants import Role
from app.models.user import User, PasswordResetTicket, ProfileUpdateRequest
from app.models.patient import Patient
from app.models.notification import Notification
from app.schemas.auth import (
    TokenResponse,
    LoginRequest,
    StaffRegisterRequest,
    PatientAccountCreateRequest,
    ForgotPasswordRequest,
    ResolveTicketRequest,
    ChangePasswordRequest,
    ProfileUpdateRequestIn,
    ProfileUpdateResolveIn,
    AdminUserUpdateIn,
)
from app.api.deps import get_current_user, require_role

router = APIRouter(prefix="/api/auth", tags=["Authentication & Access"])

@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    validate_password_length(req.password)
    user = db.scalar(select(User).where(User.email == req.email))
    if not user or not user.active or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    token = create_access_token({"sub": user.id, "role": user.role, "email": user.email})
    record_audit(
        db=db,
        actor_id=user.id,
        action="user_login",
        target_type="user",
        target_id=user.id,
        details={"email": user.email, "role": user.role}
    )
    db.commit()

    return TokenResponse(
        token=token,
        user={
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "phone": user.phone,
            "specialization": user.specialization,
            "availability_status": user.availability_status or "online",
            "away_until": user.away_until.isoformat() if user.away_until else None,
            "must_change_password": bool(user.must_change_password),
        }
    )

@router.post("/register-staff", status_code=status.HTTP_201_CREATED)
def register_staff(req: StaffRegisterRequest, db: Session = Depends(get_db)):
    if req.staff_code != settings.STAFF_REGISTRATION_CODE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid hospital staff registration authorization code."
        )

    validate_password_length(req.password)
    existing = db.scalar(select(User).where(User.email == req.email))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this email address already exists."
        )

    user = User(
        email=req.email,
        password_hash=get_password_hash(req.password),
        name=req.name,
        role=req.role,
        active=True,
        phone=req.phone.strip(),
        specialization=req.specialization.strip() if req.specialization else None,
        availability_status="online",
    )
    db.add(user)
    db.flush()

    try:
        from app.api.announcements_api import create_new_staff_announcement
        create_new_staff_announcement(db=db, new_staff_user=user)
    except Exception as e:
        logger.warning(f"Could not auto-generate staff announcement: {e}")

    record_audit(
        db=db,
        actor_id=user.id,
        action="staff_registered",
        target_type="user",
        target_id=user.id,
        new_values={"email": user.email, "role": user.role, "name": user.name, "phone": req.phone.strip()}
    )
    db.commit()

    return {
        "message": f"Hospital staff account ({req.role}) created successfully.",
        "user_id": user.id,
        "email": user.email,
        "role": user.role,
        "phone": user.phone
    }

@router.post("/create-patient-account", status_code=status.HTTP_201_CREATED)
def create_patient_account(
    req: PatientAccountCreateRequest,
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value)),
    db: Session = Depends(get_db)
):
    patient = db.get(Patient, req.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    validate_password_length(req.password)
    existing = db.scalar(select(User).where(User.email == req.email))
    if existing:
        raise HTTPException(status_code=409, detail="A user account with this email already exists.")

    pat_user = db.get(User, patient.user_id) if patient.user_id else None
    patient_name = pat_user.name if pat_user else patient.name

    user = User(
        email=req.email,
        password_hash=get_password_hash(req.password),
        name=patient_name,
        role=Role.PATIENT.value,
        active=True
    )
    db.add(user)
    db.flush()

    patient.user_id = user.id

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_account_created",
        target_type="patient",
        target_id=patient.id,
        new_values={"user_id": user.id, "email": req.email}
    )
    db.commit()

    return {"message": "Patient user account created successfully.", "user_id": user.id, "patient_id": patient.id}

@router.post("/forgot-password")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == req.email))
    if not user:
        # Do not leak whether the email exists
        return {
            "status": "received",
            "message": "If an account matches this email, a support ticket has been recorded for hospital administrator review."
        }

    ticket = PasswordResetTicket(
        user_id=user.id,
        email=user.email,
        status="open",
        reason=req.reason,
        created_at=dt.datetime.utcnow()
    )
    db.add(ticket)
    db.flush()

    # Create in-app alerts for all hospital administrators
    admins = db.scalars(select(User).where(User.role == Role.ADMIN.value, User.active == True)).all()
    for admin in admins:
        admin_notif = Notification(
            user_id=admin.id,
            title="Password Reset Request",
            message=f"Password reset request from {user.name} ({user.email}, Role: {user.role.title()}). Reason: {req.reason}",
            category="password_reset",
            link="/staff/requests?tab=reset_tickets"
        )
        db.add(admin_notif)

    record_audit(
        db=db,
        actor_id=user.id,
        action="password_reset_ticket_created",
        target_type="password_reset_ticket",
        target_id=ticket.id,
        details={"email": user.email, "reason": req.reason}
    )
    db.commit()

    return {
        "status": "ticket_created",
        "ticket_id": ticket.id,
        "message": "Support ticket created for hospital administrator review."
    }

@router.get("/reset-tickets")
def list_reset_tickets(
    status_filter: Optional[str] = None,
    current_user: User = Depends(require_role(Role.ADMIN.value)),
    db: Session = Depends(get_db)
):
    query = select(PasswordResetTicket).order_by(PasswordResetTicket.created_at.desc())
    if status_filter:
        query = query.where(PasswordResetTicket.status == status_filter)
    tickets = db.scalars(query).all()

    result = []
    for t in tickets:
        target_user = db.get(User, t.user_id)
        target_phone = target_user.phone if target_user else None
        if not target_phone and target_user and target_user.role == Role.PATIENT.value:
            p = db.scalar(select(Patient).where(Patient.user_id == target_user.id))
            if p:
                target_phone = p.phone

        result.append({
            "id": t.id,
            "user_id": t.user_id,
            "user_email": target_user.email if target_user else (t.email or "unknown"),
            "user_name": target_user.name if target_user else "unknown",
            "user_role": target_user.role if target_user else "unknown",
            "user_phone": target_phone,
            "status": t.status,
            "reason": t.reason,
            "handled_by": t.handled_by,
            "resolution_notes": t.resolution_notes,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None
        })
    return result

@router.post("/reset-tickets/{ticket_id}/resolve")
def resolve_reset_ticket(
    ticket_id: str,
    req: ResolveTicketRequest,
    current_user: User = Depends(require_role(Role.ADMIN.value)),
    db: Session = Depends(get_db)
):
    ticket = db.get(PasswordResetTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Password reset ticket not found.")
    if ticket.status != "open":
        raise HTTPException(status_code=400, detail="Ticket has already been resolved.")

    target_user = db.get(User, ticket.user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="Associated user not found.")

    sms_sent = False
    if req.action == "resolved":
        if not req.new_password:
            raise HTTPException(status_code=422, detail="A new password is required to resolve the ticket.")
        validate_password_length(req.new_password)
        target_user.password_hash = get_password_hash(req.new_password)
        target_user.must_change_password = req.must_change_password
        ticket.status = "resolved"

        # In-app notification to target user
        user_notif = Notification(
            user_id=target_user.id,
            title="Password Reset Completed",
            message="Your account password has been reset by hospital administrator. Please log in using your updated temporary password.",
            category="account_security"
        )
        db.add(user_notif)

        # Dispatch SMS to registered phone number if requested
        if req.notify_sms:
            target_phone = target_user.phone
            pat = None
            if not target_phone and target_user.role == Role.PATIENT.value:
                pat = db.scalar(select(Patient).where(Patient.user_id == target_user.id))
                if pat:
                    target_phone = pat.phone

            if target_phone:
                from app.services.sms_gateway_service import queue_and_send_sms
                sms_body = f"Recovera Hospital Alert: Your login password has been reset by the administrator. Updated Password: {req.new_password}. Please log in and change your password."
                try:
                    queue_and_send_sms(
                        db=db,
                        phone=target_phone,
                        message_text=sms_body,
                        category="account_alert",
                        patient_id=pat.id if pat else None
                    )
                    sms_sent = True
                except Exception:
                    pass
    else:
        ticket.status = "rejected"
        user_notif = Notification(
            user_id=target_user.id,
            title="Password Reset Declined",
            message=f"Your password reset request was declined: {req.notes or 'No reason provided'}",
            category="account_security"
        )
        db.add(user_notif)

    ticket.handled_by = current_user.id
    ticket.resolution_notes = req.notes or f"Ticket marked as {req.action} by Admin."
    ticket.resolved_at = dt.datetime.utcnow()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action=f"password_reset_ticket_{req.action}",
        target_type="password_reset_ticket",
        target_id=ticket.id,
        details={"user_id": target_user.id, "action": req.action, "notes": ticket.resolution_notes, "sms_sent": sms_sent}
    )
    db.commit()

    return {"message": f"Ticket {ticket_id} successfully {req.action}.", "status": ticket.status, "sms_sent": sms_sent}

@router.get("/me")
def get_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    patient_info = None
    if current_user.role == Role.PATIENT.value:
        p = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if p:
            patient_info = {
                "patient_id": p.id,
                "patient_code": p.patient_code,
                "dob": p.dob.isoformat() if p.dob else None,
                "gender": p.gender,
                "preferred_language": p.preferred_language,
                "care_status": getattr(p, "care_status", "active"),
                "phone": p.phone or current_user.phone
            }

    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "role": current_user.role,
        "phone": current_user.phone,
        "specialization": current_user.specialization,
        "availability_status": current_user.availability_status or "online",
        "away_until": current_user.away_until.isoformat() if current_user.away_until else None,
        "must_change_password": bool(current_user.must_change_password),
        "patient_profile": patient_info,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None
    }

@router.post("/refresh")
def refresh_session_token(current_user: User = Depends(get_current_user)):
    """Issues a fresh access token for an active authenticated user to prevent session expiration."""
    access_token = create_access_token(
        data={"sub": current_user.email, "role": current_user.role, "user_id": current_user.id}
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "name": current_user.name,
            "role": current_user.role,
            "must_change_password": bool(current_user.must_change_password)
        }
    }

@router.post("/change-password")
def change_password(
    req: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.must_change_password:
        if not req.old_password:
            raise HTTPException(status_code=400, detail="Current password is required.")
        if not verify_password(req.old_password, current_user.password_hash):
            raise HTTPException(status_code=400, detail="Current password does not match our records.")

    if req.confirm_password and req.confirm_password != req.new_password:
        raise HTTPException(status_code=400, detail="New password and confirm password do not match.")

    validate_password_length(req.new_password)
    current_user.password_hash = get_password_hash(req.new_password)
    current_user.must_change_password = False

    notif = Notification(
        user_id=current_user.id,
        title="Password Updated Successfully",
        message="Your account password has been updated. Use your new password for future logins.",
        category="account_security"
    )
    db.add(notif)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="password_changed",
        target_type="user",
        target_id=current_user.id,
        details={"email": current_user.email}
    )
    db.commit()

    return {"message": "Password changed successfully.", "must_change_password": False}

@router.post("/profile-update-request", status_code=status.HTTP_201_CREATED)
def create_profile_update_request(
    req: ProfileUpdateRequestIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    curr_val = ""
    if req.field_name == "name":
        curr_val = current_user.name
    elif req.field_name == "phone":
        curr_val = current_user.phone or ""
    elif req.field_name == "specialization":
        curr_val = current_user.specialization or ""
    elif current_user.role == Role.PATIENT.value:
        p = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if p:
            if req.field_name == "preferred_language":
                curr_val = p.preferred_language or "en"
            elif req.field_name == "sms_language":
                curr_val = getattr(p, "sms_language", None) or p.preferred_language or "en"
            elif req.field_name == "dob":
                curr_val = p.dob.isoformat() if p.dob else ""
            elif req.field_name == "gender":
                curr_val = p.gender or ""
            elif req.field_name == "phone":
                curr_val = p.phone or current_user.phone or ""

    # If requester is an Administrator, directly apply changes without pending status
    if current_user.role == Role.ADMIN.value:
        fn = req.field_name.lower().strip()
        val = req.requested_value.strip()
        if fn == "name":
            current_user.name = val
        elif fn == "phone":
            current_user.phone = val
        elif fn == "specialization":
            current_user.specialization = val
        elif fn == "email":
            val_email = val.lower()
            existing = db.scalar(select(User).where(User.email == val_email, User.id != current_user.id))
            if existing:
                raise HTTPException(status_code=400, detail=f"Email '{val_email}' is already taken.")
            current_user.email = val_email

        ticket = ProfileUpdateRequest(
            user_id=current_user.id,
            field_name=req.field_name,
            current_value=curr_val,
            requested_value=req.requested_value,
            reason=req.reason or "Direct administrator update",
            status="approved",
            admin_notes="Directly updated and authorized by administrator.",
            reviewed_by=current_user.id,
            created_at=dt.datetime.utcnow()
        )
        db.add(ticket)
        db.flush()

        record_audit(
            db=db,
            actor_id=current_user.id,
            action="admin_direct_profile_updated",
            target_type="user",
            target_id=current_user.id,
            details={"field": req.field_name, "value": req.requested_value, "reason": req.reason}
        )
        db.commit()

        return {
            "id": ticket.id,
            "field_name": ticket.field_name,
            "current_value": ticket.current_value,
            "requested_value": ticket.requested_value,
            "reason": ticket.reason,
            "status": "approved",
            "message": f"Profile field '{ticket.field_name}' updated and applied immediately.",
            "created_at": ticket.created_at.isoformat()
        }

    ticket = ProfileUpdateRequest(
        user_id=current_user.id,
        field_name=req.field_name,
        current_value=curr_val,
        requested_value=req.requested_value,
        reason=req.reason,
        status="pending",
        created_at=dt.datetime.utcnow()
    )
    db.add(ticket)
    db.flush()

    admins = db.scalars(select(User).where(User.role == Role.ADMIN.value, User.active == True)).all()
    for admin in admins:
        admin_notif = Notification(
            user_id=admin.id,
            title="Profile Modification Request",
            message=f"{current_user.name} ({current_user.role.title()}) requested to update '{req.field_name}' to '{req.requested_value}'. Reason: {req.reason}",
            category="profile_update",
            link="/staff/requests?tab=profile_requests"
        )
        db.add(admin_notif)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="profile_update_requested",
        target_type="profile_update_request",
        target_id=ticket.id,
        details={"field": req.field_name, "requested_value": req.requested_value, "reason": req.reason}
    )
    db.commit()

    return {
        "id": ticket.id,
        "field_name": ticket.field_name,
        "current_value": ticket.current_value,
        "requested_value": ticket.requested_value,
        "reason": ticket.reason,
        "status": ticket.status,
        "created_at": ticket.created_at.isoformat()
    }

@router.get("/profile-update-requests")
def list_profile_update_requests(
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = select(ProfileUpdateRequest).order_by(ProfileUpdateRequest.created_at.desc())
    if current_user.role != Role.ADMIN.value:
        query = query.where(ProfileUpdateRequest.user_id == current_user.id)
    elif status_filter:
        query = query.where(ProfileUpdateRequest.status == status_filter)

    requests = db.scalars(query).all()
    res = []
    for r in requests:
        u = db.get(User, r.user_id)
        res.append({
            "id": r.id,
            "user_id": r.user_id,
            "user_name": u.name if u else "Unknown",
            "user_email": u.email if u else "Unknown",
            "user_role": u.role if u else "Unknown",
            "field_name": r.field_name,
            "current_value": r.current_value,
            "requested_value": r.requested_value,
            "reason": r.reason,
            "status": r.status,
            "admin_notes": r.admin_notes,
            "reviewed_by": r.reviewed_by,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None
        })
    return res

@router.post("/profile-update-requests/{request_id}/approve")
def approve_profile_update_request(
    request_id: str,
    current_user: User = Depends(require_role(Role.ADMIN.value)),
    db: Session = Depends(get_db)
):
    req_record = db.get(ProfileUpdateRequest, request_id)
    if not req_record:
        raise HTTPException(status_code=404, detail="Profile update request not found.")
    if req_record.status != "pending":
        raise HTTPException(status_code=400, detail=f"Request is already {req_record.status}.")

    target_user = db.get(User, req_record.user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="Associated user not found.")

    fn = req_record.field_name.lower().strip()
    val = req_record.requested_value.strip()

    if fn == "name":
        target_user.name = val
    elif fn == "phone":
        target_user.phone = val
        p = db.scalar(select(Patient).where(Patient.user_id == target_user.id))
        if p:
            p.phone = val
    elif fn == "specialization":
        target_user.specialization = val
    elif fn == "preferred_language":
        p = db.scalar(select(Patient).where(Patient.user_id == target_user.id))
        if p:
            p.preferred_language = val
    elif fn == "sms_language":
        p = db.scalar(select(Patient).where(Patient.user_id == target_user.id))
        if p:
            p.sms_language = val
    elif fn == "gender":
        p = db.scalar(select(Patient).where(Patient.user_id == target_user.id))
        if p:
            p.gender = val
    elif fn == "dob":
        p = db.scalar(select(Patient).where(Patient.user_id == target_user.id))
        if p:
            try:
                p.dob = dt.date.fromisoformat(val)
            except Exception:
                pass

    req_record.status = "approved"
    req_record.reviewed_by = current_user.id
    req_record.reviewed_at = dt.datetime.utcnow()

    notif = Notification(
        user_id=target_user.id,
        title="Profile Update Approved",
        message=f"Administrator approved your profile update request for '{req_record.field_name}'. Your profile has been updated to: {req_record.requested_value}.",
        category="account_security"
    )
    db.add(notif)

    target_phone = target_user.phone
    if not target_phone and target_user.role == Role.PATIENT.value:
        p = db.scalar(select(Patient).where(Patient.user_id == target_user.id))
        if p:
            target_phone = p.phone
    if target_phone:
        from app.services.sms_gateway_service import queue_and_send_sms
        try:
            queue_and_send_sms(
                db=db,
                phone=target_phone,
                message_text=f"Recovera Alert: Your profile update for '{req_record.field_name}' was approved and updated by the Administrator.",
                category="account_alert"
            )
        except Exception:
            pass

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="profile_update_approved",
        target_type="profile_update_request",
        target_id=req_record.id,
        details={"field": req_record.field_name, "value": req_record.requested_value, "user_id": target_user.id}
    )
    db.commit()

    return {"message": "Profile update request approved and changes applied successfully.", "status": "approved"}

@router.post("/profile-update-requests/{request_id}/reject")
def reject_profile_update_request(
    request_id: str,
    req: ProfileUpdateResolveIn,
    current_user: User = Depends(require_role(Role.ADMIN.value)),
    db: Session = Depends(get_db)
):
    req_record = db.get(ProfileUpdateRequest, request_id)
    if not req_record:
        raise HTTPException(status_code=404, detail="Profile update request not found.")
    if req_record.status != "pending":
        raise HTTPException(status_code=400, detail=f"Request is already {req_record.status}.")

    target_user = db.get(User, req_record.user_id)
    if not target_user:
        raise HTTPException(status_code=404, detail="Associated user not found.")

    req_record.status = "rejected"
    req_record.admin_notes = req.admin_notes or "Request declined by administrator."
    req_record.reviewed_by = current_user.id
    req_record.reviewed_at = dt.datetime.utcnow()

    notif = Notification(
        user_id=target_user.id,
        title="Profile Update Declined",
        message=f"Your profile update request for '{req_record.field_name}' was declined: {req_record.admin_notes}",
        category="account_security"
    )
    db.add(notif)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="profile_update_rejected",
        target_type="profile_update_request",
        target_id=req_record.id,
        details={"field": req_record.field_name, "notes": req_record.admin_notes, "user_id": target_user.id}
    )
    db.commit()

    return {"message": "Profile update request declined.", "status": "rejected"}


@router.get("/users")
def list_users_for_admin(
    role: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Admin endpoint to list all system users (doctors, nurses, patients, admins)."""
    query = select(User).order_by(User.created_at.desc())
    if role:
        query = query.where(User.role == role)
    users = db.scalars(query).all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "name": u.name,
            "role": u.role,
            "phone": u.phone,
            "specialization": u.specialization,
            "availability_status": u.availability_status or "online",
            "away_until": u.away_until.isoformat() if u.away_until else None,
            "active": u.active,
            "created_at": u.created_at.isoformat() if u.created_at else None
        }
        for u in users
    ]


@router.delete("/users/{user_id}")
def delete_user_for_admin(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Admin endpoint to delete a doctor, nurse, or user account permanently."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own administrator account.")

    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    if user.role == Role.ADMIN.value:
        raise HTTPException(
            status_code=403,
            detail="Security restriction: An administrator cannot delete another administrator account."
        )

    user_name = user.name
    user_email = user.email
    user_role = user.role

    # If patient: cascade delete entire patient entity and associated user
    if user.role == Role.PATIENT.value:
        from app.models.patient import Patient
        from app.api.patients import _perform_patient_deletion
        p = db.scalar(select(Patient).where(Patient.user_id == user_id))
        if p:
            _perform_patient_deletion(patient=p, db=db, actor_id=current_user.id)
            return {"message": f"Patient user {user_name} ({user_email}) and profile deleted successfully."}

    from sqlalchemy import delete, update, text, select
    from app.models.notification import Notification
    from app.models.chat_session import ChatSession, ChatMessage
    from app.models.audit_log import AuditLog
    from app.models.appointment import Appointment, DoctorAvailability
    from app.models.consultation import Consultation, ConsultationMessage, ConsultationParticipant, ConsultationSignal
    from app.models.patient import PatientDoctorAssignment, Patient, PatientDeletionTicket
    from app.models.clinical_document import ClinicalDocument
    from app.models.medication import Medication, MedicationAdherence
    from app.models.risk_history import RiskClassificationHistory

    # 1. Doctor assignments & primary doctor
    db.execute(delete(PatientDoctorAssignment).where(
        (PatientDoctorAssignment.doctor_id == user_id) |
        (PatientDoctorAssignment.assigned_by == user_id)
    ))
    db.execute(update(PatientDoctorAssignment).where(PatientDoctorAssignment.revoked_by == user_id).values(revoked_by=None))
    try:
        db.execute(text("DELETE FROM patient_staff_assignments WHERE staff_user_id = :uid"), {"uid": user_id})
    except Exception:
        pass

    patients_with_doc = db.scalars(select(Patient).where(Patient.assigned_doctor_id == user_id)).all()
    for p in patients_with_doc:
        p.assigned_doctor_id = None

    db.execute(update(Patient).where(Patient.created_by == user_id).values(created_by=current_user.id))
    db.execute(update(Patient).where(Patient.sms_opt_in_recorded_by == user_id).values(sms_opt_in_recorded_by=None))

    # 2. Appointments & Doctor schedules
    db.execute(delete(DoctorAvailability).where(DoctorAvailability.doctor_id == user_id))
    db.execute(delete(Appointment).where(Appointment.doctor_id == user_id))
    db.execute(update(Appointment).where(Appointment.booked_by == user_id).values(booked_by=current_user.id))

    # 3. Consultations
    db.execute(delete(ConsultationSignal).where(ConsultationSignal.sender_id == user_id))
    db.execute(delete(ConsultationMessage).where(ConsultationMessage.sender_id == user_id))
    db.execute(delete(ConsultationParticipant).where(ConsultationParticipant.user_id == user_id))
    c_sub = select(Consultation.id).where((Consultation.doctor_id == user_id) | (Consultation.initiated_by == user_id))
    db.execute(delete(ConsultationSignal).where(ConsultationSignal.consultation_id.in_(c_sub)))
    db.execute(delete(ConsultationMessage).where(ConsultationMessage.consultation_id.in_(c_sub)))
    db.execute(delete(ConsultationParticipant).where(ConsultationParticipant.consultation_id.in_(c_sub)))
    db.execute(delete(Consultation).where((Consultation.doctor_id == user_id) | (Consultation.initiated_by == user_id)))

    # 4. Clinical documents & Medications
    db.execute(update(ClinicalDocument).where(ClinicalDocument.uploaded_by == user_id).values(uploaded_by=current_user.id))
    db.execute(update(ClinicalDocument).where(ClinicalDocument.reviewed_by == user_id).values(reviewed_by=None))
    db.execute(update(ClinicalDocument).where(ClinicalDocument.assigned_doctor_id == user_id).values(assigned_doctor_id=None))
    try:
        db.execute(text("UPDATE documents SET uploaded_by = :aid WHERE uploaded_by = :uid"), {"aid": current_user.id, "uid": user_id})
        db.execute(text("UPDATE documents SET approved_by = NULL WHERE approved_by = :uid"), {"uid": user_id})
    except Exception:
        pass
    db.execute(update(Medication).where(Medication.created_by == user_id).values(created_by=current_user.id))
    db.execute(update(MedicationAdherence).where(MedicationAdherence.recorded_by == user_id).values(recorded_by=current_user.id))

    # 5. Tickets & Risk history
    db.execute(update(PatientDeletionTicket).where(PatientDeletionTicket.requested_by == user_id).values(requested_by=current_user.id))
    db.execute(update(PatientDeletionTicket).where(PatientDeletionTicket.reviewed_by == user_id).values(reviewed_by=None))
    db.execute(update(RiskClassificationHistory).where(RiskClassificationHistory.overridden_by == user_id).values(overridden_by=None))

    # 6. Chat sessions & messages
    chat_sub = select(ChatSession.id).where(ChatSession.user_id == user_id)
    db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(chat_sub)))
    db.execute(delete(ChatSession).where(ChatSession.user_id == user_id))

    # 7. Notifications & Reset tickets & Profile update requests & Audit logs
    db.execute(delete(Notification).where(Notification.user_id == user_id))
    db.execute(delete(ProfileUpdateRequest).where(ProfileUpdateRequest.user_id == user_id))
    db.execute(update(ProfileUpdateRequest).where(ProfileUpdateRequest.reviewed_by == user_id).values(reviewed_by=None))
    try:
        db.execute(text("DELETE FROM password_reset_tickets WHERE requester_user_id = :uid OR resolved_by = :uid"), {"uid": user_id})
    except Exception:
        pass
    db.execute(update(AuditLog).where(AuditLog.actor_id == user_id).values(actor_id=None))

    # 8. Delete user
    db.delete(user)
    db.flush()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="user_deleted",
        target_type="user",
        target_id=user_id,
        details={"name": user_name, "email": user_email, "role": user_role}
    )
    db.commit()
    return {"message": f"User {user_name} ({user_email}, {user_role}) deleted permanently."}


@router.patch("/users/{user_id}")
def update_user_for_admin(
    user_id: str,
    req: AdminUserUpdateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Admin endpoint to edit ANY user account details (name, email, phone, role, specialization, status, password)."""
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    # Guard: prevent administrator from demoting or deactivating their OWN account if they are the sole active admin
    if user_id == current_user.id:
        if (req.role and req.role != Role.ADMIN.value) or (req.active is False):
            admin_count = db.scalar(select(func.count(User.id)).where(User.role == Role.ADMIN.value, User.active == True)) or 0
            if admin_count <= 1:
                raise HTTPException(
                    status_code=400,
                    detail="Security restriction: You cannot demote or deactivate your own account as the sole active administrator."
                )

    updated_fields = {}

    if req.name is not None and req.name.strip():
        user.name = req.name.strip()
        updated_fields["name"] = user.name

    if req.email is not None and req.email.strip():
        new_email = req.email.strip().lower()
        if new_email != user.email:
            existing = db.scalar(select(User).where(User.email == new_email, User.id != user.id))
            if existing:
                raise HTTPException(status_code=400, detail=f"Email '{new_email}' is already in use by another user.")
            user.email = new_email
            updated_fields["email"] = user.email

    if req.phone is not None:
        user.phone = req.phone.strip()
        updated_fields["phone"] = user.phone
        # Keep associated patient phone in sync
        p = db.scalar(select(Patient).where(Patient.user_id == user.id))
        if p:
            p.phone = user.phone

    if req.role is not None:
        user.role = req.role
        updated_fields["role"] = user.role

    if req.specialization is not None:
        user.specialization = req.specialization.strip() if req.specialization else None
        updated_fields["specialization"] = user.specialization

    if req.availability_status is not None:
        user.availability_status = req.availability_status
        updated_fields["availability_status"] = user.availability_status

    if req.active is not None:
        user.active = req.active
        updated_fields["active"] = user.active

    if req.password is not None and req.password.strip():
        validate_password_length(req.password.strip())
        user.password_hash = get_password_hash(req.password.strip())
        user.must_change_password = False
        updated_fields["password_reset"] = True

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="admin_updated_user",
        target_type="user",
        target_id=user.id,
        details={"updated_fields": updated_fields, "target_email": user.email, "target_name": user.name}
    )
    db.commit()

    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "role": user.role,
        "phone": user.phone,
        "specialization": user.specialization,
        "availability_status": user.availability_status or "online",
        "active": user.active,
        "message": f"User account '{user.name}' updated successfully."
    }

