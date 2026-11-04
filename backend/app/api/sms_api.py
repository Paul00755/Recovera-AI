import datetime as dt
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_

from app.core.database import get_db
from app.core.constants import Role, SMSCategory, SMSJobStatus
from app.models.sms_job import SMSJob
from app.models.user import User
from app.models.patient import Patient
from app.api.deps import get_current_user, require_role
from app.services.sms_gateway_service import (
    get_gateway_status,
    update_companion_telemetry,
    queue_and_send_sms,
    retry_sms_job
)
from app.services.medication_reminder_service import process_due_medication_reminders

router = APIRouter(prefix="/api/sms", tags=["Hospital SMS Gateway & Automations"])


class CompanionSyncRequest(BaseModel):
    battery_pct: Optional[int] = Field(default=None, ge=0, le=100)
    signal: Optional[str] = None
    carrier: Optional[str] = None
    device_name: Optional[str] = None


class TestSMSRequest(BaseModel):
    phone: str
    message: str


def _to_utc_iso(d: Optional[dt.datetime]) -> Optional[str]:
    if not d:
        return None
    s = d.isoformat()
    if not s.endswith("Z") and "+" not in s and "-" not in s[10:]:
        s += "Z"
    return s


def _job_to_dict(job: SMSJob, db: Session) -> dict:
    patient = db.get(Patient, job.patient_id) if job.patient_id else None
    if not patient and job.recipient_phone:
        patient = db.scalar(select(Patient).where(Patient.phone == job.recipient_phone.strip()))

    patient_name = None
    if patient:
        patient_user = db.get(User, patient.user_id) if patient.user_id else None
        patient_name = (patient_user.name if patient_user else None) or patient.patient_code
    elif job.recipient_phone:
        # Check if recipient is a doctor, nurse, or admin user
        staff_user = db.scalar(select(User).where(User.phone == job.recipient_phone.strip()))
        if staff_user:
            patient_name = f"{staff_user.name} ({staff_user.role.title()})"

    sent_dt = job.last_attempt or (job.created_at if job.state in (SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value) else None)
    return {
        "id": job.id,
        "patient_id": job.patient_id or (patient.id if patient else None),
        "patient_name": patient_name,
        "recipient_phone": job.recipient_phone,
        "category": job.category,
        "message_text": job.message_text,
        "idempotency_key": job.idempotency_key,
        "state": job.state,
        "scheduled_for": _to_utc_iso(job.scheduled_for),
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "last_attempt": _to_utc_iso(job.last_attempt),
        "sent_at": _to_utc_iso(sent_dt),
        "provider_message_id": job.provider_message_id,
        "error_code": job.error_code,
        "error_message": job.error_message,
        "created_at": _to_utc_iso(job.created_at),
    }


@router.get("/gateway/status")
def gateway_status_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve live status, telemetry, quota, and statistics of the Hospital SMS Gateway."""
    return get_gateway_status(db)


@router.post("/gateway/sync")
def sync_companion_telemetry_endpoint(
    payload: CompanionSyncRequest
):
    """Webhook/heartbeat endpoint called by the companion app to update battery, carrier, and signal metrics."""
    updated = update_companion_telemetry(payload.model_dump(exclude_unset=True))
    return {"status": "ok", "telemetry": updated}


@router.post("/gateway/test")
def send_test_sms_endpoint(
    payload: TestSMSRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Dispatch an immediate test SMS to verify gateway connectivity and carrier transmission."""
    if not payload.phone.strip() or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Phone number and message are required.")

    job = queue_and_send_sms(
        db=db,
        phone=payload.phone,
        message_text=payload.message,
        category="test"
    )
    return {
        "status": "dispatched",
        "job": _job_to_dict(job, db)
    }


@router.get("/jobs")
def list_sms_jobs_endpoint(
    patient_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    phone: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.NURSE.value))
):
    """List recent SMS jobs with filtering by category, status, patient, or phone number."""
    stmt = select(SMSJob)

    conditions = []
    if patient_id:
        conditions.append(SMSJob.patient_id == patient_id)
    if category:
        conditions.append(SMSJob.category == category)
    if state:
        conditions.append(SMSJob.state == state)
    if phone:
        conditions.append(SMSJob.recipient_phone.contains(phone))

    if conditions:
        stmt = stmt.where(and_(*conditions))

    stmt = stmt.order_by(SMSJob.created_at.desc()).offset(offset).limit(limit)
    jobs = db.scalars(stmt).all()

    total = db.scalar(
        select(func.count(SMSJob.id)).where(and_(*conditions) if conditions else True)
    ) or 0

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "jobs": [_job_to_dict(j, db) for j in jobs]
    }


@router.post("/jobs/{job_id}/retry")
def retry_sms_job_endpoint(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.NURSE.value))
):
    """Retry sending a failed or queued SMS job."""
    try:
        job = retry_sms_job(db, job_id)
        return {
            "status": "retried",
            "job": _job_to_dict(job, db)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/process-reminders")
def process_reminders_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.NURSE.value))
):
    """Trigger the two-step automated medication reminder evaluation cycle."""
    result = process_due_medication_reminders(db)
    return result


@router.post("/jobs/{job_id}/cancel")
def cancel_sms_job_endpoint(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Admin endpoint to cancel/stop a queued or failed notification job."""
    job = db.get(SMSJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="SMS job not found.")

    if job.state in (SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value):
        raise HTTPException(status_code=400, detail="Cannot cancel a message that has already been delivered.")

    job.state = SMSJobStatus.CANCELLED.value
    db.commit()
    return {
        "status": "cancelled",
        "job": _job_to_dict(job, db)
    }


@router.delete("/jobs/{job_id}")
def delete_sms_job_endpoint(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Admin endpoint to delete a message in retry/failed/queued status so it is logged as deleted and never retried."""
    job = db.get(SMSJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="SMS job not found.")

    if job.state in (SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value):
        raise HTTPException(status_code=400, detail="Cannot delete a message that has already been delivered to carrier.")

    job.state = SMSJobStatus.DELETED.value
    job.error_message = f"Deleted by administrator ({current_user.name})"
    job.attempts = job.max_attempts  # Max out attempts so background workers never retry it
    db.commit()

    return {
        "status": "deleted",
        "job": _job_to_dict(job, db)
    }


@router.delete("/jobs")
def purge_sms_jobs_endpoint(
    state: Optional[str] = Query(None, description="Optional state filter e.g. cancelled, failed, or all"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Admin endpoint to purge/clear notification history."""
    from sqlalchemy import delete
    stmt = delete(SMSJob)
    if state and state != "all":
        stmt = stmt.where(SMSJob.state == state)

    result = db.execute(stmt)
    db.commit()
    return {
        "message": f"Successfully purged {result.rowcount} SMS notification logs.",
        "purged_count": result.rowcount
    }


@router.delete("/notifications")
def purge_notifications_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Admin endpoint to purge all in-app clinical alerts history."""
    from sqlalchemy import delete
    from app.models.notification import Notification
    result = db.execute(delete(Notification))
    db.commit()
    return {
        "message": f"Successfully purged {result.rowcount} in-app clinical alerts.",
        "purged_count": result.rowcount
    }
