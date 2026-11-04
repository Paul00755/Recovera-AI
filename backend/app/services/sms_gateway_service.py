import uuid
import time
import logging
import datetime as dt
from typing import Optional, Dict, Any, List
import requests
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_

from app.core.config import settings
from app.core.constants import SMSCategory, SMSJobStatus
from app.models.sms_job import SMSJob
from app.models.patient import Patient

logger = logging.getLogger("recovera.sms_gateway")

_last_probe_time: float = 0.0
_cached_probe_result: Optional[Dict[str, Any]] = None

# In-memory companion telemetry state (synced from Android Companion or initialized defaults)
_companion_telemetry: Dict[str, Any] = {
    "battery_pct": 89,
    "signal": "5G VoLTE",
    "carrier": "Hospital SIM (Primary)",
    "last_sync": dt.datetime.utcnow().isoformat(),
    "device_name": "Hospital Android Gateway"
}


class BaseSMSDriver:
    def send(self, phone: str, message: str) -> Dict[str, Any]:
        raise NotImplementedError


class AndroidDeviceDriver(BaseSMSDriver):
    """Communicates with the Android SMS Gateway companion app over authenticated HTTP with automatic failover."""

    def __init__(
        self,
        base_url: str,
        backup_url: Optional[str] = None,
        token: str = "",
        timeout: float = 3.5
    ):
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.backup_url = backup_url.rstrip("/") if backup_url else None
        self.token = token
        self.timeout = timeout

    def _send_to_url(self, target_base_url: str, phone: str, message: str) -> Dict[str, Any]:
        target = target_base_url.rstrip("/")
        if target.endswith(("/send-sms", "/send", "/message", "/api/send")):
            candidate_endpoints = [target]
        else:
            candidate_endpoints = [f"{target}/send", f"{target}/send-sms"]

        payload = {
            "phone": phone,
            "to": phone,
            "message": message,
            "text": message,
            "token": self.token
        }

        last_error = ""
        for endpoint in candidate_endpoints:
            try:
                resp = requests.post(endpoint, json=payload, timeout=self.timeout)
                if resp.status_code in (200, 201, 202):
                    data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                    return {
                        "success": True,
                        "provider_message_id": data.get("id") or data.get("message_id") or data.get("to") or f"gw_{uuid.uuid4().hex[:8]}"
                    }
                last_error = f"Companion HTTP {resp.status_code}: {resp.text}"
                if resp.status_code != 404:
                    break
            except Exception as exc:
                last_error = str(exc)

        return {
            "success": False,
            "error": last_error or "Unable to connect to SMS companion endpoint"
        }

    def send(self, phone: str, message: str) -> Dict[str, Any]:
        primary_err: Optional[str] = None

        # 1. Attempt Primary Endpoint (e.g., Local Hotspot or Wi-Fi IP)
        if self.base_url:
            try:
                result = self._send_to_url(self.base_url, phone, message)
                if result.get("success"):
                    result["endpoint_used"] = "primary"
                    return result
                primary_err = result.get("error", "Unknown primary error")
            except Exception as exc:
                primary_err = str(exc)
                logger.warning("Primary SMS companion dispatch to %s failed: %s", self.base_url, exc)

        # 2. Automatic Failover to Backup Endpoint (e.g., Tailscale mesh IP)
        if self.backup_url:
            logger.info("Initiating automatic failover to backup SMS companion at %s (Reason: %s)", self.backup_url, primary_err)
            try:
                result = self._send_to_url(self.backup_url, phone, message)
                if result.get("success"):
                    result["endpoint_used"] = "backup_failover"
                    return result
                backup_err = result.get("error", "Unknown backup error")
            except Exception as exc:
                backup_err = str(exc)
                logger.warning("Backup SMS companion dispatch to %s failed: %s", self.backup_url, exc)

            return {
                "success": False,
                "error": f"Both endpoints failed. Primary ({self.base_url}): {primary_err}; Backup ({self.backup_url}): {backup_err}"
            }

        return {
            "success": False,
            "error": primary_err or "Primary gateway URL is not configured."
        }


class MockDeviceDriver(BaseSMSDriver):
    """Simulated device driver for regression testing, CI/CD, and offline demonstration."""

    def send(self, phone: str, message: str) -> Dict[str, Any]:
        # Always succeeds in mock mode and generates deterministic test provider ID
        return {
            "success": True,
            "provider_message_id": f"sim_{uuid.uuid4().hex[:10]}"
        }


def get_driver() -> BaseSMSDriver:
    if settings.ONEPLUS_GATEWAY_ENABLED and (settings.ONEPLUS_GATEWAY_URL or settings.ONEPLUS_GATEWAY_BACKUP_URL):
        return AndroidDeviceDriver(
            base_url=settings.ONEPLUS_GATEWAY_URL,
            backup_url=settings.ONEPLUS_GATEWAY_BACKUP_URL,
            token=settings.ONEPLUS_GATEWAY_TOKEN
        )
    return MockDeviceDriver()


def update_companion_telemetry(telemetry_data: Dict[str, Any]) -> Dict[str, Any]:
    global _companion_telemetry
    _companion_telemetry.update({
        "battery_pct": telemetry_data.get("battery_pct", _companion_telemetry["battery_pct"]),
        "signal": telemetry_data.get("signal", _companion_telemetry["signal"]),
        "carrier": telemetry_data.get("carrier", _companion_telemetry["carrier"]),
        "last_sync": dt.datetime.utcnow().isoformat(),
    })
    return _companion_telemetry


def get_gateway_status(db: Session) -> Dict[str, Any]:
    """Compile comprehensive gateway health metrics, limits, real-time connectivity probe, and queue statistics."""
    now = dt.datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    total_queued = db.scalar(
        select(func.count(SMSJob.id)).where(SMSJob.state == SMSJobStatus.QUEUED.value)
    ) or 0
    total_sent = db.scalar(
        select(func.count(SMSJob.id)).where(SMSJob.state.in_([SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value]))
    ) or 0
    total_failed = db.scalar(
        select(func.count(SMSJob.id)).where(SMSJob.state == SMSJobStatus.FAILED.value)
    ) or 0
    total_cancelled = db.scalar(
        select(func.count(SMSJob.id)).where(SMSJob.state == SMSJobStatus.CANCELLED.value)
    ) or 0
    today_sent = db.scalar(
        select(func.count(SMSJob.id)).where(
            SMSJob.created_at >= today_start,
            SMSJob.state.in_([SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value])
        )
    ) or 0

    driver = get_driver()
    driver_type = "Android Companion Gateway" if isinstance(driver, AndroidDeviceDriver) else "Simulated Gateway Driver"

    # Real-time probe of primary and backup endpoints (cached for 5 seconds to prevent latency spikes)
    global _last_probe_time, _cached_probe_result
    now_ts = time.time()
    real_status = "offline"
    latency_ms = None
    handshake_msg = "Simulated Driver Active (Zero-dependency)"
    active_endpoint = None

    if _cached_probe_result and (now_ts - _last_probe_time < 5.0):
        real_status = _cached_probe_result["real_status"]
        latency_ms = _cached_probe_result["latency_ms"]
        handshake_msg = _cached_probe_result["handshake_msg"]
        active_endpoint = _cached_probe_result["active_endpoint"]
    elif settings.ONEPLUS_GATEWAY_ENABLED and settings.ONEPLUS_GATEWAY_URL:
        t0 = time.time()
        try:
            target_probe = settings.ONEPLUS_GATEWAY_URL.rstrip("/")
            probe_url = target_probe if target_probe.endswith(("/send-sms", "/send")) else f"{target_probe}/send-sms"
            resp = requests.post(probe_url, json={}, timeout=0.4)
            elapsed = round((time.time() - t0) * 1000, 1)
            real_status = "online"
            latency_ms = elapsed
            handshake_msg = f"Connected (HTTP {resp.status_code} in {elapsed}ms)"
            active_endpoint = settings.ONEPLUS_GATEWAY_URL
        except Exception as e:
            if settings.ONEPLUS_GATEWAY_BACKUP_URL:
                try:
                    t1 = time.time()
                    b_probe = settings.ONEPLUS_GATEWAY_BACKUP_URL.rstrip("/")
                    b_url = b_probe if b_probe.endswith(("/send-sms", "/send")) else f"{b_probe}/send-sms"
                    b_resp = requests.post(b_url, json={}, timeout=0.4)
                    elapsed = round((time.time() - t1) * 1000, 1)
                    real_status = "online"
                    latency_ms = elapsed
                    handshake_msg = f"Connected via Tailscale Failover (HTTP {b_resp.status_code} in {elapsed}ms)"
                    active_endpoint = settings.ONEPLUS_GATEWAY_BACKUP_URL
                except Exception as b_e:
                    real_status = "offline"
                    latency_ms = None
                    handshake_msg = f"Unreachable: Primary ({type(e).__name__}), Backup ({type(b_e).__name__})"
                    active_endpoint = None
            else:
                real_status = "offline"
                latency_ms = None
                handshake_msg = f"Unreachable: {type(e).__name__}"
                active_endpoint = None

        _last_probe_time = now_ts
        _cached_probe_result = {
            "real_status": real_status,
            "latency_ms": latency_ms,
            "handshake_msg": handshake_msg,
            "active_endpoint": active_endpoint
        }
    elif isinstance(driver, MockDeviceDriver):
        real_status = "online"
        latency_ms = 4.2
        handshake_msg = "Simulated Driver Active & Ready"
        active_endpoint = "Local Virtual Interface"
    else:
        real_status = "offline"
        latency_ms = None
        handshake_msg = "Gateway Disabled"
        active_endpoint = None

    total_attempted = total_sent + total_failed
    delivery_rate = round((total_sent / total_attempted * 100), 1) if total_attempted > 0 else 100.0

    last_job = db.scalar(select(SMSJob).order_by(SMSJob.created_at.desc()).limit(1))

    return {
        "status": real_status,
        "driver": driver_type,
        "is_live_companion": isinstance(driver, AndroidDeviceDriver),
        "gateway_url": settings.ONEPLUS_GATEWAY_URL if settings.ONEPLUS_GATEWAY_ENABLED else None,
        "primary_url": settings.ONEPLUS_GATEWAY_URL if settings.ONEPLUS_GATEWAY_ENABLED else None,
        "backup_url": settings.ONEPLUS_GATEWAY_BACKUP_URL if settings.ONEPLUS_GATEWAY_ENABLED else None,
        "active_endpoint": active_endpoint or settings.ONEPLUS_GATEWAY_URL,
        "fallback_enabled": bool(settings.ONEPLUS_GATEWAY_ENABLED and settings.ONEPLUS_GATEWAY_URL and settings.ONEPLUS_GATEWAY_BACKUP_URL),
        "latency_ms": latency_ms,
        "handshake_msg": handshake_msg,
        "telemetry": {
            "status_text": "Active & Connected" if real_status == "online" else "Disconnected",
            "ping": f"{latency_ms} ms" if latency_ms is not None else "Simulated",
            "transport": "HTTP/REST JSON" if settings.ONEPLUS_GATEWAY_ENABLED else "In-Memory Queue",
            "route": handshake_msg,
            "delivery_rate": f"{delivery_rate}%",
            "last_dispatch": last_job.created_at.isoformat() if last_job else "No messages sent yet"
        },
        "stats": {
            "total_queued": total_queued,
            "total_sent": total_sent,
            "total_failed": total_failed,
            "total_cancelled": total_cancelled,
            "today_sent": today_sent,
            "daily_limit": settings.SMS_DAILY_LIMIT,
            "remaining_today": max(0, settings.SMS_DAILY_LIMIT - today_sent),
            "rate_limit_seconds": settings.SMS_RATE_LIMIT_SECONDS,
            "delivery_rate_pct": delivery_rate
        }
    }


def dispatch_sms_job(db: Session, job: SMSJob) -> bool:
    """Dispatches a single SMSJob through the configured driver with retry and quota guards."""
    if job.state in (SMSJobStatus.CANCELLED.value, SMSJobStatus.DELETED.value):
        logger.info("Skipping dispatch for cancelled/deleted SMSJob %s", job.id)
        return False

    now = dt.datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # 1. Enforce daily limit
    today_count = db.scalar(
        select(func.count(SMSJob.id)).where(
            SMSJob.created_at >= today_start,
            SMSJob.state.in_([SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value])
        )
    ) or 0

    if today_count >= settings.SMS_DAILY_LIMIT:
        job.state = SMSJobStatus.FAILED.value
        job.error_code = "QUOTA_EXCEEDED"
        job.error_message = f"Hospital SMS daily quota ({settings.SMS_DAILY_LIMIT}) has been reached."
        job.last_attempt = now
        db.commit()
        return False

    # 2. Execute via Driver
    driver = get_driver()
    job.attempts += 1
    job.last_attempt = now

    result = driver.send(phone=job.recipient_phone, message=job.message_text)

    if result.get("success"):
        job.state = SMSJobStatus.SENT.value
        job.provider_message_id = result.get("provider_message_id")
        job.error_message = None
        job.first_failed_at = None
        db.commit()
        return True
    else:
        if not getattr(job, "first_failed_at", None):
            job.first_failed_at = now
        job.state = SMSJobStatus.FAILED.value
        job.error_message = result.get("error", "Dispatch failed")
        db.commit()
        return False


def queue_and_send_sms(
    db: Session,
    phone: str,
    message_text: str,
    category: str = SMSCategory.APPOINTMENT.value,
    patient_id: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    scheduled_for: Optional[dt.datetime] = None
) -> Optional[SMSJob]:
    """Queues an SMSJob (with idempotency guard and language localization) and attempts immediate dispatch if scheduled now."""
    clean_phone = phone.strip()
    if patient_id:
        patient = db.get(Patient, patient_id)
        if patient:
            if getattr(patient, "care_status", "active") == "idle" and category in (
                SMSCategory.MEDICATION_REMINDER.value,
                SMSCategory.TRACKER_FOLLOWUP.value,
                SMSCategory.CHECKIN_REMINDER.value
            ):
                logger.info("Suppressing SMS reminder for idle patient %s", patient.patient_code)
                return None
            from app.services.notification_service import localize_sms_message
            pref_lang = (getattr(patient, "sms_language", None) or patient.preferred_language or "en").lower().strip()[:2]
            message_text = localize_sms_message(message_text, pref_lang)
    elif clean_phone:
        patient = db.scalar(select(Patient).where(Patient.phone == clean_phone))
        if patient:
            patient_id = patient.id
            from app.services.notification_service import localize_sms_message
            pref_lang = (getattr(patient, "sms_language", None) or patient.preferred_language or "en").lower().strip()[:2]
            message_text = localize_sms_message(message_text, pref_lang)
    key = idempotency_key or f"sms_{uuid.uuid4().hex}"
    now = dt.datetime.utcnow()
    sched = scheduled_for or now

    # Check for existing idempotency key to prevent duplicates
    existing = db.scalar(select(SMSJob).where(SMSJob.idempotency_key == key))
    if existing:
        return existing

    job = SMSJob(
        patient_id=patient_id,
        recipient_phone=clean_phone,
        category=category,
        message_text=message_text,
        idempotency_key=key,
        state=SMSJobStatus.QUEUED.value,
        scheduled_for=sched,
        created_at=now
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # If scheduled for now (or past), dispatch immediately
    if sched <= now:
        dispatch_sms_job(db, job)

    return job


def retry_sms_job(db: Session, job_id: str) -> SMSJob:
    """Manually re-triggers dispatch of a failed or queued SMS job (Admin action resets 12h window)."""
    job = db.get(SMSJob, job_id)
    if not job:
        raise ValueError("SMS Job not found")

    job.first_failed_at = None
    job.error_message = None
    job.state = SMSJobStatus.QUEUED.value
    dispatch_sms_job(db, job)
    return job


def process_failed_sms_retries(db: Session) -> int:
    """
    Periodically processes failed SMS notifications with automatic retry:
    - Retries once every 2 minutes.
    - Limits auto-retries to 12 hours from the first failure.
    - After 12 hours, auto-retry halts and requires manual administrator retry.
    """
    now = dt.datetime.utcnow()
    twelve_hours_ago = now - dt.timedelta(hours=12)
    two_minutes_ago = now - dt.timedelta(minutes=2)

    # Find jobs in FAILED state with attempts below max
    failed_jobs = db.scalars(
        select(SMSJob).where(
            SMSJob.state == SMSJobStatus.FAILED.value,
            SMSJob.attempts < SMSJob.max_attempts
        )
    ).all()

    retried_count = 0
    for job in failed_jobs:
        # Check first failure timestamp
        fail_time = getattr(job, "first_failed_at", None) or job.created_at

        # If 12 hours have elapsed from first failure, stop auto-retrying
        if fail_time < twelve_hours_ago:
            if not job.error_message or "Auto-retry expired" not in job.error_message:
                job.error_message = f"Auto-retry expired (12h window elapsed). Manual administrator retry required. (Last error: {job.error_message or 'Dispatch failed'})"
                db.commit()
            continue

        # Check if at least 2 minutes have elapsed since last attempt
        if job.last_attempt and job.last_attempt > two_minutes_ago:
            continue

        logger.info("Auto-retrying failed SMSJob %s (attempt %d, failing since %s)", job.id, job.attempts + 1, fail_time.isoformat())
        dispatch_sms_job(db, job)
        retried_count += 1

    return retried_count


def process_due_sms_jobs(db: Session, now: Optional[dt.datetime] = None) -> int:
    """Dispatches any queued SMS jobs whose scheduled_for timestamp has arrived."""
    if now is None:
        now = dt.datetime.utcnow()
    due_jobs = db.scalars(
        select(SMSJob).where(
            SMSJob.state == SMSJobStatus.QUEUED.value,
            SMSJob.scheduled_for <= now
        )
    ).all()
    count = 0
    for job in due_jobs:
        try:
            if dispatch_sms_job(db, job):
                count += 1
        except Exception as e:
            logger.warning("Failed to dispatch due SMS job %s: %s", job.id, e)
    return count


# Compatibility alias for queue_and_send_sms
queue_sms_job = queue_and_send_sms

