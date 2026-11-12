import datetime as dt
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.constants import SMSCategory, RiskLevel, DoctorStatus, ConsultationStatus
from app.api.auth import router as auth_router
from app.api.patients import router as patients_router
from app.api.documents import router as documents_router
from app.api.recovery_plans import router as recovery_plans_router
from app.api.medications import router as medications_router
from app.api.appointments import router as appointments_router
from app.api.consultations_api import router as consultations_api_router
from app.api.notifications import router as notifications_router
from app.api.sms_api import router as sms_router
from app.api.admin_api import router as admin_router
from app.api.checkins import router as checkins_router
from app.api.chatbot_api import router as chatbot_router
from app.api.risk_api import router as risk_router
from app.api.roster import router as roster_router
from app.api.audit_api import router as audit_router
from app.api.sync_api import router as sync_router
from app.api.analytics_api import router as analytics_router
from app.api.tts_api import router as tts_router
from app.api.staff_chat_api import router as staff_chat_router, staff_chat_websocket
from app.api.announcements_api import router as announcements_router
from app.api.digital_twin_api import router as digital_twin_router

app = FastAPI(
    title="Recovera AI API",
    description="Hospital-Supervised Post-Discharge Recovery Management Platform (Version 6.0)",
    version="6.0.0"
)

# CORS middleware
origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://([a-zA-Z0-9\-]+\.)*pages\.dev$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(patients_router)
app.include_router(documents_router)
app.include_router(recovery_plans_router)
app.include_router(medications_router)
app.include_router(appointments_router)
app.include_router(consultations_api_router)
app.include_router(notifications_router)
app.include_router(sms_router)
app.include_router(admin_router)
app.include_router(checkins_router)
app.include_router(chatbot_router)
app.include_router(risk_router)
app.include_router(roster_router)
app.include_router(audit_router)
app.include_router(sync_router)
app.include_router(analytics_router)
app.include_router(tts_router)
app.include_router(staff_chat_router)
app.include_router(announcements_router)
app.include_router(digital_twin_router)

# Staff Chat WebSocket alias
@app.websocket("/ws/staff-chat")
async def ws_staff_chat_alias(websocket: WebSocket, token: str = None):
    await staff_chat_websocket(websocket, token)

@app.on_event("startup")
def on_startup():
    """Ensure database tables exist on server startup, preventing missing table errors."""
    from app.models import init_db
    from app.core.database import engine
    from sqlalchemy import text
    init_db()

    # Ensure added columns exist in SQLite or Postgres
    try:
        with engine.begin() as conn:
            if engine.dialect.name == "sqlite":
                user_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(users)")).fetchall()]
                if "phone" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN phone VARCHAR(40);"))
                if "must_change_password" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN must_change_password BOOLEAN DEFAULT 0;"))
                appt_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(appointments)")).fetchall()]
                if "appointment_type" not in appt_cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN appointment_type VARCHAR(50) DEFAULT 'follow_up';"))
                if "decision_status" not in appt_cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN decision_status VARCHAR(30);"))
                if "reschedule_proposed_time" not in appt_cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN reschedule_proposed_time DATETIME;"))
                if "reschedule_reason" not in appt_cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN reschedule_reason TEXT;"))
                audit_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(audit_logs)")).fetchall()]
                if "previous_hash" not in audit_cols:
                    conn.execute(text("ALTER TABLE audit_logs ADD COLUMN previous_hash VARCHAR(64);"))
                if "entry_hash" not in audit_cols:
                    conn.execute(text("ALTER TABLE audit_logs ADD COLUMN entry_hash VARCHAR(64);"))
                doc_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(clinical_documents)")).fetchall()]
                if "assigned_doctor_id" not in doc_cols:
                    conn.execute(text("ALTER TABLE clinical_documents ADD COLUMN assigned_doctor_id VARCHAR(36);"))
                pat_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(patients)")).fetchall()]
                if "care_status" not in pat_cols:
                    conn.execute(text("ALTER TABLE patients ADD COLUMN care_status VARCHAR(20) DEFAULT 'active';"))
                sms_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(sms_jobs)")).fetchall()]
                if "first_failed_at" not in sms_cols:
                    conn.execute(text("ALTER TABLE sms_jobs ADD COLUMN first_failed_at DATETIME;"))
            elif engine.dialect.name == "postgresql":
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(40);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS must_change_password BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(500);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS about VARCHAR(500);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP;"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS mute_notifications BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS appointment_type VARCHAR(50) DEFAULT 'follow_up';"))
                conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS decision_status VARCHAR(30);"))
                conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS reschedule_proposed_time TIMESTAMP;"))
                conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS reschedule_reason TEXT;"))
                conn.execute(text("ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS previous_hash VARCHAR(64);"))
                conn.execute(text("ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS entry_hash VARCHAR(64);"))
                conn.execute(text("ALTER TABLE clinical_documents ADD COLUMN IF NOT EXISTS assigned_doctor_id VARCHAR(36);"))
                conn.execute(text("ALTER TABLE patients ADD COLUMN IF NOT EXISTS care_status VARCHAR(20) DEFAULT 'active';"))
                conn.execute(text("ALTER TABLE sms_jobs ADD COLUMN IF NOT EXISTS first_failed_at TIMESTAMP;"))
                conn.execute(text("ALTER TABLE staff_conversations ADD COLUMN IF NOT EXISTS direct_key VARCHAR(75);"))
                conn.execute(text("ALTER TABLE recovery_plans ADD COLUMN IF NOT EXISTS version_number INTEGER DEFAULT 1;"))
                conn.execute(text("ALTER TABLE recovery_plans ADD COLUMN IF NOT EXISTS source_document_ids TEXT;"))
    except Exception as e:
        pass

    # Seed initial administrator if no admin exists
    try:
        from sqlalchemy import select
        from app.core.database import SessionLocal
        from app.core.constants import Role
        from app.core.security import get_password_hash
        from app.models.user import User

        db = SessionLocal()
        try:
            admin = db.scalar(select(User).where(User.role == Role.ADMIN.value))
            if not admin:
                admin = User(
                    email="admin@hospital.org",
                    password_hash=get_password_hash("AdminSecure2026!"),
                    name="Hospital Administrator",
                    role=Role.ADMIN.value,
                    phone="+919876543210",
                    active=True,
                    availability_status="online"
                )
                db.add(admin)
                db.commit()

            # Initial admin ensured; staff seeding disabled in production
        finally:
            db.close()
    except Exception as e:
        # Graceful fallback: startup continues even if seed warning occurs
        pass

    # Start automated medication reminder background worker thread
    try:
        import threading
        import time

        def _reminder_loop():
            last_retry_ts = 0.0
            while True:
                try:
                    time.sleep(30)
                    now_ts = time.time()
                    from app.core.database import SessionLocal
                    from app.services.medication_reminder_service import process_due_medication_reminders
                    from app.services.sms_gateway_service import process_failed_sms_retries
                    session = SessionLocal()
                    try:
                        process_due_medication_reminders(session)
                        from app.services.medication_reminder_service import process_upcoming_appointment_reminders
                        process_upcoming_appointment_reminders(session)
                        from app.services.sms_gateway_service import process_due_sms_jobs
                        process_due_sms_jobs(session)
                        # Auto-retry failed notifications/SMS every 2 minutes (120s) up to 12 hours
                        if now_ts - last_retry_ts >= 120.0:
                            process_failed_sms_retries(session)
                            last_retry_ts = now_ts
                    finally:
                        session.close()
                except Exception:
                    pass

        t = threading.Thread(target=_reminder_loop, daemon=True)
        t.start()
    except Exception:
        pass

@app.get("/")
def root():
    return {
        "message": "Recovera AI API is operational",
        "version": "6.0.0",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }

@app.get("/health")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": dt.datetime.utcnow().isoformat(),
        "phase": "Phase 6 (Specialization Scheduling & Teleconsultations)",
        "sms_gateway": {
            "driver": "Hospital SMS Gateway",
            "configured_url": settings.ONEPLUS_GATEWAY_URL,
            "enabled": settings.ONEPLUS_GATEWAY_ENABLED,
            "rate_limit_seconds": settings.SMS_RATE_LIMIT_SECONDS,
            "daily_limit": settings.SMS_DAILY_LIMIT,
            "categories": [c.value for c in SMSCategory]
        },
        "operations": {
            "timezone": settings.HOSPITAL_TIMEZONE,
            "consultation_max_minutes": settings.CONSULTATION_MAX_MINUTES,
            "appointment_duration_minutes": settings.APPOINTMENT_DURATION_MINUTES
        }
    }
