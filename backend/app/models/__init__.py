from app.core.database import Base, engine
from app.models.user import User, PasswordResetTicket, ProfileUpdateRequest
from app.models.patient import Patient, MedicalHistory
from app.models.clinical_document import ClinicalDocument, OCRExtraction
from app.models.recovery_plan import RecoveryPlan
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence, MedicationAdherence
from app.models.checkin import DailyCheckIn
from app.models.appointment import DoctorAvailability, Appointment
from app.models.consultation import Consultation, ConsultationParticipant, ConsultationMessage, ConsultationSignal
from app.models.sms_job import SMSJob
from app.models.risk_history import RiskClassificationHistory
from app.models.audit_log import AuditLog
from app.models.notification import Notification
from app.models.chat_session import ChatSession, ChatMessage
from app.models.staff_chat import (
    StaffConversation,
    StaffConversationParticipant,
    StaffMessage,
    StaffMessageDeletion,
    StaffCall,
)
from app.models.announcement import (
    StaffAnnouncement,
    StaffAnnouncementRead,
)
from app.models.digital_twin import (
    DigitalTwinState,
    DigitalTwinStateHistory,
    RecoveryTrajectoryMilestone,
    DigitalTwinPrediction,
    DigitalTwinDeviationAlert,
)

__all__ = [
    "Base",
    "engine",
    "User",
    "PasswordResetTicket",
    "ProfileUpdateRequest",
    "Patient",
    "MedicalHistory",
    "ClinicalDocument",
    "OCRExtraction",
    "RecoveryPlan",
    "Medication",
    "MedicationSchedule",
    "MedicationOccurrence",
    "MedicationAdherence",
    "DailyCheckIn",
    "DoctorAvailability",
    "Appointment",
    "Consultation",
    "ConsultationParticipant",
    "ConsultationMessage",
    "ConsultationSignal",
    "SMSJob",
    "RiskClassificationHistory",
    "AuditLog",
    "Notification",
    "ChatSession",
    "ChatMessage",
    "StaffConversation",
    "StaffConversationParticipant",
    "StaffMessage",
    "StaffMessageDeletion",
    "StaffCall",
    "StaffAnnouncement",
    "StaffAnnouncementRead",
    "DigitalTwinState",
    "DigitalTwinStateHistory",
    "RecoveryTrajectoryMilestone",
    "DigitalTwinPrediction",
    "DigitalTwinDeviationAlert",
]

def init_db():
    Base.metadata.create_all(bind=engine)
    from sqlalchemy import text
    try:
        with engine.begin() as conn:
            if engine.dialect.name == "sqlite":
                user_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(users)")).fetchall()]
                if "phone" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN phone VARCHAR(40);"))
                if "avatar_url" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN avatar_url VARCHAR(500);"))
                if "about" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN about VARCHAR(500);"))
                if "last_seen_at" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN last_seen_at DATETIME;"))
                if "mute_notifications" not in user_cols:
                    conn.execute(text("ALTER TABLE users ADD COLUMN mute_notifications BOOLEAN DEFAULT 0;"))
                appt_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(appointments)")).fetchall()]
                if "decision_status" not in appt_cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN decision_status VARCHAR(30);"))
                if "reschedule_proposed_time" not in appt_cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN reschedule_proposed_time DATETIME;"))
                if "reschedule_reason" not in appt_cols:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN reschedule_reason TEXT;"))
                conv_cols = [c[1] for c in conn.execute(text("PRAGMA table_info(staff_conversations)")).fetchall()]
                if "direct_key" not in conv_cols:
                    conn.execute(text("ALTER TABLE staff_conversations ADD COLUMN direct_key VARCHAR(75);"))
            elif engine.dialect.name == "postgresql":
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(40);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url VARCHAR(500);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS about VARCHAR(500);"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP;"))
                conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS mute_notifications BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS decision_status VARCHAR(30);"))
                conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS reschedule_proposed_time TIMESTAMP;"))
                conn.execute(text("ALTER TABLE appointments ADD COLUMN IF NOT EXISTS reschedule_reason TEXT;"))
                conn.execute(text("ALTER TABLE staff_conversations ADD COLUMN IF NOT EXISTS direct_key VARCHAR(75);"))
    except Exception:
        pass
