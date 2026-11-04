from enum import Enum

class Role(str, Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    NURSE = "nurse"
    PATIENT = "patient"

class DocStatus(str, Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    NEEDS_REVIEW = "needs_review"
    PUBLISHED = "published"

class OccurrenceState(str, Enum):
    PENDING = "pending"
    TAKEN = "taken"
    NOT_TAKEN = "not_taken"
    MISSED = "missed"
    CANCELLED = "cancelled"

class MedicalSpecialty(str, Enum):
    GENERAL_MEDICINE = "general_medicine"
    NEUROLOGY = "neurology"
    ENT = "ent"
    ORTHOPAEDICS = "orthopaedics"
    PSYCHIATRY = "psychiatry"
    CARDIOLOGY = "cardiology"
    DERMATOLOGY = "dermatology"
    OPHTHALMOLOGY = "ophthalmology"
    PAEDIATRICS = "paediatrics"
    GYNAECOLOGY = "gynaecology"
    ONCOLOGY = "oncology"
    UROLOGY = "urology"
    RADIOLOGY = "radiology"
    ANAESTHESIOLOGY = "anaesthesiology"

class DoctorStatus(str, Enum):
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"  # doctor is in an active consultation

class AppointmentStatus(str, Enum):
    SCHEDULED = "scheduled"
    STARTED = "started"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RESCHEDULED = "rescheduled"
    MISSED = "missed"
    EXPIRED = "expired"

class ConsultationMode(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    CHAT = "chat"

class ConsultationStatus(str, Enum):
    WAITING = "waiting"
    ACCEPTED = "accepted"
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class SMSJobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELETED = "deleted"

class SMSCategory(str, Enum):
    REGISTRATION = "registration"
    MEDICATION_REMINDER = "medication_reminder"
    TRACKER_FOLLOWUP = "tracker_followup"
    APPOINTMENT = "appointment"
    CONSULTATION = "consultation"
    CHECKIN_REMINDER = "checkin_reminder"
    CRITICAL_ALERT = "critical_alert"

class RiskLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    INSUFFICIENT_DATA = "insufficient_data"
