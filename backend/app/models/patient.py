import uuid
import datetime as dt
from typing import Optional, Dict, Any
from sqlalchemy import String, DateTime, Boolean, ForeignKey, JSON, Date, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_code: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)
    dob: Mapped[Optional[dt.date]] = mapped_column("date_of_birth", Date, nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    preferred_language: Mapped[str] = mapped_column(String(10), default="en")
    sms_language: Mapped[str] = mapped_column(String(10), default="en")
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    care_status: Mapped[str] = mapped_column(String(20), default="active", index=True)  # active, idle
    sms_opt_in: Mapped[bool] = mapped_column(Boolean, default=False)
    sms_opt_in_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    sms_opt_in_recorded_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    previous_history: Mapped[bool] = mapped_column(Boolean, default=False)
    history: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict, nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    assigned_doctor_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)

    # Patient Active Clinical Condition / Disease Tracking
    active_condition: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    condition_stage: Mapped[Optional[str]] = mapped_column(String(40), default="active", nullable=True)  # acute/prior_state, recovering, stable, resolved
    condition_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    condition_updated_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id], lazy="joined")
    doctor_assignments = relationship("PatientDoctorAssignment", back_populates="patient", cascade="all, delete-orphan", lazy="selectin")

    @property
    def name(self) -> str:
        if self.user and getattr(self.user, "name", None):
            return self.user.name
        return self.patient_code

    @property
    def email(self) -> Optional[str]:
        if self.user and getattr(self.user, "email", None):
            return self.user.email
        return None

class PatientDoctorAssignment(Base):
    __tablename__ = "patient_doctor_assignments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    assigned_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    assigned_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    department: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    revoked_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    revoked_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)

    doctor = relationship("User", foreign_keys=[doctor_id], lazy="joined")
    patient = relationship("Patient", foreign_keys=[patient_id], back_populates="doctor_assignments")

class MedicalHistory(Base):
    __tablename__ = "medical_histories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), unique=True, index=True)
    has_history: Mapped[bool] = mapped_column(Boolean, default=False)
    conditions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=list)
    past_medications: Mapped[Dict[str, Any]] = mapped_column(JSON, default=list)
    past_treatments: Mapped[Dict[str, Any]] = mapped_column(JSON, default=list)
    past_hospitalizations: Mapped[Dict[str, Any]] = mapped_column(JSON, default=list)
    notes: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

class PatientDeletionTicket(Base):
    __tablename__ = "patient_deletion_tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    requested_by: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending, approved, rejected
    admin_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    reviewed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    patient = relationship("Patient", foreign_keys=[patient_id], lazy="joined")
    requester = relationship("User", foreign_keys=[requested_by], lazy="joined")
    reviewer = relationship("User", foreign_keys=[reviewed_by], lazy="joined")

class PatientReactivationTicket(Base):
    __tablename__ = "patient_reactivation_tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"), index=True)
    requested_by: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)  # pending, approved, rejected
    clinical_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    reviewed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    patient = relationship("Patient", foreign_keys=[patient_id], lazy="joined")
    requester = relationship("User", foreign_keys=[requested_by], lazy="joined")
    reviewer = relationship("User", foreign_keys=[reviewed_by], lazy="joined")


