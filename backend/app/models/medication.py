import uuid
import datetime as dt
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Integer, Text, Date, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from app.core.constants import OccurrenceState

class Medication(Base):
    __tablename__ = "medications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    generic_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    strength: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dose: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    route: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    start_date: Mapped[dt.date] = mapped_column(Date)
    end_date: Mapped[dt.date] = mapped_column(Date)
    instructions: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="active")  # active, discontinued, completed
    version: Mapped[int] = mapped_column(Integer, default=1)
    source_document_id: Mapped[Optional[str]] = mapped_column(ForeignKey("clinical_documents.id"), nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

class MedicationSchedule(Base):
    __tablename__ = "medication_schedules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    medication_id: Mapped[Optional[str]] = mapped_column(ForeignKey("medications.id"), index=True, nullable=True)
    patient_id: Mapped[Optional[str]] = mapped_column(ForeignKey("patients.id"), nullable=True)
    label: Mapped[str] = mapped_column(String(80))  # breakfast, lunch, dinner, etc.
    time_hhmm: Mapped[str] = mapped_column(String(5))  # "08:00", "13:00", etc.
    dose: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    food_relation: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # before_food, after_food, with_food

class MedicationOccurrence(Base):
    __tablename__ = "medication_occurrences"
    __table_args__ = (UniqueConstraint("schedule_id", "scheduled_at", name="uq_occurrence"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    medication_id: Mapped[str] = mapped_column(ForeignKey("medications.id"), index=True)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("medication_schedules.id"), index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    scheduled_at: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    state: Mapped[str] = mapped_column(String(30), default=OccurrenceState.PENDING.value)
    snooze_count: Mapped[int] = mapped_column(Integer, default=0)
    snoozed_until: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    finalized_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

class MedicationAdherence(Base):
    __tablename__ = "medication_adherence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    occurrence_id: Mapped[str] = mapped_column(ForeignKey("medication_occurrences.id"), unique=True, index=True)
    state: Mapped[str] = mapped_column(String(30))  # taken, not_taken
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # forgot, side_effects, felt_unwell, etc.
    recorded_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    recorded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
