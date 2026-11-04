import uuid
import datetime as dt
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Boolean, Integer, Time, Date, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from app.core.constants import AppointmentStatus, DoctorStatus

class DoctorAvailability(Base):
    __tablename__ = "doctor_availabilities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default=DoctorStatus.ONLINE.value)  # online, away
    day_of_week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 0=Mon, 6=Sun (for recurring)
    start_time: Mapped[dt.time] = mapped_column(Time)
    end_time: Mapped[dt.time] = mapped_column(Time)
    specific_date: Mapped[Optional[dt.date]] = mapped_column(Date, nullable=True)  # for date-specific
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    booked_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    start_time: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    end_time: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    status: Mapped[str] = mapped_column(String(30), default=AppointmentStatus.SCHEDULED.value)
    appointment_type: Mapped[str] = mapped_column(String(50), default="follow_up")
    decision_status: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)  # accepted, denied, reschedule_requested
    reschedule_proposed_time: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    reschedule_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
