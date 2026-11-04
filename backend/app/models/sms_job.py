import uuid
import datetime as dt
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from app.core.constants import SMSJobStatus, SMSCategory

class SMSJob(Base):
    __tablename__ = "sms_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[Optional[str]] = mapped_column(ForeignKey("patients.id"), nullable=True, index=True)
    recipient_phone: Mapped[str] = mapped_column(String(40), index=True)
    category: Mapped[str] = mapped_column(String(40), default=SMSCategory.MEDICATION_REMINDER.value)
    message_text: Mapped[str] = mapped_column(Text)
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    state: Mapped[str] = mapped_column(String(30), default=SMSJobStatus.QUEUED.value)
    scheduled_for: Mapped[dt.datetime] = mapped_column(DateTime, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=360)  # Up to 12 hours at 2-min intervals
    last_attempt: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    first_failed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    provider_message_id: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
