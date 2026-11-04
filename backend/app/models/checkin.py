import uuid
import datetime as dt
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Integer, Float, Boolean, Text, Date
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base

class DailyCheckIn(Base):
    __tablename__ = "daily_checkins"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    checkin_date: Mapped[dt.date] = mapped_column(Date, index=True)
    pain_level: Mapped[int] = mapped_column(Integer)  # 0 to 10 scale
    pain_location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    sleep_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    activity_level: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # rest, light, normal, active
    fever: Mapped[bool] = mapped_column(Boolean, default=False)
    cough: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    submitted_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
