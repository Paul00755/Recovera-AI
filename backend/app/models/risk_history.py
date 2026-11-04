import uuid
import datetime as dt
from typing import Optional, Dict, Any
from sqlalchemy import String, DateTime, ForeignKey, Boolean, Integer, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from app.core.constants import RiskLevel

class RiskClassificationHistory(Base):
    __tablename__ = "risk_classification_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    level: Mapped[str] = mapped_column(String(30), default=RiskLevel.INSUFFICIENT_DATA.value)
    explainability_reasons: Mapped[Dict[str, Any]] = mapped_column(JSON, default=list)
    insufficient_data: Mapped[bool] = mapped_column(Boolean, default=False)
    data_window_days: Mapped[int] = mapped_column(Integer, default=7)
    is_override: Mapped[bool] = mapped_column(Boolean, default=False)
    overridden_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    override_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evaluated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
