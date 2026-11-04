import uuid
import datetime as dt
from typing import Optional, Dict, Any
from sqlalchemy import String, DateTime, ForeignKey, JSON, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from app.core.constants import DocStatus

class RecoveryPlan(Base):
    __tablename__ = "recovery_plans"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    version_number: Mapped[Optional[int]] = mapped_column(Integer, default=1, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default=DocStatus.PUBLISHED.value if hasattr(DocStatus, "PUBLISHED") else "published")
    source_document_ids: Mapped[Optional[Any]] = mapped_column(JSON, default=list, nullable=True)
    source_versions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=True)
    instructions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=True)
    warning_signs: Mapped[Dict[str, Any]] = mapped_column(JSON, default=list, nullable=True)
    content: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    approved_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    approved_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
