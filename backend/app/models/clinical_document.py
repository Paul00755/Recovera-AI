import uuid
import datetime as dt
from typing import Optional, Dict, Any
from sqlalchemy import String, DateTime, ForeignKey, JSON, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base
from app.core.constants import DocStatus

class ClinicalDocument(Base):
    __tablename__ = "clinical_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    doc_type: Mapped[str] = mapped_column(String(40))  # prescription, discharge_summary, recovery_instructions, follow_up_schedule
    filename: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(30), default=DocStatus.DRAFT.value)
    version: Mapped[int] = mapped_column(Integer, default=1)
    extracted_text: Mapped[str] = mapped_column(Text, default="")
    structured_data: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    uploaded_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    reviewed_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    assigned_doctor_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

class OCRExtraction(Base):
    __tablename__ = "ocr_extractions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("clinical_documents.id"), unique=True, index=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    cleaned_text: Mapped[str] = mapped_column(Text, default="")
    extracted_fields: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    confidence_scores: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
