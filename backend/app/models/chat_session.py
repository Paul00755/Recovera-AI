import uuid
import datetime as dt
from typing import Optional, List, Dict, Any
from sqlalchemy import String, DateTime, ForeignKey, JSON, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    patient_id: Mapped[Optional[str]] = mapped_column(ForeignKey("patients.id", ondelete="SET NULL"), nullable=True, index=True)
    role: Mapped[str] = mapped_column(String(20), default="patient")
    title: Mapped[str] = mapped_column(String(255), default="New Consultation")
    language: Mapped[str] = mapped_column(String(10), default="en")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    sender: Mapped[str] = mapped_column(String(20)) # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(JSON, nullable=True, default=list)
    attachment_info: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False)
    prescription_boundary: Mapped[bool] = mapped_column(Boolean, default=False)
    model_used: Mapped[str] = mapped_column(String(64), default="gemini-2.5-flash")
    suggested_actions: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)

    session = relationship("ChatSession", back_populates="messages")
