import uuid
import datetime as dt
from typing import Optional
from sqlalchemy import String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from app.core.constants import Role

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column("full_name", String(200))
    role: Mapped[str] = mapped_column(String(20), default=Role.PATIENT.value)
    active: Mapped[bool] = mapped_column("is_active", Boolean, default=True)
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    specialization: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)  # for doctors
    availability_status: Mapped[str] = mapped_column(String(20), default="online")  # online, away, busy
    away_until: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    about: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    last_seen_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    mute_notifications: Mapped[bool] = mapped_column(Boolean, default=False)

class PasswordResetTicket(Base):
    __tablename__ = "password_reset_tickets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[Optional[str]] = mapped_column("requester_user_id", ForeignKey("users.id"), index=True, nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(320), nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="open")  # open, resolved, rejected
    reason: Mapped[str] = mapped_column("message", Text, default="")
    handled_by: Mapped[Optional[str]] = mapped_column("resolved_by", ForeignKey("users.id"), nullable=True)
    resolution_notes: Mapped[Optional[str]] = mapped_column("resolution_note", Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    resolved_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

class ProfileUpdateRequest(Base):
    __tablename__ = "profile_update_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    field_name: Mapped[str] = mapped_column(String(100))
    current_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    requested_value: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="pending")  # pending, approved, rejected
    admin_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reviewed_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    reviewed_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
