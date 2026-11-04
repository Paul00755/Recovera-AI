import uuid
import datetime as dt
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean, Integer, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class StaffConversation(Base):
    __tablename__ = "staff_conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_type: Mapped[str] = mapped_column(String(20), default="direct")  # direct, group
    direct_key: Mapped[Optional[str]] = mapped_column(String(75), index=True, nullable=True)  # min_uid_max_uid for 1-to-1 uniqueness
    title: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # for group chats
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_by: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)

    participants = relationship("StaffConversationParticipant", back_populates="conversation", cascade="all, delete-orphan")
    messages = relationship("StaffMessage", back_populates="conversation", cascade="all, delete-orphan")


class StaffConversationParticipant(Base):
    __tablename__ = "staff_conversation_participants"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("staff_conversations.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    joined_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    last_read_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    is_muted: Mapped[bool] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool] = mapped_column(Boolean, default=False)
    cleared_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)

    conversation = relationship("StaffConversation", back_populates="participants")
    user = relationship("User", foreign_keys=[user_id])


class StaffMessage(Base):
    __tablename__ = "staff_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("staff_conversations.id", ondelete="CASCADE"), index=True)
    sender_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    message_type: Mapped[str] = mapped_column(String(20), default="text")  # text, voice, call_event
    content: Mapped[str] = mapped_column(Text, default="")
    voice_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    voice_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # in seconds
    reply_to_id: Mapped[Optional[str]] = mapped_column(ForeignKey("staff_messages.id", ondelete="SET NULL"), nullable=True)
    is_edited: Mapped[bool] = mapped_column(Boolean, default=False)
    edited_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    is_deleted_for_everyone: Mapped[bool] = mapped_column(Boolean, default=False)
    deleted_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)

    conversation = relationship("StaffConversation", back_populates="messages")
    sender = relationship("User", foreign_keys=[sender_id])
    reply_to = relationship("StaffMessage", remote_side=[id])


class StaffMessageDeletion(Base):
    """Tracks 'Delete for me' deletions per user without affecting other participants."""
    __tablename__ = "staff_message_deletions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id: Mapped[str] = mapped_column(ForeignKey("staff_messages.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    deleted_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class StaffCall(Base):
    __tablename__ = "staff_calls"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    caller_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    receiver_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    call_type: Mapped[str] = mapped_column(String(20), default="voice")  # voice, video
    status: Mapped[str] = mapped_column(String(20), default="calling")  # calling, connected, completed, missed, declined, busy
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    connected_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)

    caller = relationship("User", foreign_keys=[caller_id])
    receiver = relationship("User", foreign_keys=[receiver_id])
