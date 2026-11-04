import uuid
import datetime as dt
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base

class StaffAnnouncement(Base):
    __tablename__ = "staff_announcements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50), default="general")  # general, emergency, events, scheduling, training, policy, new_staff
    priority: Mapped[str] = mapped_column(String(20), default="normal")  # normal, important, emergency
    image_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    author_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    author_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    author_role: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    is_auto_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    staff_ref_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    author = relationship("User", foreign_keys=[author_id])
    reads = relationship("StaffAnnouncementRead", back_populates="announcement", cascade="all, delete-orphan")


class StaffAnnouncementRead(Base):
    __tablename__ = "staff_announcement_reads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    announcement_id: Mapped[str] = mapped_column(ForeignKey("staff_announcements.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    read_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    announcement = relationship("StaffAnnouncement", back_populates="reads")
    user = relationship("User", foreign_keys=[user_id])
