import uuid
import datetime as dt
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func, or_, desc, case

from app.core.database import get_db
from app.core.constants import Role
from app.models.user import User
from app.models.announcement import StaffAnnouncement, StaffAnnouncementRead
from app.api.deps import require_role
from app.services.staff_chat_ws import staff_ws_manager

router = APIRouter(prefix="/api/announcements", tags=["Staff Announcements"])

UPLOAD_ANNOUNCEMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "uploads" / "announcements"
UPLOAD_ANNOUNCEMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Strict staff-only dependency: Patients will receive 403 Forbidden
require_staff = require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value)
require_admin = require_role(Role.ADMIN.value)


class AnnouncementCreateRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=255)
    content: str = Field(..., min_length=5)
    category: str = Field(default="general")  # general, emergency, events, scheduling, training, policy, new_staff
    priority: str = Field(default="normal")  # normal, important, emergency
    image_url: Optional[str] = None


class AnnouncementUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=3, max_length=255)
    content: Optional[str] = Field(None, min_length=5)
    category: Optional[str] = None
    priority: Optional[str] = None
    image_url: Optional[str] = None


def serialize_announcement(announcement: StaffAnnouncement, user_id: str, is_read: bool = False) -> Dict[str, Any]:
    return {
        "id": announcement.id,
        "title": announcement.title,
        "content": announcement.content,
        "category": announcement.category,
        "priority": announcement.priority,
        "image_url": announcement.image_url,
        "author_id": announcement.author_id,
        "author_name": announcement.author_name or (announcement.author.name if announcement.author else "Hospital Administration"),
        "author_role": announcement.author_role or (announcement.author.role if announcement.author else "admin"),
        "is_auto_generated": announcement.is_auto_generated,
        "staff_ref_id": announcement.staff_ref_id,
        "created_at": announcement.created_at.isoformat() if announcement.created_at else None,
        "updated_at": announcement.updated_at.isoformat() if announcement.updated_at else None,
        "is_read": is_read,
    }


def create_new_staff_announcement(db: Session, new_staff_user: User) -> Optional[StaffAnnouncement]:
    """
    Idempotently creates and broadcasts an official welcome announcement
    whenever a new hospital staff member is registered.
    """
    # 1. Prevent duplicate welcome announcements
    existing = db.scalar(
        select(StaffAnnouncement).where(
            StaffAnnouncement.staff_ref_id == new_staff_user.id,
            StaffAnnouncement.is_auto_generated == True
        )
    )
    if existing:
        return existing

    # 2. Pick an admin user as author or use system fallback
    admin_user = db.scalar(select(User).where(User.role == Role.ADMIN.value, User.active == True).limit(1))
    author_id = admin_user.id if admin_user else new_staff_user.id
    author_name = "Hospital Administration"
    author_role = "admin"

    role_display = new_staff_user.role.capitalize()
    specialty_display = f" in {new_staff_user.specialization}" if new_staff_user.specialization else ""
    title = f"Welcome to Our New Team Member – {new_staff_user.name}"
    content = (
        f"We are delighted to welcome {new_staff_user.name} who has joined Recovera AI as a {role_display}{specialty_display}. "
        f"Please extend a warm welcome and reach out to connect on staff chat!"
    )

    announcement = StaffAnnouncement(
        title=title,
        content=content,
        category="new_staff",
        priority="important",
        image_url=new_staff_user.avatar_url or "https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=400&auto=format&fit=crop&q=80",
        author_id=author_id,
        author_name=author_name,
        author_role=author_role,
        is_auto_generated=True,
        staff_ref_id=new_staff_user.id,
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow(),
    )
    db.add(announcement)
    db.flush()

    return announcement


@router.get("", response_model=List[Dict[str, Any]])
def list_announcements(
    category: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    unread_only: bool = Query(False),
    current_user: User = Depends(require_staff),
    db: Session = Depends(get_db)
):
    """
    Lists announcements for all hospital staff.
    Supports filtering by category, priority, unread status, and search query.
    Patients are rejected with 403 Forbidden.
    """
    # Find all announcement IDs read by current_user
    read_ids = set(
        db.scalars(
            select(StaffAnnouncementRead.announcement_id).where(
                StaffAnnouncementRead.user_id == current_user.id
            )
        ).all()
    )

    query = select(StaffAnnouncement)

    # Filtering
    if category and category.lower() != "all":
        query = query.where(func.lower(StaffAnnouncement.category) == category.lower())

    if priority and priority.lower() != "all":
        query = query.where(func.lower(StaffAnnouncement.priority) == priority.lower())

    if search:
        s = f"%{search.strip().lower()}%"
        query = query.where(
            or_(
                func.lower(StaffAnnouncement.title).like(s),
                func.lower(StaffAnnouncement.content).like(s),
                func.lower(StaffAnnouncement.author_name).like(s)
            )
        )

    # Order by: emergency first, then important, then created_at DESC
    priority_order = case(
        (StaffAnnouncement.priority == "emergency", 1),
        (StaffAnnouncement.priority == "important", 2),
        else_=3
    )
    query = query.order_by(priority_order, desc(StaffAnnouncement.created_at))

    announcements = db.scalars(query).all()

    results = []
    for ann in announcements:
        is_read = ann.id in read_ids
        if unread_only and is_read:
            continue
        results.append(serialize_announcement(ann, current_user.id, is_read=is_read))

    return results


@router.get("/unread-count")
def get_unread_announcements_count(
    current_user: User = Depends(require_staff),
    db: Session = Depends(get_db)
):
    """
    Returns unread announcements count for current staff user badge display.
    """
    total = db.scalar(select(func.count(StaffAnnouncement.id))) or 0
    read_count = db.scalar(
        select(func.count(StaffAnnouncementRead.id)).where(
            StaffAnnouncementRead.user_id == current_user.id
        )
    ) or 0
    unread = max(0, total - read_count)
    return {"unread_count": unread, "total": total}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_announcement(
    req: AnnouncementCreateRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Publish a new hospital announcement. Exclusively allowed for Admins.
    Doctors and nurses will receive 403 Forbidden.
    """
    announcement = StaffAnnouncement(
        title=req.title.strip(),
        content=req.content.strip(),
        category=req.category.lower(),
        priority=req.priority.lower(),
        image_url=req.image_url,
        author_id=current_user.id,
        author_name=current_user.name,
        author_role=current_user.role,
        is_auto_generated=False,
        created_at=dt.datetime.utcnow(),
        updated_at=dt.datetime.utcnow(),
    )
    db.add(announcement)
    db.commit()
    db.refresh(announcement)

    # Broadcast to all connected staff in real-time
    serialized = serialize_announcement(announcement, current_user.id, is_read=False)
    try:
        await staff_ws_manager.broadcast_all({
            "type": "new_announcement",
            "data": serialized
        })
    except Exception:
        pass

    return serialized


@router.put("/{announcement_id}")
async def update_announcement(
    announcement_id: str,
    req: AnnouncementUpdateRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Update an existing announcement. Exclusively allowed for Admins.
    """
    announcement = db.get(StaffAnnouncement, announcement_id)
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found.")

    if req.title is not None:
        announcement.title = req.title.strip()
    if req.content is not None:
        announcement.content = req.content.strip()
    if req.category is not None:
        announcement.category = req.category.lower()
    if req.priority is not None:
        announcement.priority = req.priority.lower()
    if req.image_url is not None:
        announcement.image_url = req.image_url

    announcement.updated_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(announcement)

    serialized = serialize_announcement(announcement, current_user.id, is_read=True)
    try:
        await staff_ws_manager.broadcast_all({
            "type": "announcement_updated",
            "data": serialized
        })
    except Exception:
        pass

    return serialized


@router.delete("/{announcement_id}")
async def delete_announcement(
    announcement_id: str,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Delete an announcement. Exclusively allowed for Admins.
    """
    announcement = db.get(StaffAnnouncement, announcement_id)
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found.")

    db.delete(announcement)
    db.commit()

    try:
        await staff_ws_manager.broadcast_all({
            "type": "announcement_deleted",
            "data": {"id": announcement_id}
        })
    except Exception:
        pass

    return {"status": "deleted", "id": announcement_id}


@router.post("/{announcement_id}/read")
def mark_announcement_as_read(
    announcement_id: str,
    current_user: User = Depends(require_staff),
    db: Session = Depends(get_db)
):
    """
    Marks an announcement as read for the current staff user.
    """
    announcement = db.get(StaffAnnouncement, announcement_id)
    if not announcement:
        raise HTTPException(status_code=404, detail="Announcement not found.")

    existing_read = db.scalar(
        select(StaffAnnouncementRead).where(
            StaffAnnouncementRead.announcement_id == announcement_id,
            StaffAnnouncementRead.user_id == current_user.id
        )
    )
    if not existing_read:
        read_record = StaffAnnouncementRead(
            announcement_id=announcement_id,
            user_id=current_user.id,
            read_at=dt.datetime.utcnow()
        )
        db.add(read_record)
        db.commit()

    return {"status": "marked_as_read", "id": announcement_id}


@router.post("/mark-all-read")
def mark_all_announcements_as_read(
    current_user: User = Depends(require_staff),
    db: Session = Depends(get_db)
):
    """
    Marks all announcements as read for the current staff user.
    """
    all_announcements = db.scalars(select(StaffAnnouncement.id)).all()
    already_read = set(
        db.scalars(
            select(StaffAnnouncementRead.announcement_id).where(
                StaffAnnouncementRead.user_id == current_user.id
            )
        ).all()
    )

    to_add = []
    for ann_id in all_announcements:
        if ann_id not in already_read:
            to_add.append(StaffAnnouncementRead(
                announcement_id=ann_id,
                user_id=current_user.id,
                read_at=dt.datetime.utcnow()
            ))

    if to_add:
        db.add_all(to_add)
        db.commit()

    return {"status": "all_marked_read", "count": len(to_add)}


@router.post("/upload-image")
async def upload_announcement_image(
    file: UploadFile = File(...),
    current_user: User = Depends(require_staff)
):
    """
    Accepts staff image upload for announcements (PNG, JPG, JPEG, WEBP, GIF),
    validates file integrity, saves safely, and returns the public serving URL.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No image file provided.")

    allowed_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: PNG, JPG, JPEG, WEBP, GIF."
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds maximum limit of 10MB.")

    file_id = uuid.uuid4().hex[:12]
    saved_name = f"ann_{file_id}{ext}"
    target_path = UPLOAD_ANNOUNCEMENTS_DIR / saved_name

    with open(target_path, "wb") as f:
        f.write(content)

    return {
        "url": f"/api/announcements/images/{saved_name}",
        "filename": saved_name
    }


@router.get("/images/{filename}")
def get_announcement_image(filename: str):
    """Safely serves uploaded announcement cover images."""
    clean_name = Path(filename).name
    file_path = UPLOAD_ANNOUNCEMENTS_DIR / clean_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Announcement image not found.")

    ext = file_path.suffix.lower()
    media_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    media_type = media_map.get(ext, "application/octet-stream")
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"}
    )

