import datetime as dt
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func, update, delete

from app.core.database import get_db
from app.core.constants import Role
from app.models.notification import Notification
from app.models.user import User
from app.api.deps import get_current_user

router = APIRouter(prefix="/api/notifications", tags=["In-App Notifications & Alerts"])


def _notif_to_dict(n: Notification) -> dict:
    return {
        "id": n.id,
        "user_id": n.user_id,
        "title": n.title,
        "message": n.message,
        "category": n.category,
        "link": n.link,
        "is_read": n.is_read,
        "created_at": n.created_at.isoformat() if n.created_at else dt.datetime.utcnow().isoformat(),
    }


def auto_prune_old_notifications(db: Session, max_age_days: int = 15) -> int:
    """Auto-deletes alerts and notifications older than 15 days to optimize database efficiency."""
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=max_age_days)
    try:
        deleted = db.execute(delete(Notification).where(Notification.created_at < cutoff)).rowcount
        if deleted > 0:
            db.commit()
        return deleted
    except Exception:
        db.rollback()
        return 0


@router.get("", response_model=List[dict])
def list_notifications(
    scope: Optional[str] = Query("me", description="'all' for system-wide notifications (admin only), or 'me' for own notifications"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve in-app notifications for the logged-in user or all system alerts for admin. Automatically cleans up alerts older than 15 days."""
    auto_prune_old_notifications(db, max_age_days=15)

    query = select(Notification)
    if current_user.role == Role.ADMIN.value and scope == "all":
        pass
    else:
        query = query.where(Notification.user_id == current_user.id)
    notifs = db.scalars(query.order_by(Notification.created_at.desc()).limit(100)).all()
    return [_notif_to_dict(n) for n in notifs]


@router.post("/prune")
def prune_old_alerts_endpoint(
    days: int = Query(15, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Explicitly prune notifications older than the given days (default 15 days)."""
    pruned = auto_prune_old_notifications(db, max_age_days=days)
    return {"status": "ok", "pruned_count": pruned, "cutoff_days": days}


@router.delete("")
def clear_notifications(
    scope: Optional[str] = Query("all", description="'all' clears all system alerts for admin, or 'me' for user's own alerts"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Purge / clear notifications. Admin can clear all alerts system-wide or for self."""
    if current_user.role == Role.ADMIN.value and scope == "all":
        deleted = db.execute(delete(Notification)).rowcount
    else:
        deleted = db.execute(delete(Notification).where(Notification.user_id == current_user.id)).rowcount
    db.commit()
    return {"message": "In-app alerts cleared successfully.", "deleted_count": deleted}


@router.delete("/{notif_id}")
def delete_single_notification(
    notif_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a single notification by ID."""
    notif = db.get(Notification, notif_id)
    if not notif or (notif.user_id != current_user.id and current_user.role != Role.ADMIN.value):
        raise HTTPException(status_code=404, detail="Notification not found.")
    db.delete(notif)
    db.commit()
    return {"message": "Notification deleted successfully.", "id": notif_id}


@router.get("/unread-count")
def get_unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the unread notification count for the topbar alert symbol. Prunes expired notifications older than 15 days."""
    auto_prune_old_notifications(db, max_age_days=15)
    count = db.scalar(
        select(func.count(Notification.id))
        .where(Notification.user_id == current_user.id, Notification.is_read == False)
    ) or 0
    return {"unread_count": count}


@router.put("/{notif_id}/read")
def mark_as_read(
    notif_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mark a specific notification as read."""
    notif = db.get(Notification, notif_id)
    if not notif or notif.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Notification not found.")

    notif.is_read = True
    db.commit()
    return {"status": "ok", "id": notif.id}


@router.put("/read-all")
def mark_all_as_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mark all unread notifications as read for current user."""
    db.execute(
        update(Notification)
        .where(Notification.user_id == current_user.id, Notification.is_read == False)
        .values(is_read=True)
    )
    db.commit()
    return {"status": "ok"}
