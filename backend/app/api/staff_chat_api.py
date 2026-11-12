import os
import uuid
import datetime as dt
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Form,
    Query
)
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, or_, and_, func
from sqlalchemy.orm import Session

from app.core.database import get_db, SessionLocal
from app.core.constants import Role
from app.core.security import decode_access_token
from app.core.audit import record_audit
from app.models.user import User
from app.models.notification import Notification
from app.models.staff_chat import (
    StaffConversation,
    StaffConversationParticipant,
    StaffMessage,
    StaffMessageDeletion,
    StaffCall,
)
from app.api.deps import get_current_user, require_role, bearer_scheme
from fastapi.security import HTTPAuthorizationCredentials
from app.services.staff_chat_ws import staff_ws_manager
from app.services.staff_chat_service import (
    format_utc_iso,
    get_or_create_direct_conversation,
    serialize_staff_user,
    serialize_message,
    serialize_conversation_summary,
)

router = APIRouter(prefix="/api/staff-chat", tags=["Staff Internal Communication"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads" / "voice_messages"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class CreateConversationRequest(BaseModel):
    target_user_id: Optional[str] = None
    conversation_type: Optional[str] = "direct"
    title: Optional[str] = None
    participant_ids: Optional[List[str]] = None

class SendMessageRequest(BaseModel):
    content: Optional[str] = ""
    message_type: Optional[str] = "text"  # text, voice, call_event
    voice_url: Optional[str] = None
    voice_duration: Optional[float] = None
    reply_to_id: Optional[str] = None

class EditMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)

class UpdateProfileRequest(BaseModel):
    avatar_url: Optional[str] = None
    about: Optional[str] = None
    availability_status: Optional[str] = None  # online, away, busy, offline
    mute_notifications: Optional[bool] = None

class InitiateCallRequest(BaseModel):
    receiver_id: str
    call_type: str = "voice"  # voice, video

class UpdateCallRequest(BaseModel):
    status: str  # connected, completed, missed, declined, busy
    duration_seconds: Optional[int] = 0

# ---------------------------------------------------------------------------
# Security & Authorization Helper
# ---------------------------------------------------------------------------

def enforce_staff_access(user: User):
    if user.role not in (Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Staff internal communication is exclusively accessible to hospital staff."
        )

# ---------------------------------------------------------------------------
# 1. Staff Directory
# ---------------------------------------------------------------------------

@router.get("/directory")
def get_staff_directory(
    q: Optional[str] = None,
    role: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    query = select(User).where(
        User.role.in_([Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value]),
        User.active == True
    )

    if role and role != "all":
        query = query.where(User.role == role)

    if q:
        search_pattern = f"%{q.strip()}%"
        query = query.where(
            or_(
                User.name.ilike(search_pattern),
                User.email.ilike(search_pattern),
                User.specialization.ilike(search_pattern)
            )
        )

    users = db.scalars(query.order_by(User.name.asc())).all()
    online_ids = staff_ws_manager.get_online_user_ids()
    return [serialize_staff_user(u, online_ids) for u in users]

# ---------------------------------------------------------------------------
# 2. Staff Conversations & Unread Counts
# ---------------------------------------------------------------------------

@router.get("/unread-count")
def get_staff_chat_unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Ultra-fast lightweight unread badge query for the main shell navigation."""
    enforce_staff_access(current_user)

    subquery_deleted = (
        select(StaffMessageDeletion.message_id)
        .where(StaffMessageDeletion.user_id == current_user.id)
        .scalar_subquery()
    )

    query = (
        select(func.count(StaffMessage.id))
        .join(StaffConversationParticipant, StaffConversationParticipant.conversation_id == StaffMessage.conversation_id)
        .where(
            StaffConversationParticipant.user_id == current_user.id,
            StaffConversationParticipant.is_archived == False,
            StaffMessage.sender_id != current_user.id,
            StaffMessage.is_deleted_for_everyone == False,
            or_(
                StaffConversationParticipant.last_read_at == None,
                StaffMessage.created_at > StaffConversationParticipant.last_read_at
            ),
            or_(
                StaffConversationParticipant.cleared_at == None,
                StaffMessage.created_at > StaffConversationParticipant.cleared_at
            ),
            StaffMessage.id.not_in(subquery_deleted)
        )
    )
    total_unread = db.scalar(query) or 0
    return {"unread_count": total_unread}

@router.get("/conversations")
def list_conversations(
    filter_role: Optional[str] = None,
    unread_only: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    from sqlalchemy.orm import joinedload

    # Find conversations current_user participates in
    conv_ids = db.scalars(
        select(StaffConversationParticipant.conversation_id)
        .where(
            StaffConversationParticipant.user_id == current_user.id,
            StaffConversationParticipant.is_archived == False
        )
    ).all()

    if not conv_ids:
        return []

    convs = db.scalars(
        select(StaffConversation)
        .options(
            joinedload(StaffConversation.participants).joinedload(StaffConversationParticipant.user)
        )
        .where(StaffConversation.id.in_(conv_ids))
        .order_by(StaffConversation.updated_at.desc())
    ).unique().all()

    # Deduplicate direct conversations by peer id
    seen_peers = set()
    deduped_convs = []
    for c in convs:
        if c.conversation_type == "direct":
            peer_id = None
            if getattr(c, "direct_key", None):
                parts = c.direct_key.split("_")
                if len(parts) == 2:
                    peer_id = parts[1] if parts[0] == current_user.id else parts[0]
            if not peer_id:
                for p in c.participants:
                    if p.user_id != current_user.id:
                        peer_id = p.user_id
                        break
            if peer_id:
                if peer_id in seen_peers:
                    continue
                seen_peers.add(peer_id)
        deduped_convs.append(c)

    online_ids = staff_ws_manager.get_online_user_ids()
    results = [serialize_conversation_summary(c, current_user.id, db, online_ids) for c in deduped_convs]

    if filter_role and filter_role != "all":
        results = [
            r for r in results
            if (r.get("other_participant") or r.get("other_user")) and
            (r.get("other_participant") or r.get("other_user")).get("role") == filter_role
        ]

    if unread_only:
        results = [r for r in results if r.get("unread_count", 0) > 0]

    return results

@router.post("/conversations", status_code=status.HTTP_201_CREATED)
@router.post("/conversations/direct", status_code=status.HTTP_200_OK)
def start_or_get_conversation(
    req: CreateConversationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    if req.conversation_type == "direct" or req.target_user_id:
        if not req.target_user_id:
            raise HTTPException(status_code=422, detail="target_user_id required for direct conversation.")
        if req.target_user_id == current_user.id:
            raise HTTPException(status_code=400, detail="Cannot start a direct conversation with yourself.")

        target = db.get(User, req.target_user_id)
        if not target or not target.active or target.role == Role.PATIENT.value:
            raise HTTPException(status_code=404, detail="Target hospital staff member not found.")

        conv = get_or_create_direct_conversation(db, current_user.id, target.id)
        online_ids = staff_ws_manager.get_online_user_ids()
        return serialize_conversation_summary(conv, current_user.id, db, online_ids)

    # Group Conversation
    now = dt.datetime.utcnow()
    conv = StaffConversation(
        id=str(uuid.uuid4()),
        conversation_type="group",
        title=req.title or "Staff Team Chat",
        created_by=current_user.id,
        created_at=now,
        updated_at=now
    )
    db.add(conv)
    db.flush()

    participant_ids = set(req.participant_ids or [])
    participant_ids.add(current_user.id)

    for uid in participant_ids:
        u = db.get(User, uid)
        if u and u.active and u.role != Role.PATIENT.value:
            part = StaffConversationParticipant(
                id=str(uuid.uuid4()),
                conversation_id=conv.id,
                user_id=uid,
                joined_at=now
            )
            db.add(part)

    db.commit()
    db.refresh(conv)
    online_ids = staff_ws_manager.get_online_user_ids()
    return serialize_conversation_summary(conv, current_user.id, db, online_ids)

# ---------------------------------------------------------------------------
# 3. Messages & History
# ---------------------------------------------------------------------------

@router.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(
    conversation_id: str,
    limit: int = 100,
    before: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    participant = db.scalar(
        select(StaffConversationParticipant).where(
            StaffConversationParticipant.conversation_id == conversation_id,
            StaffConversationParticipant.user_id == current_user.id
        )
    )
    if not participant:
        raise HTTPException(status_code=403, detail="Access denied: Not a participant in this conversation.")

    # Exclude deleted-for-me
    deleted_msg_ids = set(
        db.scalars(
            select(StaffMessageDeletion.message_id).where(
                StaffMessageDeletion.user_id == current_user.id
            )
        ).all()
    )

    from sqlalchemy.orm import joinedload
    query = select(StaffMessage).options(
        joinedload(StaffMessage.sender),
        joinedload(StaffMessage.reply_to).joinedload(StaffMessage.sender)
    ).where(StaffMessage.conversation_id == conversation_id)
    if participant.cleared_at:
        query = query.where(StaffMessage.created_at > participant.cleared_at)
    if deleted_msg_ids:
        query = query.where(StaffMessage.id.not_in(deleted_msg_ids))

    if before:
        try:
            cleaned = before.rstrip("Z")
            before_dt = dt.datetime.fromisoformat(cleaned)
            query = query.where(StaffMessage.created_at < before_dt)
        except Exception:
            pass

    messages = db.scalars(query.order_by(StaffMessage.created_at.asc()).limit(limit)).unique().all()

    # Automatically mark messages as read
    participant.last_read_at = dt.datetime.utcnow() + dt.timedelta(seconds=2)
    db.commit()

    return [serialize_message(m, current_user.id, db) for m in messages]

@router.post("/conversations/{conversation_id}/messages", status_code=status.HTTP_201_CREATED)
async def send_message(
    conversation_id: str,
    req: SendMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    participant = db.scalar(
        select(StaffConversationParticipant).where(
            StaffConversationParticipant.conversation_id == conversation_id,
            StaffConversationParticipant.user_id == current_user.id
        )
    )
    if not participant:
        raise HTTPException(status_code=403, detail="Access denied: Not a participant in this conversation.")

    if req.message_type == "text" and not req.content.strip():
        raise HTTPException(status_code=422, detail="Message content cannot be empty.")

    now = dt.datetime.utcnow()
    msg = StaffMessage(
        id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        sender_id=current_user.id,
        message_type=req.message_type or "text",
        content=req.content.strip() if req.content else ("Voice message" if req.message_type == "voice" else ""),
        voice_url=req.voice_url,
        voice_duration=req.voice_duration,
        reply_to_id=req.reply_to_id,
        created_at=now
    )
    db.add(msg)

    # Update conversation timestamp & auto-read for sender
    conv = db.get(StaffConversation, conversation_id)
    if conv:
        conv.updated_at = now
    participant.last_read_at = now

    db.commit()
    db.refresh(msg)

    # Broadcast new message event to all participants via WebSocket
    serialized = serialize_message(msg, current_user.id, db)
    participant_ids = db.scalars(
        select(StaffConversationParticipant.user_id).where(
            StaffConversationParticipant.conversation_id == conversation_id
        )
    ).all()

    await staff_ws_manager.broadcast_to_users(
        participant_ids,
        {
            "type": "new_message",
            "conversation_id": conversation_id,
            "message": serialized
        }
    )

    return serialized

# ---------------------------------------------------------------------------
# 4. Message Editing & Deletion
# ---------------------------------------------------------------------------

@router.put("/messages/{message_id}")
async def edit_message(
    message_id: str,
    req: EditMessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    msg = db.get(StaffMessage, message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found.")
    if msg.sender_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot edit another staff member's message.")
    if msg.is_deleted_for_everyone:
        raise HTTPException(status_code=400, detail="Cannot edit a deleted message.")
    if msg.message_type != "text":
        raise HTTPException(status_code=400, detail="Only text messages can be edited.")

    msg.content = req.content.strip()
    msg.is_edited = True
    msg.edited_at = dt.datetime.utcnow()
    db.commit()
    db.refresh(msg)

    serialized = serialize_message(msg, current_user.id, db)
    participant_ids = db.scalars(
        select(StaffConversationParticipant.user_id).where(
            StaffConversationParticipant.conversation_id == msg.conversation_id
        )
    ).all()

    await staff_ws_manager.broadcast_to_users(
        participant_ids,
        {
            "type": "message_updated",
            "conversation_id": msg.conversation_id,
            "message": serialized
        }
    )

    return serialized

@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: str,
    mode: str = Query("me", regex="^(me|everyone)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    msg = db.get(StaffMessage, message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found.")

    participant = db.scalar(
        select(StaffConversationParticipant).where(
            StaffConversationParticipant.conversation_id == msg.conversation_id,
            StaffConversationParticipant.user_id == current_user.id
        )
    )
    if not participant:
        raise HTTPException(status_code=403, detail="Access denied.")

    participant_ids = db.scalars(
        select(StaffConversationParticipant.user_id).where(
            StaffConversationParticipant.conversation_id == msg.conversation_id
        )
    ).all()

    now = dt.datetime.utcnow()

    if mode == "me":
        # Delete for me only
        existing = db.scalar(
            select(StaffMessageDeletion).where(
                StaffMessageDeletion.message_id == message_id,
                StaffMessageDeletion.user_id == current_user.id
            )
        )
        if not existing:
            del_rec = StaffMessageDeletion(
                id=str(uuid.uuid4()),
                message_id=message_id,
                user_id=current_user.id,
                deleted_at=now
            )
            db.add(del_rec)
            db.commit()

        await staff_ws_manager.send_to_user(
            current_user.id,
            {
                "type": "message_deleted_for_me",
                "conversation_id": msg.conversation_id,
                "message_id": message_id
            }
        )
        return {"status": "deleted_for_me", "message_id": message_id}

    elif mode == "everyone":
        # Delete for everyone: Only sender or Administrator
        if msg.sender_id != current_user.id and current_user.role != Role.ADMIN.value:
            raise HTTPException(status_code=403, detail="Cannot delete for everyone: Only message author or administrator can delete for everyone.")

        msg.is_deleted_for_everyone = True
        msg.deleted_at = now
        msg.content = "🚫 This message was deleted"
        msg.voice_url = None

        record_audit(
            db=db,
            actor_id=current_user.id,
            action="staff_message_deleted_for_everyone",
            target_type="staff_message",
            target_id=msg.id,
            details={"conversation_id": msg.conversation_id, "sender_id": msg.sender_id}
        )
        db.commit()
        db.refresh(msg)

        serialized = serialize_message(msg, current_user.id, db)
        await staff_ws_manager.broadcast_to_users(
            participant_ids,
            {
                "type": "message_deleted_for_everyone",
                "conversation_id": msg.conversation_id,
                "message": serialized
            }
        )
        return serialized

@router.delete("/messages/{message_id}/delete-for-me")
async def delete_message_for_me_alias(
    message_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return await delete_message(message_id=message_id, mode="me", db=db, current_user=current_user)

@router.delete("/messages/{message_id}/delete-for-everyone")
async def delete_message_for_everyone_alias(
    message_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return await delete_message(message_id=message_id, mode="everyone", db=db, current_user=current_user)

# ---------------------------------------------------------------------------
# 5. Read Receipts & Clear Chat
# ---------------------------------------------------------------------------

@router.post("/conversations/{conversation_id}/read")
async def mark_conversation_read(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    participant = db.scalar(
        select(StaffConversationParticipant).where(
            StaffConversationParticipant.conversation_id == conversation_id,
            StaffConversationParticipant.user_id == current_user.id
        )
    )
    if not participant:
        raise HTTPException(status_code=403, detail="Access denied.")

    now = dt.datetime.utcnow()
    participant.last_read_at = now + dt.timedelta(seconds=2)
    db.commit()

    participant_ids = db.scalars(
        select(StaffConversationParticipant.user_id).where(
            StaffConversationParticipant.conversation_id == conversation_id,
            StaffConversationParticipant.user_id != current_user.id
        )
    ).all()

    await staff_ws_manager.broadcast_to_users(
        participant_ids,
        {
            "type": "read_receipt",
            "conversation_id": conversation_id,
            "read_by": current_user.id,
            "read_at": format_utc_iso(now)
        }
    )

    # Calculate remaining total unread count for current_user
    total_unread = get_staff_chat_unread_count(db, current_user)["unread_count"]

    # Also notify current_user over WebSocket so all their open tabs/sidebar update immediately
    await staff_ws_manager.broadcast_to_users(
        [current_user.id],
        {
            "type": "staff_chat_unread_update",
            "conversation_id": conversation_id,
            "unread_count": total_unread
        }
    )

    return {"status": "read", "conversation_id": conversation_id, "unread_count": total_unread}

@router.post("/conversations/{conversation_id}/clear")
def clear_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    participant = db.scalar(
        select(StaffConversationParticipant).where(
            StaffConversationParticipant.conversation_id == conversation_id,
            StaffConversationParticipant.user_id == current_user.id
        )
    )
    if not participant:
        raise HTTPException(status_code=403, detail="Access denied.")

    participant.cleared_at = dt.datetime.utcnow()
    db.commit()
    return {"status": "cleared", "conversation_id": conversation_id}

# ---------------------------------------------------------------------------
# 6. Conversation Search (Across Historical Messages)
# ---------------------------------------------------------------------------

@router.get("/conversations/{conversation_id}/search")
def search_in_conversation(
    conversation_id: str,
    q: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    participant = db.scalar(
        select(StaffConversationParticipant).where(
            StaffConversationParticipant.conversation_id == conversation_id,
            StaffConversationParticipant.user_id == current_user.id
        )
    )
    if not participant:
        raise HTTPException(status_code=403, detail="Access denied.")

    deleted_msg_ids = set(
        db.scalars(
            select(StaffMessageDeletion.message_id).where(
                StaffMessageDeletion.user_id == current_user.id
            )
        ).all()
    )

    query = select(StaffMessage).where(
        StaffMessage.conversation_id == conversation_id,
        StaffMessage.is_deleted_for_everyone == False,
        StaffMessage.content.ilike(f"%{q.strip()}%")
    )
    if participant.cleared_at:
        query = query.where(StaffMessage.created_at > participant.cleared_at)
    if deleted_msg_ids:
        query = query.where(StaffMessage.id.not_in(deleted_msg_ids))

    matches = db.scalars(query.order_by(StaffMessage.created_at.asc())).all()

    return {
        "query": q,
        "total_results": len(matches),
        "results": [serialize_message(m, current_user.id, db) for m in matches]
    }

# ---------------------------------------------------------------------------
# 7. Voice Messages Storage & Audio Streaming
# ---------------------------------------------------------------------------

@router.post("/voice")
async def upload_voice_message(
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    duration: Optional[float] = Form(None),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    target_audio = file or audio
    if not target_audio:
        raise HTTPException(status_code=400, detail="Audio file required.")

    # Validate audio MIME
    content_type = target_audio.content_type or ""
    if not (content_type.startswith("audio/") or content_type in ("video/webm", "application/octet-stream")):
        raise HTTPException(status_code=400, detail="Invalid audio file type. Only audio recordings are supported.")

    file_id = str(uuid.uuid4())
    ext = ".webm" if "webm" in content_type else ".mp3" if "mpeg" in content_type else ".wav"
    file_name = f"voice_{file_id}{ext}"
    file_path = UPLOAD_DIR / file_name

    contents = await target_audio.read()
    if len(contents) < 10:
        raise HTTPException(status_code=400, detail="Audio file too short or empty.")
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file exceeds 25MB limit.")

    with open(file_path, "wb") as f:
        f.write(contents)

    voice_url = f"/api/staff-chat/voice/{file_name}"
    return {
        "voice_url": voice_url,
        "filename": file_name,
        "duration": duration or 0.0,
        "size_bytes": len(contents)
    }

@router.get("/voice/{filename}")
def stream_voice_message(
    filename: str,
    token: Optional[str] = Query(None),
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db)
):
    # Authenticate via Bearer header or token query parameter
    auth_token = creds.credentials if creds else token
    if auth_token:
        try:
            payload = decode_access_token(auth_token)
            user_id = payload.get("sub")
            if user_id:
                user = db.get(User, user_id)
                if user and user.role == Role.PATIENT.value:
                    raise HTTPException(status_code=403, detail="Staff access only.")
        except HTTPException:
            raise
        except Exception:
            pass

    clean_filename = Path(filename).name
    file_path = UPLOAD_DIR / clean_filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio message recording not found.")

    content_type = "audio/webm"
    if clean_filename.endswith(".wav"):
        content_type = "audio/wav"
    elif clean_filename.endswith(".mp3"):
        content_type = "audio/mpeg"

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f"inline; filename={clean_filename}"
    }
    return FileResponse(file_path, media_type=content_type, headers=headers)

# ---------------------------------------------------------------------------
# 8. Staff Profile Management & Presence
# ---------------------------------------------------------------------------

@router.get("/profile")
def get_my_staff_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)
    online_ids = staff_ws_manager.get_online_user_ids()
    res = serialize_staff_user(current_user, online_ids)
    res["mute_notifications"] = getattr(current_user, "mute_notifications", False)
    return res

@router.put("/profile")
async def update_my_staff_profile(
    req: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    if req.avatar_url is not None:
        current_user.avatar_url = req.avatar_url
    if req.about is not None:
        current_user.about = req.about.strip()[:500]
    if req.availability_status is not None:
        current_user.availability_status = req.availability_status
    if req.mute_notifications is not None:
        current_user.mute_notifications = req.mute_notifications

    db.commit()
    db.refresh(current_user)

    # Broadcast presence update via WebSocket
    await staff_ws_manager.broadcast_presence(
        current_user.id,
        current_user.availability_status,
        custom_status=current_user.about
    )

    online_ids = staff_ws_manager.get_online_user_ids()
    res = serialize_staff_user(current_user, online_ids)
    res["mute_notifications"] = getattr(current_user, "mute_notifications", False)
    return res

# ---------------------------------------------------------------------------
# 9. Voice & Video Calls API
# ---------------------------------------------------------------------------

@router.post("/calls", status_code=status.HTTP_201_CREATED)
async def initiate_call(
    req: InitiateCallRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    receiver = db.get(User, req.receiver_id)
    if not receiver or not receiver.active or receiver.role == Role.PATIENT.value:
        raise HTTPException(status_code=404, detail="Target hospital staff member not found.")

    online_user_ids = staff_ws_manager.get_online_user_ids()
    is_receiver_online = (
        receiver.id in online_user_ids and
        (receiver.availability_status or "online") != "offline"
    )

    call_label = "video" if req.call_type == "video" else "voice"

    # 1. Receiver is OFFLINE: do not initiate ringing call, send in-app notification & alert
    if not is_receiver_online or (receiver.availability_status or "") == "offline":
        notif = Notification(
            user_id=receiver.id,
            title=f"Missed {call_label.capitalize()} Call",
            message=f"{current_user.name} was trying to reach you via {call_label} call.",
            category="staff_call",
            link="/staff/staff-chat",
            is_read=False
        )
        db.add(notif)

        # Log call_event in direct conversation history
        conv = get_or_create_direct_conversation(db, current_user.id, receiver.id)
        missed_msg = StaffMessage(
            conversation_id=conv.id,
            sender_id=current_user.id,
            content=f"Missed {call_label} call",
            message_type="call_event",
            created_at=dt.datetime.utcnow()
        )
        db.add(missed_msg)
        conv.updated_at = missed_msg.created_at
        db.commit()

        # Broadcast the new missed call message
        await staff_ws_manager.broadcast_to_users(
            [receiver.id, current_user.id],
            {
                "type": "new_message",
                "conversation_id": conv.id,
                "message": serialize_message(missed_msg, db=db, current_user_id=current_user.id)
            }
        )

        return {
            "offline": True,
            "detail": f"{receiver.name} is currently offline. An in-app alert has been sent to them.",
            "receiver_name": receiver.name,
            "call_type": req.call_type
        }

    # 2. Collision / Glare handling: check if receiver is already in the middle of calling current_user
    existing_call = db.scalar(
        select(StaffCall).where(
            StaffCall.caller_id == receiver.id,
            StaffCall.receiver_id == current_user.id,
            StaffCall.status == "calling"
        ).order_by(StaffCall.started_at.desc())
    )
    if existing_call:
        existing_call.status = "connected"
        existing_call.connected_at = dt.datetime.utcnow()
        db.commit()
        db.refresh(existing_call)

        await staff_ws_manager.send_to_user(
            receiver.id,
            {
                "type": "call_accept",
                "call_id": existing_call.id,
                "user_id": current_user.id
            }
        )
        return {
            "call_id": existing_call.id,
            "caller_id": existing_call.caller_id,
            "receiver_id": existing_call.receiver_id,
            "call_type": existing_call.call_type,
            "status": "connected",
            "collision_resolved": True,
            "started_at": format_utc_iso(existing_call.started_at)
        }

    # 3. Standard New Call Initiation
    now = dt.datetime.utcnow()
    call = StaffCall(
        id=str(uuid.uuid4()),
        caller_id=current_user.id,
        receiver_id=receiver.id,
        call_type=req.call_type or "voice",
        status="calling",
        started_at=now
    )
    db.add(call)
    db.commit()
    db.refresh(call)

    # Send incoming call invite over WebSocket to receiver
    await staff_ws_manager.send_to_user(
        receiver.id,
        {
            "type": "incoming_call",
            "call_id": call.id,
            "call_type": call.call_type,
            "caller_id": current_user.id,
            "caller_name": current_user.name,
            "caller_role": current_user.role,
            "caller_avatar": getattr(current_user, "avatar_url", None),
            "caller_specialization": current_user.specialization or current_user.role,
            "caller": serialize_staff_user(current_user, online_user_ids)
        }
    )

    return {
        "call_id": call.id,
        "caller_id": call.caller_id,
        "receiver_id": call.receiver_id,
        "call_type": call.call_type,
        "status": call.status,
        "started_at": format_utc_iso(call.started_at)
    }

@router.put("/calls/{call_id}")
async def update_call_status(
    call_id: str,
    req: UpdateCallRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    enforce_staff_access(current_user)

    call = db.get(StaffCall, call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call record not found.")

    if current_user.id not in (call.caller_id, call.receiver_id):
        raise HTTPException(status_code=403, detail="Access denied.")

    now = dt.datetime.utcnow()
    call.status = req.status
    if req.status == "connected" and not call.connected_at:
        call.connected_at = now
    elif req.status in ("completed", "ended", "missed", "declined", "busy"):
        call.ended_at = now
        if req.duration_seconds is not None:
            call.duration_seconds = req.duration_seconds
        elif call.connected_at:
            call.duration_seconds = int((now - call.connected_at).total_seconds())

        # Automatically insert call event message in conversation
        try:
            conv = get_or_create_direct_conversation(db, call.caller_id, call.receiver_id)
            if conv:
                type_label = "Video call" if call.call_type == "video" else "Voice call"
                dur_secs = call.duration_seconds or 0
                mins = dur_secs // 60
                secs = dur_secs % 60
                dur_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
                if req.status in ("completed", "ended"):
                    event_text = f"{type_label} ended ({dur_str})"
                elif req.status == "missed":
                    event_text = f"Missed {type_label.lower()}"
                elif req.status == "declined":
                    event_text = f"Declined {type_label.lower()}"
                else:
                    event_text = f"{type_label} ({req.status})"

                call_msg = StaffMessage(
                    id=str(uuid.uuid4()),
                    conversation_id=conv.id,
                    sender_id=call.caller_id,
                    message_type="call_event",
                    content=event_text,
                    created_at=now
                )
                db.add(call_msg)
                conv.updated_at = now
                db.commit()

                # Broadcast new message event
                ser = serialize_message(call_msg, current_user.id, db)
                await staff_ws_manager.broadcast_to_users(
                    [call.caller_id, call.receiver_id],
                    {
                        "type": "new_message",
                        "conversation_id": conv.id,
                        "message": ser
                    }
                )
        except Exception:
            pass

    db.commit()
    db.refresh(call)

    # Relay call state update to other peer
    peer_id = call.receiver_id if current_user.id == call.caller_id else call.caller_id
    await staff_ws_manager.send_to_user(
        peer_id,
        {
            "type": f"call_{req.status}",
            "call_id": call.id,
            "duration_seconds": call.duration_seconds
        }
    )

    return {
        "call_id": call.id,
        "status": call.status,
        "duration_seconds": call.duration_seconds,
        "ended_at": format_utc_iso(call.ended_at)
    }

# ---------------------------------------------------------------------------
# 10. WebSocket Endpoint for Real-time Staff Chat, Presence & WebRTC Signaling
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def staff_chat_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    if not token:
        await websocket.close(code=4001, reason="Authentication token missing")
        return

    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        await websocket.close(code=4001, reason="Invalid token")
        return

    db = SessionLocal()
    user = None
    try:
        user = db.get(User, user_id)
    finally:
        db.close()

    if not user or not user.active or user.role == Role.PATIENT.value:
        await websocket.close(code=4003, reason="Forbidden: Staff only")
        return

    await staff_ws_manager.connect(websocket, user_id)

    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("type")

            # 1. Heartbeat Ping/Pong
            if event_type == "ping":
                await websocket.send_json({"type": "pong", "ts": dt.datetime.utcnow().isoformat()})

            # 2. Typing Indicators
            elif event_type in ("typing_start", "typing_stop"):
                conversation_id = data.get("conversation_id")
                if conversation_id:
                    db = SessionLocal()
                    try:
                        p_ids = db.scalars(
                            select(StaffConversationParticipant.user_id).where(
                                StaffConversationParticipant.conversation_id == conversation_id,
                                StaffConversationParticipant.user_id != user_id
                            )
                        ).all()
                        await staff_ws_manager.broadcast_to_users(
                            p_ids,
                            {
                                "type": event_type,
                                "conversation_id": conversation_id,
                                "user_id": user_id,
                                "user_name": user.name
                            }
                        )
                    finally:
                        db.close()

            # 3. WebRTC Call Signaling (relay offer, answer, ice-candidate, hangup)
            elif event_type in ("call_signal", "webrtc_offer", "webrtc_answer", "webrtc_ice_candidate"):
                target_user_id = data.get("target_user_id") or data.get("to")
                if target_user_id:
                    await staff_ws_manager.send_to_user(
                        target_user_id,
                        {
                            "type": event_type,
                            "from_user_id": user_id,
                            "signal": data.get("signal") or data.get("payload") or data.get("candidate"),
                            "call_id": data.get("call_id")
                        }
                    )

            # 4. Call Control Signaling
            elif event_type in ("call_accept", "call_reject", "call_hangup", "call_busy"):
                target_user_id = data.get("target_user_id") or data.get("to")
                if target_user_id:
                    await staff_ws_manager.send_to_user(
                        target_user_id,
                        {
                            "type": event_type,
                            "from_user_id": user_id,
                            "call_id": data.get("call_id")
                        }
                    )

    except WebSocketDisconnect:
        became_offline = staff_ws_manager.disconnect(websocket, user_id)
        if became_offline:
            db = SessionLocal()
            try:
                u = db.get(User, user_id)
                if u:
                    u.last_seen_at = dt.datetime.utcnow()
                    u.availability_status = "offline"
                    db.commit()
            finally:
                db.close()
            await staff_ws_manager.broadcast_presence(user_id, "offline")
    except Exception as e:
        staff_ws_manager.disconnect(websocket, user_id)
