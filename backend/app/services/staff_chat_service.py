import uuid
import datetime as dt
from typing import Optional, List, Dict, Any
from sqlalchemy import select, or_, and_, func
from sqlalchemy.orm import Session

from app.models.user import User
from app.models.staff_chat import (
    StaffConversation,
    StaffConversationParticipant,
    StaffMessage,
    StaffMessageDeletion,
    StaffCall,
)
from app.core.constants import Role

def format_utc_iso(dt_val: Optional[dt.datetime]) -> Optional[str]:
    if not dt_val:
        return None
    return dt_val.strftime("%Y-%m-%dT%H:%M:%SZ")

def get_or_create_direct_conversation(db: Session, user1_id: str, user2_id: str) -> StaffConversation:
    """Finds an existing direct 1-on-1 conversation between user1 and user2 or creates a new one."""
    direct_key = f"{min(user1_id, user2_id)}_{max(user1_id, user2_id)}"

    # 1. Fast O(1) indexed lookup by direct_key
    try:
        conv = db.scalar(
            select(StaffConversation).where(
                StaffConversation.direct_key == direct_key,
                StaffConversation.conversation_type == "direct"
            ).limit(1)
        )
        if conv:
            return conv
    except Exception:
        pass

    # 2. Fallback: match by participant pair
    user1_conv_ids = db.scalars(
        select(StaffConversationParticipant.conversation_id)
        .join(StaffConversation, StaffConversation.id == StaffConversationParticipant.conversation_id)
        .where(
            StaffConversationParticipant.user_id == user1_id,
            StaffConversation.conversation_type == "direct"
        )
    ).all()

    if user1_conv_ids:
        # Check which of these has user2 as participant
        matching_conv_id = db.scalar(
            select(StaffConversationParticipant.conversation_id)
            .where(
                StaffConversationParticipant.conversation_id.in_(user1_conv_ids),
                StaffConversationParticipant.user_id == user2_id
            )
            .limit(1)
        )
        if matching_conv_id:
            conv = db.get(StaffConversation, matching_conv_id)
            if conv:
                if not getattr(conv, "direct_key", None):
                    try:
                        conv.direct_key = direct_key
                        db.commit()
                    except Exception:
                        pass
                return conv

    # 3. Create new direct conversation
    now = dt.datetime.utcnow()
    conv = StaffConversation(
        id=str(uuid.uuid4()),
        conversation_type="direct",
        direct_key=direct_key,
        created_by=user1_id,
        created_at=now,
        updated_at=now
    )
    db.add(conv)
    db.flush()

    # Add participants
    p1 = StaffConversationParticipant(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        user_id=user1_id,
        joined_at=now
    )
    p2 = StaffConversationParticipant(
        id=str(uuid.uuid4()),
        conversation_id=conv.id,
        user_id=user2_id,
        joined_at=now
    )
    db.add(p1)
    db.add(p2)
    db.commit()
    db.refresh(conv)
    return conv

def serialize_staff_user(user: User, online_user_ids: Optional[set] = None) -> Dict[str, Any]:
    is_online = (user.id in online_user_ids) if online_user_ids is not None else (user.availability_status == "online")
    status = user.availability_status or ("online" if is_online else "offline")
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "specialization": user.specialization or ("Clinical Staff" if user.role == Role.NURSE.value else "Administration" if user.role == Role.ADMIN.value else "General Medicine"),
        "phone": user.phone or "",
        "avatar_url": getattr(user, "avatar_url", None),
        "about": getattr(user, "about", None) or "Available for patient care & team coordination.",
        "availability_status": status,
        "is_online": is_online,
        "last_seen_at": format_utc_iso(getattr(user, "last_seen_at", None) or user.created_at)
    }

def serialize_message(m: StaffMessage, current_user_id: str, db: Session) -> Dict[str, Any]:
    content = m.content
    voice_url = m.voice_url
    if m.is_deleted_for_everyone:
        content = "🚫 This message was deleted"
        voice_url = None

    reply_dict = None
    if m.reply_to_id:
        parent = getattr(m, "reply_to", None) or db.get(StaffMessage, m.reply_to_id)
        if parent:
            p_sender = getattr(parent, "sender", None)
            p_content = "🚫 This message was deleted" if parent.is_deleted_for_everyone else parent.content
            if parent.message_type == "voice" and not parent.is_deleted_for_everyone:
                p_content = "🎤 Voice message"
            reply_dict = {
                "id": parent.id,
                "sender_id": parent.sender_id,
                "sender_name": p_sender.name if p_sender else "Staff Member",
                "content": p_content,
                "message_type": parent.message_type
            }

    sender = getattr(m, "sender", None)
    return {
        "id": m.id,
        "conversation_id": m.conversation_id,
        "sender_id": m.sender_id,
        "sender_name": sender.name if sender else "Staff Member",
        "sender_role": sender.role if sender else "staff",
        "sender_avatar": getattr(sender, "avatar_url", None) if sender else None,
        "message_type": m.message_type,
        "content": content,
        "voice_url": voice_url,
        "voice_duration": m.voice_duration,
        "reply_to": reply_dict,
        "is_edited": m.is_edited,
        "edited_at": format_utc_iso(m.edited_at),
        "is_deleted_for_everyone": m.is_deleted_for_everyone,
        "is_outgoing": m.sender_id == current_user_id,
        "created_at": format_utc_iso(m.created_at),
    }

def serialize_conversation_summary(
    conv: StaffConversation,
    current_user_id: str,
    db: Session,
    online_user_ids: Optional[set] = None
) -> Dict[str, Any]:
    # Find current user's participant record
    my_part = None
    other_user = None

    for p in conv.participants:
        if p.user_id == current_user_id:
            my_part = p
        else:
            if conv.conversation_type == "direct" and other_user is None:
                other_user = getattr(p, "user", None) or db.get(User, p.user_id)

    # In case other_user is still None in direct conversation (e.g. participant relation wasn't loaded)
    if conv.conversation_type == "direct" and other_user is None:
        if getattr(conv, "direct_key", None):
            parts = conv.direct_key.split("_")
            if len(parts) == 2:
                peer_id = parts[1] if parts[0] == current_user_id else parts[0]
                other_user = db.get(User, peer_id)
        if other_user is None:
            peer_part = db.scalar(
                select(StaffConversationParticipant)
                .where(
                    StaffConversationParticipant.conversation_id == conv.id,
                    StaffConversationParticipant.user_id != current_user_id
                ).limit(1)
            )
            if peer_part:
                other_user = getattr(peer_part, "user", None) or db.get(User, peer_part.user_id)

    # Compute unread count
    cleared_at = my_part.cleared_at if my_part else None
    last_read_at = my_part.last_read_at if my_part else None

    # Get last active message (not deleted for me)
    deleted_msg_ids = set()
    if my_part:
        deleted_msg_ids = set(
            db.scalars(
                select(StaffMessageDeletion.message_id)
                .where(StaffMessageDeletion.user_id == current_user_id)
            ).all()
        )

    msg_query = select(StaffMessage).where(StaffMessage.conversation_id == conv.id)
    if cleared_at:
        msg_query = msg_query.where(StaffMessage.created_at > cleared_at)
    if deleted_msg_ids:
        msg_query = msg_query.where(StaffMessage.id.not_in(deleted_msg_ids))

    from sqlalchemy.orm import joinedload
    last_msg = db.scalar(
        msg_query.options(joinedload(StaffMessage.sender))
        .order_by(StaffMessage.created_at.desc())
        .limit(1)
    )

    # Count unread messages
    unread_count = 0
    if last_read_at:
        unread_count = db.scalar(select(func.count(StaffMessage.id)).where(
            StaffMessage.conversation_id == conv.id,
            StaffMessage.created_at > last_read_at,
            StaffMessage.sender_id != current_user_id,
            StaffMessage.is_deleted_for_everyone == False
        )) or 0
    else:
        unread_count = db.scalar(select(func.count(StaffMessage.id)).where(
            StaffMessage.conversation_id == conv.id,
            StaffMessage.sender_id != current_user_id,
            StaffMessage.is_deleted_for_everyone == False
        )) or 0

    last_msg_dict = None
    if last_msg:
        sender_name = last_msg.sender.name if getattr(last_msg, "sender", None) else "Staff"
        preview = last_msg.content
        if last_msg.is_deleted_for_everyone:
            preview = "🚫 This message was deleted"
        elif last_msg.message_type == "voice":
            preview = f"🎤 Voice message ({round(last_msg.voice_duration or 0)}s)"
        elif last_msg.message_type == "call_event":
            preview = f"📞 {last_msg.content}"

        last_msg_dict = {
            "id": last_msg.id,
            "sender_id": last_msg.sender_id,
            "sender_name": sender_name,
            "message_type": last_msg.message_type,
            "content": preview,
            "created_at": format_utc_iso(last_msg.created_at),
            "is_outgoing": last_msg.sender_id == current_user_id
        }

    title = conv.title
    icon_url = conv.icon_url
    other_user_dict = None

    if conv.conversation_type == "direct" and other_user:
        other_user_dict = serialize_staff_user(other_user, online_user_ids)
        title = other_user.name
        icon_url = getattr(other_user, "avatar_url", None)

    return {
        "id": conv.id,
        "conversation_type": conv.conversation_type,
        "is_group": conv.conversation_type == "group",
        "title": title or (other_user.name if other_user else "Staff Member"),
        "icon_url": icon_url,
        "description": conv.description,
        "other_user": other_user_dict,
        "other_participant": other_user_dict,
        "participants_count": len(conv.participants),
        "is_muted": bool(my_part and my_part.is_muted),
        "is_archived": bool(my_part and my_part.is_archived),
        "unread_count": unread_count,
        "last_message": last_msg_dict,
        "updated_at": format_utc_iso(conv.updated_at)
    }
