import datetime as dt
import logging
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.user import User
from app.models.staff_chat import (
    StaffConversation,
    StaffConversationParticipant,
    StaffMessage,
    StaffCall,
)
from app.models.announcement import StaffAnnouncement, StaffAnnouncementRead
from app.core.constants import Role
from app.core.security import get_password_hash
from app.services.staff_chat_service import get_or_create_direct_conversation

logger = logging.getLogger("recovera.seed_staff")

DEFAULT_PASSWORD = "Password123!"

STAFF_PROFILES = []

def seed_staff_system_data(db: Session):
    """Seed is disabled in production to preserve authentic hospital accounts."""
    return {}
    {
        "email": "arjun.menon@hospital.org",
        "name": "Dr. Arjun Menon",
        "role": Role.DOCTOR.value,
        "specialization": "Neurology • Doctor",
        "phone": "+91 98765 43212",
        "about": "Okay, I will check and update.",
        "avatar_url": "https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=400&auto=format&fit=crop&q=80",
        "availability_status": "online",
    },
    {
        "email": "neha.kapoor@hospital.org",
        "name": "Dr. Neha Kapoor",
        "role": Role.DOCTOR.value,
        "specialization": "Radiology • Doctor",
        "phone": "+91 98765 43213",
        "about": "Diagnostic imaging specialist. Diagnostic CT/MRI reviews.",
        "avatar_url": "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=400&auto=format&fit=crop&q=80",
        "availability_status": "away",
    },
    {
        "email": "vikram.reddy@hospital.org",
        "name": "Dr. Vikram Reddy",
        "role": Role.DOCTOR.value,
        "specialization": "Orthopedics • Doctor",
        "phone": "+91 98765 43214",
        "about": "Joint reconstruction and post-op fracture rehabilitation.",
        "avatar_url": "https://images.unsplash.com/photo-1537368910025-700350fe46c7?w=400&auto=format&fit=crop&q=80",
        "availability_status": "busy",
    },
    {
        "email": "latha.nurse@hospital.org",
        "name": "Nurse Latha",
        "role": Role.NURSE.value,
        "specialization": "Clinical Staff • Nurse",
        "phone": "+91 98765 43215",
        "about": "Thank you for the update! Ward rounds underway.",
        "avatar_url": "https://images.unsplash.com/photo-1576091160550-2173dba999ef?w=400&auto=format&fit=crop&q=80",
        "availability_status": "online",
    },
]

ANNOUNCEMENTS = [
    {
        "title": "Fire Safety Drill - 2 Nov 2026",
        "category": "emergency",
        "priority": "emergency",
        "content": "Mandatory for all staff. Please assemble at main lobby at 10 AM. Ward shift leads must verify roster muster points before proceeding.",
        "image_url": "https://images.unsplash.com/photo-1581092160607-ee22621dd758?w=500&auto=format&fit=crop&q=80",
    },
    {
        "title": "New Laboratory System Deployment",
        "category": "general",
        "priority": "important",
        "content": "The new lab information system will be live from 5 Nov 2026. Electronic pathology requisitions and automated telemetry notifications will be integrated directly.",
        "image_url": "https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=500&auto=format&fit=crop&q=80",
    },
    {
        "title": "CPR Refresher Training",
        "category": "training",
        "priority": "normal",
        "content": "Scheduled for 3 Nov 2026 at Conference Hall B. Hands-on pediatric and adult CPR recertification with certified clinical instructors.",
        "image_url": "https://images.unsplash.com/photo-1579684385127-1ef15d508118?w=500&auto=format&fit=crop&q=80",
    },
    {
        "title": "Welcome to Our New Team Member - Dr. Arjun Menon",
        "category": "new_staff",
        "priority": "important",
        "content": "We are happy to welcome Dr. Arjun Menon (Neurology) to our hospital clinical team. Please welcome him to Recovera AI!",
        "image_url": "https://images.unsplash.com/photo-1622253692010-333f2da6031d?w=500&auto=format&fit=crop&q=80",
        "is_auto_generated": True,
    },
]


def seed_staff_system_data(db: Session):
    """
    Populates staff users, realistic conversations, voice messages, and announcements
    to match the reference WhatsApp staff portal design.
    """
    # 1. Create or update staff users
    created_staff = {}
    for prof in STAFF_PROFILES:
        existing = db.scalar(select(User).where(User.email == prof["email"]))
        if not existing:
            u = User(
                email=prof["email"],
                password_hash=get_password_hash(DEFAULT_PASSWORD),
                name=prof["name"],
                role=prof["role"],
                active=True,
                phone=prof["phone"],
                specialization=prof["specialization"],
                about=prof["about"],
                avatar_url=prof["avatar_url"],
                availability_status=prof["availability_status"],
                last_seen_at=dt.datetime.utcnow(),
            )
            db.add(u)
            db.flush()
            created_staff[prof["email"]] = u
            logger.info(f"Seeded staff user: {u.name} ({u.email})")
        else:
            if not existing.avatar_url:
                existing.avatar_url = prof["avatar_url"]
            if not existing.about:
                existing.about = prof["about"]
            if not existing.specialization:
                existing.specialization = prof["specialization"]
            if not existing.phone:
                existing.phone = prof["phone"]
            created_staff[prof["email"]] = existing

    # Find an admin user
    admin_user = db.scalar(select(User).where(User.email == "admin@hospital.org"))
    if not admin_user:
        admin_user = db.scalar(select(User).where(User.role == Role.ADMIN.value, User.active == True).limit(1))
    if not admin_user:
        admin_user = User(
            email="admin@recovera.ai",
            password_hash=get_password_hash(DEFAULT_PASSWORD),
            name="Hospital Administrator",
            role=Role.ADMIN.value,
            active=True,
            phone="+91 99999 00001",
            specialization="Administration",
            about="Hospital Administration & Clinical Operations",
            avatar_url="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400&auto=format&fit=crop&q=80",
            availability_status="online",
        )
        db.add(admin_user)
        db.flush()

    # Find doctor Sam Wilson or primary doctor
    sam_user = db.scalar(select(User).where(User.email.like("%sam%")).limit(1))
    if not sam_user:
        sam_user = db.scalar(select(User).where(User.role == Role.DOCTOR.value).limit(1))

    # 2. Seed Announcements
    for ann_data in ANNOUNCEMENTS:
        existing_ann = db.scalar(select(StaffAnnouncement).where(StaffAnnouncement.title == ann_data["title"]))
        if not existing_ann:
            ann = StaffAnnouncement(
                title=ann_data["title"],
                content=ann_data["content"],
                category=ann_data["category"],
                priority=ann_data["priority"],
                image_url=ann_data["image_url"],
                author_id=admin_user.id,
                author_name="Admin",
                author_role=admin_user.role,
                is_auto_generated=ann_data.get("is_auto_generated", False),
                created_at=dt.datetime.utcnow() - dt.timedelta(days=ANNOUNCEMENTS.index(ann_data)),
                updated_at=dt.datetime.utcnow(),
            )
            db.add(ann)
            db.flush()
            logger.info(f"Seeded announcement: {ann.title}")

    # 3. Seed Reference Chat Conversations
    # Pair: Primary Staff (Doctor or Admin) with Dr. Priya Sharma
    priya = created_staff.get("priya.sharma@hospital.org")
    primary_user = sam_user or admin_user

    if priya and primary_user and priya.id != primary_user.id:
        conv = get_or_create_direct_conversation(db, primary_user.id, priya.id)

        # Only seed messages if conversation has no messages
        existing_msg_count = db.scalar(
            select(StaffMessage.id).where(StaffMessage.conversation_id == conv.id).limit(1)
        )
        if not existing_msg_count:
            # Seed transcript matching screenshot
            t_base_old = dt.datetime.utcnow() - dt.timedelta(days=8)
            t_today = dt.datetime.utcnow().replace(hour=8, minute=1, second=0)

            sample_msgs = [
                # 23 October 2026
                (priya.id, "Hi Samuel, Can you share the patient's latest vitals from Ward 5?", t_base_old),
                (primary_user.id, "Sure, please find the latest vitals below.", t_base_old + dt.timedelta(minutes=1)),
                # Today
                (priya.id, "All the best da", t_today),
                (priya.id, "Exam nalla pannitu vaa", t_today + dt.timedelta(seconds=20)),
                (primary_user.id, "Mmokk daa, nandrigal ✌️", t_today + dt.timedelta(minutes=44)),
                (priya.id, "Inaki afternoon sapda veliya povom ah da?", t_today + dt.timedelta(hours=2, minutes=43)),
                (priya.id, "Madurai ku vantayaa da", t_today + dt.timedelta(hours=3, minutes=1)),
                (priya.id, "Enga polam solluuu", t_today + dt.timedelta(hours=3, minutes=1, seconds=30)),
                (primary_user.id, "Nan madurai la dhane da iruken 😅", t_today + dt.timedelta(hours=3, minutes=1, seconds=50)),
                (primary_user.id, "Diwali varaikum inga dhan irupenu sonene", t_today + dt.timedelta(hours=3, minutes=1, seconds=55)),
                (priya.id, "Anney vettuku poirupa nu nenachen da", t_today + dt.timedelta(hours=3, minutes=2)),
                (priya.id, "Thala diwali la", t_today + dt.timedelta(hours=3, minutes=2, seconds=10)),
                (priya.id, "Cheri enga polam", t_today + dt.timedelta(hours=3, minutes=2, seconds=20)),
            ]

            for sender_id, text, sent_at in sample_msgs:
                msg = StaffMessage(
                    conversation_id=conv.id,
                    sender_id=sender_id,
                    message_type="text",
                    content=text,
                    created_at=sent_at,
                )
                db.add(msg)
                conv.updated_at = sent_at

    # 4. Seed Group: General Medicine Team
    group_conv = db.scalar(
        select(StaffConversation).where(
            StaffConversation.conversation_type == "group",
            StaffConversation.title == "General Medicine Team"
        )
    )
    if not group_conv and primary_user:
        group_conv = StaffConversation(
            title="General Medicine Team",
            description="Interdisciplinary clinical updates and morning rounds coordination.",
            conversation_type="group",
            created_by=primary_user.id,
            icon_url="https://images.unsplash.com/photo-1538108149393-fbbd81895907?w=400&auto=format&fit=crop&q=80",
            created_at=dt.datetime.utcnow() - dt.timedelta(days=10),
            updated_at=dt.datetime.utcnow(),
        )
        db.add(group_conv)
        db.flush()

        # Add participants
        members = [primary_user, admin_user] + [s for s in created_staff.values()]
        for m in members:
            db.add(StaffConversationParticipant(conversation_id=group_conv.id, user_id=m.id))
        db.flush()

        # Add initial team message
        g_msg = StaffMessage(
            conversation_id=group_conv.id,
            sender_id=primary_user.id,
            message_type="text",
            content="Morning rounds at 11 AM in ICU block. Please review triage board.",
            created_at=dt.datetime.utcnow() - dt.timedelta(hours=4),
        )
        db.add(g_msg)
        group_conv.updated_at = g_msg.created_at

    # 5. Seed direct chat between Nurse Kavitha and primary user
    kavitha = created_staff.get("kavitha.icu@hospital.org")
    if kavitha and primary_user and kavitha.id != primary_user.id:
        conv_k = get_or_create_direct_conversation(db, primary_user.id, kavitha.id)
        existing_k_msgs = db.scalar(select(StaffMessage.id).where(StaffMessage.conversation_id == conv_k.id).limit(1))
        if not existing_k_msgs:
            db.add(StaffMessage(
                conversation_id=conv_k.id,
                sender_id=kavitha.id,
                message_type="text",
                content="Patient in Bed 12 has completed their recovery exercise.",
                created_at=dt.datetime.utcnow() - dt.timedelta(minutes=25),
            ))
            conv_k.updated_at = dt.datetime.utcnow() - dt.timedelta(minutes=25)

    db.commit()
    logger.info("Staff system seeding completed successfully.")
