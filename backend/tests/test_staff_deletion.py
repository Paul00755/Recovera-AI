import uuid
import datetime as dt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.models.user import User, Role
from app.models.appointment import DoctorAvailability
from app.models.staff_chat import StaffConversation, StaffConversationParticipant, StaffMessage
from app.models.announcement import StaffAnnouncement
from app.core.security import get_password_hash, create_access_token

def test_admin_delete_doctor_with_cascades(client: TestClient, db_session: Session):
    # Retrieve or create admin
    admin = db_session.query(User).filter(User.role == Role.ADMIN.value).first()
    if not admin:
        admin = User(
            email=f"admin_{uuid.uuid4().hex[:6]}@hospital.org",
            password_hash=get_password_hash("AdminPass123!"),
            name="Admin Test",
            role=Role.ADMIN.value,
            active=True
        )
        db_session.add(admin)
        db_session.commit()

    admin_token = create_access_token(data={"sub": admin.id, "role": Role.ADMIN.value})
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Create dummy doctor
    doc_email = f"doc_{uuid.uuid4().hex[:6]}@hospital.org"
    doc = User(
        email=doc_email,
        password_hash=get_password_hash("DocPass123!"),
        name="Dr. To Delete",
        role=Role.DOCTOR.value,
        specialization="Cardiology",
        phone="9988776655",
        active=True
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    doc_id = doc.id

    # Add related entities
    avail = DoctorAvailability(doctor_id=doc_id, start_time=dt.time(9, 0), end_time=dt.time(17, 0))
    db_session.add(avail)

    conv = StaffConversation(conversation_type="direct", created_by=doc_id)
    db_session.add(conv)
    db_session.flush()
    p1 = StaffConversationParticipant(conversation_id=conv.id, user_id=doc_id)
    p2 = StaffConversationParticipant(conversation_id=conv.id, user_id=admin.id)
    db_session.add_all([p1, p2])
    db_session.flush()
    msg = StaffMessage(conversation_id=conv.id, sender_id=doc_id, content="Doctor msg")
    db_session.add(msg)

    ann = StaffAnnouncement(title="Notice", content="Content", author_id=doc_id)
    db_session.add(ann)
    db_session.commit()

    # Call DELETE /api/auth/users/{doc_id} as Admin
    res = client.delete(f"/api/auth/users/{doc_id}", headers=admin_headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "deleted permanently" in data["message"]

    # Verify user record is gone
    db_session.expire_all()
    deleted_user = db_session.get(User, doc_id)
    assert deleted_user is None
