# -*- coding: utf-8 -*-
"""
Tests for Role-Based Access Control (RBAC) and Admin Security Constraints:
1. An administrator cannot delete another administrator account.
2. An administrator cannot delete their own account.
3. Non-administrators (doctors, nurses, patients) are forbidden from deleting accounts.
"""
import uuid
import pytest
from app.models.user import User
from app.core.constants import Role
from app.core.security import get_password_hash, create_access_token

def test_admin_cannot_delete_another_admin(client, db_session, admin_headers):
    # 1. Create a second admin account in database
    target_admin = User(
        name="Target Admin To Protect",
        email=f"target_admin_{uuid.uuid4().hex[:6]}@hospital.local",
        password_hash=get_password_hash("AdminSecurePass123!"),
        role=Role.ADMIN.value,
        active=True
    )
    db_session.add(target_admin)
    db_session.commit()
    db_session.refresh(target_admin)

    try:
        # 2. Attempt to delete target admin using admin_headers
        resp = client.delete(f"/api/auth/users/{target_admin.id}", headers=admin_headers)
        assert resp.status_code == 403
        data = resp.json()
        assert "cannot delete another administrator account" in data["detail"].lower()

        # 3. Verify target admin was NOT deleted
        db_admin = db_session.get(User, target_admin.id)
        assert db_admin is not None
        assert db_admin.active == True
    finally:
        # Cleanup
        db_session.delete(target_admin)
        db_session.commit()

def test_admin_cannot_delete_self(client, db_session, admin_headers):
    # Decode token to get current admin ID
    token = admin_headers["Authorization"].split(" ")[1]
    from app.core.security import decode_access_token
    payload = decode_access_token(token)
    admin_id = payload["sub"]

    resp = client.delete(f"/api/auth/users/{admin_id}", headers=admin_headers)
    assert resp.status_code == 400
    assert "cannot delete your own administrator account" in resp.json()["detail"].lower()

def test_nurse_forbidden_from_deleting_users(client, db_session, nurse_headers):
    # Create dummy user
    dummy = User(
        name="Test Patient User",
        email=f"dummy_{uuid.uuid4().hex[:6]}@hospital.local",
        password_hash=get_password_hash("Pass123!"),
        role=Role.PATIENT.value,
        active=True
    )
    db_session.add(dummy)
    db_session.commit()
    db_session.refresh(dummy)

    try:
        resp = client.delete(f"/api/auth/users/{dummy.id}", headers=nurse_headers)
        assert resp.status_code == 403
    finally:
        db_session.delete(dummy)
        db_session.commit()
