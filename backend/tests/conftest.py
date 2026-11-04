# -*- coding: utf-8 -*-
"""
Pytest configuration and test fixtures for Recovera AI backend.
"""
import sys
import os
import pytest
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.core.database import SessionLocal, engine, Base
from app.core.security import create_access_token
from app.models.user import User
from app.core.constants import Role

@pytest.fixture(scope="session")
def db_session():
    """Provides a database session for testing."""
    session = SessionLocal()
    yield session
    session.close()

@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient for API integration testing."""
    return TestClient(app)

@pytest.fixture
def admin_headers(db_session):
    """Generates an authorization header for an admin user."""
    admin = db_session.query(User).filter(User.role == Role.ADMIN.value, User.active == True).first()
    if not admin:
        pytest.skip("No active admin user found in database.")
    token = create_access_token(data={"sub": admin.id, "role": admin.role})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def doctor_headers(db_session):
    """Generates an authorization header for a doctor user."""
    doctor = db_session.query(User).filter(User.role == Role.DOCTOR.value, User.active == True).first()
    if not doctor:
        pytest.skip("No active doctor user found in database.")
    token = create_access_token(data={"sub": doctor.id, "role": doctor.role})
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def nurse_headers(db_session):
    """Generates an authorization header for a nurse user."""
    nurse = db_session.query(User).filter(User.role == Role.NURSE.value, User.active == True).first()
    if not nurse:
        pytest.skip("No active nurse user found in database.")
    token = create_access_token(data={"sub": nurse.id, "role": nurse.role})
    return {"Authorization": f"Bearer {token}"}
