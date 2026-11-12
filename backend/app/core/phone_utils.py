import re
from typing import Optional
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from app.models.user import User
from app.models.patient import Patient


def normalize_phone(phone: Optional[str]) -> Optional[str]:
    """Cleans spaces, hyphens, and parenthesis from phone string."""
    if not phone:
        return None
    cleaned = re.sub(r"[\s\-\(\)]", "", phone.strip())
    return cleaned if cleaned else None


def get_phone_key(phone: Optional[str]) -> Optional[str]:
    """
    Extracts canonical digits for phone comparison.
    If phone has >= 10 digits, returns the last 10 digits (to catch +91 vs 0 vs raw).
    """
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone.strip())
    if len(digits) >= 10:
        return digits[-10:]
    return digits if digits else None


def assert_phone_number_unique(
    db: Session,
    phone: Optional[str],
    exclude_user_id: Optional[str] = None,
    exclude_patient_id: Optional[str] = None,
) -> None:
    """
    Enforces that a phone number cannot be shared across multiple accounts.
    A phone number must be unique across all Users and Patients.
    Excludes the current user/patient record when updating existing accounts.
    """
    norm = normalize_phone(phone)
    if not norm:
        return

    target_key = get_phone_key(norm)
    if not target_key:
        return

    # If an exclude_patient_id was provided, also exclude its corresponding user_id
    if exclude_patient_id and not exclude_user_id:
        pat = db.get(Patient, exclude_patient_id)
        if pat and pat.user_id:
            exclude_user_id = str(pat.user_id)

    # If an exclude_user_id was provided, also exclude its corresponding patient_id
    if exclude_user_id and not exclude_patient_id:
        pat = db.scalar(select(Patient).where(Patient.user_id == str(exclude_user_id)))
        if pat:
            exclude_patient_id = str(pat.id)

    # 1. Check all Users with a phone number
    users = db.scalars(select(User).where(User.phone.isnot(None))).all()
    for u in users:
        if exclude_user_id and str(u.id) == str(exclude_user_id):
            continue
        u_key = get_phone_key(u.phone)
        if u_key and u_key == target_key:
            role_title = (u.role or "user").capitalize()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The phone number '{phone}' is already registered to another account ({u.name} – {role_title}). A phone number must be unique to a single account.",
            )

    # 2. Check all Patients with a phone number
    patients = db.scalars(select(Patient).where(Patient.phone.isnot(None))).all()
    for p in patients:
        if exclude_patient_id and str(p.id) == str(exclude_patient_id):
            continue
        p_key = get_phone_key(p.phone)
        if p_key and p_key == target_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The phone number '{phone}' is already registered to patient {p.patient_code or p.name}. A phone number must be unique to a single account.",
            )
