import os
import sys
import argparse
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import select
from app.core.database import SessionLocal
from app.core.security import get_password_hash, validate_password_length
from app.core.audit import record_audit
from app.core.constants import Role
from app.models.user import User

def list_admins(session):
    admins = session.scalars(select(User).where(User.role == Role.ADMIN.value)).all()
    if not admins:
        print("[i] No administrator accounts found in the database.")
        return
    print("\n--- Existing Administrators ---")
    for a in admins:
        status_str = "Active" if a.active else "Inactive"
        print(f" * Email: {a.email} | Name: {a.name} | Status: {status_str}")
    print("-------------------------------\n")

def create_or_update_admin(email: str, password: str, name: str, reset: bool = False):
    email = email.strip().lower()
    password = password.strip()
    name = name.strip()

    validate_password_length(password)

    from app.models import init_db
    init_db()

    session = SessionLocal()
    try:
        existing_user = session.scalar(select(User).where(User.email == email))
        if existing_user:
            if reset:
                existing_user.password_hash = get_password_hash(password)
                existing_user.name = name or existing_user.name
                existing_user.role = Role.ADMIN.value
                existing_user.active = True
                session.commit()
                print(f"[+] Password and administrator role updated successfully for: {email}")
                record_audit(
                    db=session,
                    actor_id=existing_user.id,
                    action="admin_password_reset",
                    target_type="user",
                    target_id=existing_user.id,
                    details={"email": email}
                )
                session.commit()
            else:
                print(f"[!] An account with email '{email}' already exists (Role: {existing_user.role}).")
                print(f"    To reset its password, re-run with: --reset")
                print(f"    Or specify a different email: --email another_admin@hospital.org")
            return

        admin_user = User(
            email=email,
            password_hash=get_password_hash(password),
            name=name,
            role=Role.ADMIN.value,
            active=True
        )
        session.add(admin_user)
        session.flush()

        record_audit(
            db=session,
            actor_id=admin_user.id,
            action="admin_bootstrap_created",
            target_type="user",
            target_id=admin_user.id,
            details={"email": email, "name": name}
        )
        session.commit()
        print(f"[+] Administrator created successfully: {email} (Name: {name})")
    except Exception as e:
        session.rollback()
        print(f"[-] Error creating/updating administrator: {e}")
        sys.exit(1)
    finally:
        session.close()

def main():
    parser = argparse.ArgumentParser(description="Create or manage Recovera AI Administrator accounts.")
    parser.add_argument(
        "--email",
        default=os.getenv("RECOVERA_ADMIN_EMAIL", "admin@hospital.org"),
        help="Admin email address (default: admin@hospital.org or RECOVERA_ADMIN_EMAIL)"
    )
    parser.add_argument(
        "--password",
        default=os.getenv("RECOVERA_ADMIN_PASSWORD", "AdminSecure2026!"),
        help="Admin password (default: AdminSecure2026! or RECOVERA_ADMIN_PASSWORD)"
    )
    parser.add_argument(
        "--name",
        default=os.getenv("RECOVERA_ADMIN_NAME", "Hospital Administrator"),
        help="Admin display name (default: Hospital Administrator or RECOVERA_ADMIN_NAME)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset password if this email already exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all existing administrator accounts"
    )

    args = parser.parse_args()

    session = SessionLocal()
    try:
        if args.list:
            list_admins(session)
            return
    finally:
        session.close()

    create_or_update_admin(
        email=args.email,
        password=args.password,
        name=args.name,
        reset=args.reset
    )

if __name__ == "__main__":
    main()
