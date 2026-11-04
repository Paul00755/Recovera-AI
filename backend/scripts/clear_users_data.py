import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import text
from app.core.database import SessionLocal, engine
from app.core.security import get_password_hash
from app.core.constants import Role
from app.models.user import User
from app.models import init_db

def clear_data():
    print("[*] Initializing and preparing database wipe...")
    init_db()
    
    tables_to_clear = [
        "notifications",
        "audit_logs",
        "sms_jobs",
        "consultation_signals",
        "consultation_messages",
        "consultation_participants",
        "consultations",
        "appointments",
        "doctor_availability",
        "daily_checkins",
        "medication_adherence",
        "medication_occurrences",
        "medication_schedules",
        "medications",
        "risk_classification_history",
        "recovery_plans",
        "ocr_extractions",
        "clinical_documents",
        "medical_history",
        "patient_deletion_tickets",
        "patient_doctor_assignments",
        "patients",
        "password_reset_tickets",
        "users"
    ]
    
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            conn.execute(text("PRAGMA foreign_keys = OFF;"))
            for t in tables_to_clear:
                try:
                    conn.execute(text(f"DELETE FROM {t};"))
                    print(f"  [-] Cleared table: {t}")
                except Exception as e:
                    print(f"  [!] Skipped {t}: {e}")
            conn.execute(text("PRAGMA foreign_keys = ON;"))
        else:
            for t in tables_to_clear:
                try:
                    conn.execute(text(f"TRUNCATE TABLE {t} CASCADE;"))
                    print(f"  [-] Truncated table: {t}")
                except Exception as e:
                    print(f"  [!] Skipped {t}: {e}")

    # Clean storage uploads
    storage_dir = Path("./storage")
    if storage_dir.exists():
        for f in storage_dir.glob("*"):
            if f.is_file():
                try:
                    f.unlink()
                except Exception:
                    pass
        print("  [-] Cleared uploaded physical clinical documents in ./storage")

    # Re-seed administrator accounts
    session = SessionLocal()
    try:
        admin1 = User(
            email="admin@hospital.org",
            password_hash=get_password_hash("AdminSecure2026!"),
            name="Hospital Administrator",
            role=Role.ADMIN.value,
            phone="+919876543210",
            active=True,
            availability_status="online"
        )
        admin2 = User(
            email="paul123@gmail.com",
            password_hash=get_password_hash("AdminSecure2026!"),
            name="Paul Administrator",
            role=Role.ADMIN.value,
            phone="+919876543210",
            active=True,
            availability_status="online"
        )
        session.add(admin1)
        session.add(admin2)
        session.commit()
        print(f"[+] Bootstrap Administrators created:")
        print(f"    1. admin@hospital.org (Password: AdminSecure2026!, Phone: +919876543210)")
        print(f"    2. paul123@gmail.com (Password: AdminSecure2026!, Phone: +919876543210)")
        print("[OK] Complete fresh reset accomplished. Database is completely clean and ready.")
    finally:
        session.close()

if __name__ == "__main__":
    clear_data()
