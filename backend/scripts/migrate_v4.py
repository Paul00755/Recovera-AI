"""
Recovera AI – Database Migration v4 (Phase 6)

Idempotent: safe to run multiple times.

Changes applied:
  1. ALTER TABLE users  – add specialization, availability_status, away_until
  2. ALTER TABLE patients – add assigned_doctor_id (FK → users.id ON DELETE SET NULL)
  3. init_db() – create any new tables (appointments, consultations, etc.)
"""
import sys
from pathlib import Path

# Ensure the backend package is importable when running directly
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import text
from app.core.database import engine


def migrate_database() -> None:
    print(f"[*] Running Recovera AI v4 migration (Phase 6) on {engine.dialect.name.upper()} …")

    from app.models import init_db
    init_db()

    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS specialization VARCHAR(80);"
            ))
            print("[+] users.specialization ensured.")

            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS availability_status VARCHAR(20) DEFAULT 'online';"
            ))
            print("[+] users.availability_status ensured.")

            conn.execute(text(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS away_until TIMESTAMP;"
            ))
            print("[+] users.away_until ensured.")

            conn.execute(text(
                """
                ALTER TABLE patients
                ADD COLUMN IF NOT EXISTS assigned_doctor_id VARCHAR(36)
                REFERENCES users(id) ON DELETE SET NULL;
                """
            ))
            print("[+] patients.assigned_doctor_id ensured.")
        else:
            print("[+] SQLite schema initialized via ORM metadata.")

    print("[+] Migration v4 completed successfully.")


if __name__ == "__main__":
    migrate_database()
