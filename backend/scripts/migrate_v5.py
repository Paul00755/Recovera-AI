"""
Recovera AI – Database Migration v5 (Phase 7 - Handwritten ML & Patient Deletion Tickets)
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import text
from app.core.database import engine, Base
import app.models  # load all models

def migrate_database() -> None:
    print(f"[*] Running Recovera AI v5 migration on {engine.dialect.name.upper()} …")
    from app.models import init_db
    init_db()

    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS patient_deletion_tickets (
                    id VARCHAR(36) PRIMARY KEY,
                    patient_id VARCHAR(36) NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                    requested_by VARCHAR(36) NOT NULL REFERENCES users(id),
                    reason TEXT NOT NULL,
                    status VARCHAR(20) DEFAULT 'pending',
                    admin_notes TEXT,
                    reviewed_by VARCHAR(36) REFERENCES users(id),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TIMESTAMP
                );
            """))
            print("[+] patient_deletion_tickets ensured.")
        else:
            print("[+] SQLite schema initialized via init_db().")

    print("[+] Migration v5 completed successfully.")

if __name__ == "__main__":
    migrate_database()
