import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import text, inspect
from app.core.database import engine

def migrate_database():
    print("[*] Running Version 3.0 migration on PostgreSQL...")
    with engine.begin() as conn:
        # 1. Purge WhatsApp tables
        conn.execute(text("DROP TABLE IF EXISTS whatsapp_notifications CASCADE;"))
        conn.execute(text("DROP TABLE IF EXISTS whatsapp_contacts CASCADE;"))
        print("[+] Purged legacy WhatsApp tables.")

        # 2. Inspect patients table and add Version 3.0 columns
        patient_cols = {c["name"] for c in inspect(engine).get_columns("patients")}
        
        if "patient_code" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN patient_code VARCHAR(40) NULL;"))
            print("[+] Added patients.patient_code")
        if "sms_opt_in" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN sms_opt_in BOOLEAN NOT NULL DEFAULT FALSE;"))
            # Copy over whatsapp_opt_in if present
            if "whatsapp_opt_in" in patient_cols:
                conn.execute(text("UPDATE patients SET sms_opt_in = whatsapp_opt_in;"))
            print("[+] Added patients.sms_opt_in")
        if "sms_opt_in_at" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN sms_opt_in_at TIMESTAMP NULL;"))
            print("[+] Added patients.sms_opt_in_at")
        if "sms_opt_in_recorded_by" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN sms_opt_in_recorded_by VARCHAR(36) NULL;"))
            print("[+] Added patients.sms_opt_in_recorded_by")
        if "previous_history" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN previous_history BOOLEAN NOT NULL DEFAULT FALSE;"))
            print("[+] Added patients.previous_history")
        if "history" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN history JSON NULL;"))
            print("[+] Added patients.history")
        if "active" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE;"))
            print("[+] Added patients.active")
        if "created_by" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN created_by VARCHAR(36) NULL;"))
            print("[+] Added patients.created_by")

        # Ensure patient_code is populated for existing patients
        conn.execute(text("""
            UPDATE patients 
            SET patient_code = 'P' || TO_CHAR(created_at, 'YYMMDD') || UPPER(SUBSTRING(id::text, 1, 4))
            WHERE patient_code IS NULL;
        """))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS ix_patients_patient_code ON patients (patient_code);"))
        print("[+] Populated unique patient codes for existing patients.")

        # 3. Migrate recovery_plans and medication_schedules columns
        conn.execute(text("ALTER TABLE recovery_plans ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1;"))
        conn.execute(text("ALTER TABLE recovery_plans ADD COLUMN IF NOT EXISTS instructions JSON NULL;"))
        conn.execute(text("ALTER TABLE recovery_plans ADD COLUMN IF NOT EXISTS warning_signs JSON NULL;"))
        conn.execute(text("ALTER TABLE recovery_plans ADD COLUMN IF NOT EXISTS source_versions JSON NULL;"))
        conn.execute(text("ALTER TABLE medication_schedules ADD COLUMN IF NOT EXISTS medication_id VARCHAR(36) REFERENCES medications(id) ON DELETE CASCADE;"))
        conn.execute(text("ALTER TABLE medication_schedules ADD COLUMN IF NOT EXISTS label VARCHAR(80);"))
        conn.execute(text("ALTER TABLE medication_schedules ADD COLUMN IF NOT EXISTS time_hhmm VARCHAR(5);"))
        conn.execute(text("ALTER TABLE medication_schedules ADD COLUMN IF NOT EXISTS food_relation VARCHAR(50);"))
        conn.execute(text("ALTER TABLE medication_schedules ALTER COLUMN patient_id DROP NOT NULL;"))
        conn.execute(text("ALTER TABLE medication_schedules ALTER COLUMN plan_id DROP NOT NULL;"))
        conn.execute(text("ALTER TABLE medication_schedules ALTER COLUMN medication_name DROP NOT NULL;"))
        conn.execute(text("ALTER TABLE medication_schedules ALTER COLUMN time_of_day DROP NOT NULL;"))
        conn.execute(text("ALTER TABLE medication_schedules ALTER COLUMN active DROP NOT NULL;"))
        conn.execute(text("ALTER TABLE medication_schedules ALTER COLUMN created_at DROP NOT NULL;"))
        conn.execute(text("ALTER TABLE medication_schedules ALTER COLUMN updated_at DROP NOT NULL;"))

        # 4. Create any missing new tables
        from app.models import init_db
        init_db()
        print("[+] All Version 3.0 models verified and synchronized.")

if __name__ == "__main__":
    migrate_database()
