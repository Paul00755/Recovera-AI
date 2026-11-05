import os
from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
        ],
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATABASE_URL: str = "postgresql+psycopg://recovera:change-me@localhost:5432/recovera"
    JWT_SECRET: str = "recovera-super-secret-jwt-key-2026-production"
    JWT_EXPIRE_MINUTES: int = 1440
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173,https://recovera-ai.pages.dev"
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "https://recovera-ai.pages.dev" if (os.getenv("RENDER") or os.getenv("ENVIRONMENT") == "production") else "http://localhost:5173")
    UPLOAD_DIR: str = "./storage"
    STAFF_REGISTRATION_CODE: str = "RECOVERA-STAFF-2026"

    # OnePlus 10T 5G (Android 15) SMS Gateway & Multi-Endpoint Auto-Fallback
    ONEPLUS_GATEWAY_URL: str = "http://192.168.1.150:8080"
    ONEPLUS_GATEWAY_BACKUP_URL: Optional[str] = None  # Secondary / Tailscale mesh IP (e.g. http://100.x.y.z:8080)
    ONEPLUS_GATEWAY_TOKEN: str = "oneplus-secret-companion-token-2026"
    ONEPLUS_GATEWAY_ENABLED: bool = False
    SMS_RATE_LIMIT_SECONDS: float = 3.0
    SMS_DAILY_LIMIT: int = 500
    SMS_SENDER_NAME: str = "RecoveraAI"

    # Clinical operations
    HOSPITAL_TIMEZONE: str = "Asia/Kolkata"
    CONSULTATION_MAX_MINUTES: int = 15
    APPOINTMENT_DURATION_MINUTES: int = 60

    # Gemini AI Assistant & Clinical Intelligence
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    GEMINI_FALLBACK_MODEL: str = "gemini-flash-lite-latest"

def _resolve_canonical_upload_dir(raw_dir: str) -> str:
    p = Path(raw_dir)
    if p.is_absolute():
        return str(p)
    backend_dir = Path(__file__).resolve().parent.parent.parent
    repo_dir = backend_dir.parent
    repo_storage = (repo_dir / "storage").resolve()
    if repo_storage.exists():
        return str(repo_storage)
    return str((backend_dir / raw_dir).resolve())

settings = Settings()
settings.UPLOAD_DIR = _resolve_canonical_upload_dir(settings.UPLOAD_DIR)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
