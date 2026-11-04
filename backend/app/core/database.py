import socket
import logging
from urllib.parse import urlparse
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import settings

logger = logging.getLogger("recovera.database")

class Base(DeclarativeBase):
    pass

def _get_canonical_sqlite_url() -> str:
    from pathlib import Path
    backend_dir = Path(__file__).resolve().parent.parent.parent
    repo_dir = backend_dir.parent
    repo_db = (repo_dir / "recovera.db").resolve()
    backend_db = (backend_dir / "recovera.db").resolve()
    if repo_db.exists() and repo_db.stat().st_size > 0:
        return f"sqlite:///{repo_db.as_posix()}"
    return f"sqlite:///{backend_db.as_posix()}"

def _resolve_engine():
    db_url = settings.DATABASE_URL.strip()
    
    # Normalize Postgres URL scheme for psycopg v3 dialect
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    if db_url.startswith("postgresql+psycopg"):
        try:
            parsed = urlparse(db_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 5432
            is_local = host in ("localhost", "127.0.0.1", "0.0.0.0", "postgres")
            
            # Short probe for local container/host, longer for remote cloud (e.g. Neon, AWS RDS)
            probe_timeout = 1.5 if is_local else 8.0
            with socket.create_connection((host, port), timeout=probe_timeout):
                pass
            return create_engine(
                db_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                pool_recycle=300,
                pool_timeout=30
            )
        except OSError as e:
            if not is_local:
                logger.warning(
                    "Remote PostgreSQL socket probe to %s:%s timed out (%s). Retrying directly via SQLAlchemy pool_pre_ping.",
                    host, port, e
                )
                return create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_size=10,
                    max_overflow=20,
                    pool_recycle=300,
                    pool_timeout=30
                )

            sqlite_url = _get_canonical_sqlite_url()
            logger.warning(
                "Local PostgreSQL instance at %s:%s is unreachable. Falling back to canonical local SQLite database (%s) for zero-error execution.",
                host, port, sqlite_url
            )
            return create_engine(
                sqlite_url,
                connect_args={"check_same_thread": False}
            )
    if db_url.startswith("sqlite"):
        if db_url.startswith("sqlite:///."):
            sqlite_url = _get_canonical_sqlite_url()
            return create_engine(
                sqlite_url,
                connect_args={"check_same_thread": False}
            )
        return create_engine(
            db_url,
            connect_args={"check_same_thread": False}
        )
    return create_engine(db_url)

engine = _resolve_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
