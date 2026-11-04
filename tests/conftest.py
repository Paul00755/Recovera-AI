import os
import sys
from pathlib import Path

# Ensure backend is in python path
backend_path = Path(__file__).resolve().parent.parent / "backend"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Canonical test database (isolated from main production/development database)
os.environ["DATABASE_URL"] = "sqlite:///test_recovera.db"

# Initialize test database schema automatically
from app.models import init_db
init_db()
