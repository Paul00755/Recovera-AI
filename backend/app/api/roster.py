"""
Recovera AI - Clinician Roster & Auto-Load Balancing API (Phase 14)

Endpoints:
  GET  /api/doctors/load-stats         - Doctor workload metrics & availability
  POST /api/patients/{pid}/auto-assign - Intelligently auto-assign least loaded clinician
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.constants import Role
from app.models.user import User
from app.api.deps import get_current_user, require_role
from app.services.roster_service import get_doctor_roster_stats, auto_assign_best_doctor

router = APIRouter(tags=["Clinician Roster & Auto-Load Balancing"])

class AutoAssignRequest(BaseModel):
    specialization: Optional[str] = None
    department: Optional[str] = None

@router.get("/api/doctors/load-stats")
def get_doctor_load_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """Returns real-time capacity and workload analysis for all clinicians."""
    stats = get_doctor_roster_stats(db)
    return {
        "count": len(stats),
        "doctors": stats
    }

@router.post("/api/patients/{pid}/auto-assign")
def auto_assign_patient(
    pid: str,
    payload: Optional[AutoAssignRequest] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.NURSE.value))
):
    """Auto-assigns the optimal, least-burdened doctor to the patient based on specialization and workload."""
    spec = payload.specialization if payload else None
    dept = payload.department if payload else None
    result = auto_assign_best_doctor(
        patient_id=pid,
        db=db,
        assigned_by_user_id=current_user.id,
        specialization=spec,
        department=dept
    )
    return result
