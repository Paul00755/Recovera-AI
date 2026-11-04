"""
Recovera AI - Offline-First Field Clinic Synchronization API (Phase 16)

Endpoints:
  POST /api/sync/offline-batch - Atomically syncs queued offline check-ins and adherences
"""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.user import User
from app.schemas.sync import OfflineBatchSyncIn, OfflineBatchSyncOut
from app.services.sync_service import process_offline_batch_sync
from app.api.deps import get_current_user

router = APIRouter(prefix="/api/sync", tags=["Offline Field Clinic Synchronization"])

@router.post("/offline-batch", response_model=OfflineBatchSyncOut, status_code=status.HTTP_200_OK)
def sync_offline_batch(
    payload: OfflineBatchSyncIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Ingests and reconciles batched clinical events collected offline during connectivity outages.
    Applies deterministic Last-Write-Wins conflict resolution based on client event timestamps.
    """
    return process_offline_batch_sync(
        payload=payload,
        current_user_id=current_user.id,
        db=db
    )
