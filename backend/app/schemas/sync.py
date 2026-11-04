"""
Recovera AI - Offline Sync Schemas (Phase 16)
"""
import datetime as dt
from typing import List, Optional
from pydantic import BaseModel, Field

class OfflineCheckinItem(BaseModel):
    patient_id: str
    client_timestamp: dt.datetime
    pain_level: int = Field(..., ge=0, le=10)
    mobility_score: Optional[int] = Field(None, ge=1, le=10)
    temperature: Optional[float] = None
    blood_pressure: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class OfflineAdherenceItem(BaseModel):
    occurrence_id: str
    state: str = Field(..., pattern=r"^(taken|not_taken)$")
    reason: Optional[str] = None
    client_timestamp: dt.datetime

class OfflineBatchSyncIn(BaseModel):
    client_sync_id: str
    device_id: Optional[str] = None
    synced_at: dt.datetime = Field(default_factory=dt.datetime.utcnow)
    checkins: List[OfflineCheckinItem] = Field(default_factory=list)
    adherences: List[OfflineAdherenceItem] = Field(default_factory=list)

class OfflineBatchSyncOut(BaseModel):
    status: str
    client_sync_id: str
    checkins_synced: int
    adherences_synced: int
    conflicts_resolved: int
    timestamp: dt.datetime
