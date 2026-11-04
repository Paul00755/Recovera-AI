"""
Recovera AI - Offline Field Clinic Synchronization Service (Phase 16)

Handles atomic synchronization of offline-queued clinical check-ins and medication
adherence events collected by field nurses and healthcare workers in connectivity-deprived
or remote environments. Resolves concurrent updates using Last-Write-Wins (LWW).
"""
import datetime as dt
from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.checkin import DailyCheckIn
from app.models.medication import MedicationOccurrence, MedicationAdherence
from app.models.patient import Patient
from app.schemas.sync import OfflineBatchSyncIn, OfflineBatchSyncOut
from app.core.audit import record_audit

def process_offline_batch_sync(
    payload: OfflineBatchSyncIn,
    current_user_id: str,
    db: Session
) -> OfflineBatchSyncOut:
    """
    Synchronizes batched offline records into canonical database with LWW conflict resolution.
    Runs inside a single atomic database transaction.
    """
    checkins_synced = 0
    adherences_synced = 0
    conflicts_resolved = 0

    # 1. Process Check-ins
    for item in payload.checkins:
        patient = db.get(Patient, item.patient_id)
        if not patient:
            continue

        checkin_date = item.client_timestamp.date()
        existing = db.scalar(
            select(DailyCheckIn).where(
                DailyCheckIn.patient_id == item.patient_id,
                DailyCheckIn.checkin_date == checkin_date
            )
        )

        has_fever = False
        if item.temperature and item.temperature >= 100.4:
            has_fever = True
        elif any("fever" in s.lower() for s in item.symptoms):
            has_fever = True

        has_cough = any("cough" in s.lower() for s in item.symptoms)

        symptoms_str = f" Symptoms: {', '.join(item.symptoms)}." if item.symptoms else ""
        bp_str = f" BP: {item.blood_pressure}." if item.blood_pressure else ""
        combined_notes = f"{item.notes or ''}{symptoms_str}{bp_str}".strip()

        if existing:
            # Last-Write-Wins based on timestamp
            if item.client_timestamp > existing.submitted_at:
                existing.pain_level = item.pain_level
                existing.notes = combined_notes
                existing.fever = has_fever
                existing.cough = has_cough
                existing.submitted_at = item.client_timestamp
                conflicts_resolved += 1
                checkins_synced += 1
        else:
            new_checkin = DailyCheckIn(
                patient_id=item.patient_id,
                checkin_date=checkin_date,
                pain_level=item.pain_level,
                notes=combined_notes,
                fever=has_fever,
                cough=has_cough,
                submitted_at=item.client_timestamp
            )
            db.add(new_checkin)
            checkins_synced += 1

    # 2. Process Medication Adherences
    for item in payload.adherences:
        occ = db.get(MedicationOccurrence, item.occurrence_id)
        if not occ:
            continue

        existing_adh = db.scalar(
            select(MedicationAdherence).where(
                MedicationAdherence.occurrence_id == item.occurrence_id
            )
        )

        if existing_adh:
            # Last-Write-Wins based on client timestamp
            if item.client_timestamp > existing_adh.recorded_at:
                existing_adh.state = item.state
                existing_adh.reason = item.reason
                existing_adh.recorded_at = item.client_timestamp
                occ.state = item.state
                occ.finalized_at = item.client_timestamp
                conflicts_resolved += 1
                adherences_synced += 1
        else:
            new_adh = MedicationAdherence(
                occurrence_id=item.occurrence_id,
                state=item.state,
                reason=item.reason,
                recorded_by=current_user_id,
                recorded_at=item.client_timestamp
            )
            db.add(new_adh)
            occ.state = item.state
            occ.finalized_at = item.client_timestamp
            adherences_synced += 1

    # Record sync batch in audit log
    record_audit(
        db=db,
        actor_id=current_user_id,
        action="offline_batch_synced",
        target_type="sync_batch",
        target_id=payload.client_sync_id,
        details={
            "device_id": payload.device_id,
            "checkins_synced": checkins_synced,
            "adherences_synced": adherences_synced,
            "conflicts_resolved": conflicts_resolved,
            "synced_at": payload.synced_at.isoformat()
        }
    )

    db.commit()

    return OfflineBatchSyncOut(
        status="synced",
        client_sync_id=payload.client_sync_id,
        checkins_synced=checkins_synced,
        adherences_synced=adherences_synced,
        conflicts_resolved=conflicts_resolved,
        timestamp=dt.datetime.utcnow()
    )
