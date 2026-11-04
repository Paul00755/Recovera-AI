"""
Recovera AI - Regulatory Audit Trail with Cryptographic SHA-256 Hash Chaining (Phase 15)

Guarantees immutable, verifiable logging for all clinical modifications, prescriptions,
and clinician overrides in compliance with regulatory standards (HIPAA / GDPR).
"""
import json
import hashlib
import datetime as dt
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import select

GENESIS_HASH = "0" * 64

def compute_audit_hash(
    previous_hash: str,
    actor_id: Optional[str],
    action: str,
    target_type: str,
    target_id: Optional[str],
    metadata: Dict[str, Any],
    created_at: dt.datetime
) -> str:
    """Computes a deterministic SHA-256 hash for an audit log entry."""
    # Canonical JSON string with sorted keys
    meta_serialized = json.dumps(metadata, sort_keys=True, default=str)
    raw = f"{previous_hash}|{actor_id or ''}|{action}|{target_type}|{target_id or ''}|{meta_serialized}|{created_at.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def record_audit(
    db: Session,
    actor_id: Optional[str],
    action: str,
    target_type: str,
    target_id: Optional[str],
    old_values: Optional[Dict[str, Any]] = None,
    new_values: Optional[Dict[str, Any]] = None,
    details: Optional[Dict[str, Any]] = None
):
    from app.models.audit_log import AuditLog

    metadata: Dict[str, Any] = {}
    if details:
        metadata.update(details)
    if old_values:
        metadata["old_values"] = old_values
    if new_values:
        metadata["new_values"] = new_values

    # Query the most recent audit entry to link the hash chain
    latest_entry = db.scalar(
        select(AuditLog).order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).limit(1)
    )

    previous_hash = (
        latest_entry.entry_hash
        if (latest_entry and latest_entry.entry_hash)
        else GENESIS_HASH
    )

    created_at = dt.datetime.utcnow()
    entry_hash = compute_audit_hash(
        previous_hash=previous_hash,
        actor_id=actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        metadata=metadata,
        created_at=created_at
    )

    entry = AuditLog(
        actor_id=actor_id,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=metadata,
        previous_hash=previous_hash,
        entry_hash=entry_hash,
        created_at=created_at
    )
    db.add(entry)
    return entry

def verify_audit_log_chain(db: Session) -> Dict[str, Any]:
    """
    Traverses the audit log chronologically to cryptographically verify hash chain integrity.
    Detects any unauthorized tampering, deletions, or modifications.
    """
    from app.models.audit_log import AuditLog

    entries = db.scalars(
        select(AuditLog).order_by(AuditLog.created_at.asc(), AuditLog.id.asc())
    ).all()

    if not entries:
        return {
            "is_valid": True,
            "total_entries": 0,
            "verified_entries": 0,
            "broken_index": None,
            "latest_hash": GENESIS_HASH,
            "algorithm": "SHA-256",
            "verified_at": dt.datetime.utcnow().isoformat(),
            "message": "Audit chain is empty. Genesis state intact."
        }

    # Only verify entries that have hash fields populated
    expected_prev_hash = GENESIS_HASH
    verified_count = 0

    for idx, entry in enumerate(entries):
        if not entry.entry_hash:
            # Legacy entry before hash chaining was introduced; bridge safely
            continue

        # Check previous_hash link
        if verified_count > 0 and entry.previous_hash != expected_prev_hash:
            return {
                "is_valid": False,
                "total_entries": len(entries),
                "verified_entries": verified_count,
                "broken_index": idx,
                "broken_entry_id": entry.id,
                "error": f"Previous hash mismatch at index {idx}. Expected {expected_prev_hash[:16]}..., found {str(entry.previous_hash)[:16]}...",
                "latest_hash": expected_prev_hash,
                "algorithm": "SHA-256",
                "verified_at": dt.datetime.utcnow().isoformat()
            }

        # Re-compute hash from row data
        recomputed = compute_audit_hash(
            previous_hash=entry.previous_hash or GENESIS_HASH,
            actor_id=entry.actor_id,
            action=entry.action,
            target_type=entry.target_type,
            target_id=entry.target_id,
            metadata=entry.details or {},
            created_at=entry.created_at
        )

        if recomputed != entry.entry_hash:
            return {
                "is_valid": False,
                "total_entries": len(entries),
                "verified_entries": verified_count,
                "broken_index": idx,
                "broken_entry_id": entry.id,
                "error": f"Data tampering detected at index {idx}. Hash mismatch: expected {entry.entry_hash[:16]}..., computed {recomputed[:16]}...",
                "latest_hash": expected_prev_hash,
                "algorithm": "SHA-256",
                "verified_at": dt.datetime.utcnow().isoformat()
            }

        expected_prev_hash = entry.entry_hash
        verified_count += 1

    return {
        "is_valid": True,
        "total_entries": len(entries),
        "verified_entries": verified_count,
        "broken_index": None,
        "latest_hash": expected_prev_hash,
        "algorithm": "SHA-256",
        "verified_at": dt.datetime.utcnow().isoformat(),
        "message": f"Cryptographic audit chain intact. Verified {verified_count} sequential entries."
    }
