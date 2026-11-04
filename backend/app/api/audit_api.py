"""
Recovera AI - Regulatory Audit Trail & Export API (Phase 15)

Endpoints:
  GET /api/audit/verify-chain - Cryptographically verifies the SHA-256 hash chain
  GET /api/audit/logs         - List audit log entries with filter & pagination
  GET /api/audit/export       - Export immutable audit trail to CSV or JSON
"""
import io
import csv
import json
import datetime as dt
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.core.database import get_db
from app.core.constants import Role
from app.models.user import User
from app.models.audit_log import AuditLog
from app.api.deps import require_role
from app.core.audit import verify_audit_log_chain

router = APIRouter(prefix="/api/audit", tags=["Regulatory Audit Trail & Compliance"])

@router.get("/verify-chain")
def verify_audit_chain_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Verifies cryptographic SHA-256 integrity across the entire hospital audit ledger."""
    return verify_audit_log_chain(db)

@router.get("/logs")
def list_audit_logs(
    action: Optional[str] = Query(None),
    actor_id: Optional[str] = Query(None),
    target_type: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Returns paginated audit records for administrators."""
    query = select(AuditLog)

    if action:
        query = query.where(AuditLog.action == action)
    if actor_id:
        query = query.where(AuditLog.actor_id == actor_id)
    if target_type:
        query = query.where(AuditLog.target_type == target_type)

    query = query.order_by(desc(AuditLog.created_at)).offset(offset).limit(limit)
    entries = db.scalars(query).all()

    # Pre-fetch actors
    actor_ids = list({e.actor_id for e in entries if e.actor_id})
    user_map = {}
    if actor_ids:
        users = db.scalars(select(User).where(User.id.in_(actor_ids))).all()
        user_map = {u.id: u for u in users}

    result = []
    for e in entries:
        actor = user_map.get(e.actor_id)
        result.append({
            "id": e.id,
            "actor_id": e.actor_id,
            "actor_name": actor.name if actor else ("System" if not e.actor_id else "Unknown"),
            "actor_role": actor.role if actor else None,
            "action": e.action,
            "target_type": e.target_type,
            "target_id": e.target_id,
            "details": e.details,
            "previous_hash": e.previous_hash,
            "entry_hash": e.entry_hash,
            "created_at": e.created_at.isoformat()
        })

    return {
        "count": len(result),
        "limit": limit,
        "offset": offset,
        "logs": result
    }

@router.get("/export")
def export_audit_logs(
    format: str = Query("csv", pattern="^(csv|json)$"),
    limit: int = Query(1000, le=5000),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """Exports cryptographic audit trail for external compliance inspections."""
    entries = db.scalars(
        select(AuditLog).order_by(AuditLog.created_at.asc()).limit(limit)
    ).all()

    # Pre-fetch actors
    actor_ids = list({e.actor_id for e in entries if e.actor_id})
    user_map = {}
    if actor_ids:
        users = db.scalars(select(User).where(User.id.in_(actor_ids))).all()
        user_map = {u.id: u for u in users}

    if format == "json":
        export_data = []
        for e in entries:
            actor = user_map.get(e.actor_id)
            export_data.append({
                "id": e.id,
                "actor_id": e.actor_id,
                "actor_name": actor.name if actor else ("System" if not e.actor_id else "Unknown"),
                "actor_role": actor.role if actor else None,
                "action": e.action,
                "target_type": e.target_type,
                "target_id": e.target_id,
                "details": e.details,
                "previous_hash": e.previous_hash,
                "entry_hash": e.entry_hash,
                "created_at": e.created_at.isoformat()
            })
        json_content = json.dumps(export_data, indent=2, default=str)
        filename = f"recovera_audit_export_{dt.date.today().isoformat()}.json"
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    # CSV format
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "timestamp_utc", "actor_name", "actor_role", "action",
        "target_type", "target_id", "previous_hash", "entry_hash", "details"
    ])

    for e in entries:
        actor = user_map.get(e.actor_id)
        writer.writerow([
            e.id,
            e.created_at.isoformat(),
            actor.name if actor else ("System" if not e.actor_id else "Unknown"),
            actor.role if actor else "system",
            e.action,
            e.target_type,
            e.target_id or "",
            e.previous_hash or "",
            e.entry_hash or "",
            json.dumps(e.details or {}, default=str)
        ])

    csv_data = output.getvalue()
    filename = f"recovera_audit_export_{dt.date.today().isoformat()}.csv"
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
