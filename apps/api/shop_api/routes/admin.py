"""Audit access and explicit retention maintenance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from shop_api.deps import get_context, get_session
from shop_api.errors import ApiError, request_id
from shop_api.security import resolve_operator
from shop_api.service import AppContext, safe_media_path
from shop_core.persistence.models import AuditRow, EventRow, EvidenceRow, utcnow
from shop_core.persistence.repository import decode_cursor, encode_cursor, iso, new_id

router = APIRouter(tags=["audit"])


def _normalise(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


@router.get("/audit")
def list_audit(
    limit: int = Query(default=50),
    cursor: str | None = None,
    action: str | None = None,
    object_type: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    if limit < 1 or limit > 100:
        raise ApiError(422, "PAGE_LIMIT_INVALID", "limit must be between 1 and 100.")
    query = select(AuditRow)
    if action:
        query = query.where(AuditRow.action == action[:64])
    if object_type:
        query = query.where(AuditRow.object_type == object_type[:64])
    if cursor:
        try:
            created_at, cursor_id = decode_cursor(cursor)
        except (ValueError, KeyError, TypeError):
            raise ApiError(400, "INVALID_CURSOR", "The pagination cursor is malformed.")
        created_at = _normalise(created_at)
        query = query.where(
            or_(
                AuditRow.created_at < created_at,
                and_(AuditRow.created_at == created_at, AuditRow.id < cursor_id),
            )
        )
    rows = session.execute(
        query.order_by(AuditRow.created_at.desc(), AuditRow.id.desc()).limit(limit + 1)
    ).scalars().all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [
        {
            "audit_id": row.id,
            "actor": row.actor,
            "action": row.action,
            "object_type": row.object_type,
            "object_id": row.object_id,
            "request_id": row.request_id,
            "metadata": row.metadata_json or {},
            "created_at": iso(row.created_at),
        }
        for row in rows
    ]
    return {
        "items": items,
        "next_cursor": encode_cursor(rows[-1].created_at, rows[-1].id) if has_more and rows else None,
        "has_more": has_more,
    }


@router.post("/maintenance/retention", status_code=202)
def run_retention(
    request: Request,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> dict:
    """Apply the configured evidence TTL to expired clips.

    Metadata remains as an audit tombstone. The API performs only the explicit
    operator-triggered maintenance call; it never silently extends retention.
    """
    actor = resolve_operator(request)
    now = utcnow()
    rows = session.execute(
        select(EvidenceRow)
        .where(
            EvidenceRow.deleted_at.is_(None),
            EvidenceRow.retain_until.is_not(None),
            EvidenceRow.retain_until < now.replace(tzinfo=None),
        )
        .limit(500)
    ).scalars().all()
    paths: list[Path] = []
    expired_events = 0
    for evidence in rows:
        for relative in (evidence.object_path, evidence.thumbnail_path):
            path = safe_media_path(context.settings, relative)
            if path is not None and path.is_file():
                paths.append(path)
        evidence.deleted_at = now
        event = session.get(EventRow, evidence.event_id)
        if event is not None:
            event.evidence_status = "expired"
            event.evidence_reason = "Evidence retention period ended; metadata tombstone retained."
            expired_events += 1
    if rows:
        session.add(
            AuditRow(
                id=new_id(),
                actor=actor,
                action="retention_applied",
                object_type="evidence",
                object_id=None,
                request_id=request_id(request),
                metadata_json={"clips_expired": len(rows), "events_updated": expired_events},
            )
        )
    session.commit()
    cleanup_failures = 0
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            cleanup_failures += 1
    return {
        "clips_expired": len(rows),
        "events_updated": expired_events,
        "file_cleanup_failures": cleanup_failures,
        "applied_at": iso(now),
        "request_id": request_id(request),
    }
