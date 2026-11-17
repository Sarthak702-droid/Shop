"""Event queue, evidence media, append-only reviews and deletion."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, selectinload

from shop_api.deps import get_context, get_session
from shop_api.errors import ApiError, request_id
from shop_api.schemas import ReviewCreateRequest
from shop_api.security import resolve_operator
from shop_api.service import (
    AppContext,
    body_hash,
    event_api,
    get_idempotent,
    review_api,
    safe_media_path,
    store_idempotent,
)
from shop_core.persistence.models import (
    AuditRow,
    EventRow,
    EvidenceRow,
    JobRow,
    ReviewRow,
    SourceRow,
    utcnow,
)
from shop_core.persistence.repository import decode_cursor, encode_cursor, iso, new_id

router = APIRouter(tags=["events"])
REVIEW_STATES = ("unreviewed", "benign", "review_needed", "inconclusive")


def _normalise_cursor_time(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(UTC).replace(tzinfo=None)
    return value


def _event_or_404(session: Session, event_id: UUID) -> EventRow:
    row = session.execute(
        select(EventRow)
        .where(EventRow.id == str(event_id), EventRow.deleted_at.is_(None))
        .options(selectinload(EventRow.reviews), selectinload(EventRow.evidence))
    ).scalar_one_or_none()
    if row is None:
        raise ApiError(404, "EVENT_NOT_FOUND", "The event is not available.")
    return row


def _source_details(session: Session, source_ids: set[str]) -> dict[str, dict]:
    if not source_ids:
        return {}
    rows = session.execute(select(SourceRow).where(SourceRow.id.in_(source_ids))).scalars().all()
    return {
        row.id: {"source_id": row.id, "label": row.label, "camera_key": row.camera_key, "kind": row.kind}
        for row in rows
    }


def _with_source(session: Session, row: EventRow) -> dict:
    payload = event_api(row)
    source = session.get(SourceRow, row.source_id)
    payload["source"] = (
        {
            "source_id": source.id,
            "label": source.label,
            "camera_key": source.camera_key,
            "kind": source.kind,
        }
        if source is not None
        else None
    )
    job = session.get(JobRow, row.job_id)
    payload["job_status"] = job.state if job is not None else None
    return payload


@router.get("/events")
def list_events(
    limit: int = Query(default=20),
    cursor: str | None = None,
    review_state: str | None = None,
    reviewed: bool | None = None,
    job_id: UUID | None = None,
    include_deleted: bool = False,
    session: Session = Depends(get_session),
) -> dict:
    if limit < 1 or limit > 100:
        raise ApiError(422, "PAGE_LIMIT_INVALID", "limit must be between 1 and 100.")
    if review_state is not None and review_state not in REVIEW_STATES:
        raise ApiError(422, "REVIEW_STATE_INVALID", "The requested review state is not recognised.")
    query = select(EventRow).options(selectinload(EventRow.evidence))
    if not include_deleted:
        query = query.where(EventRow.deleted_at.is_(None))
    if review_state is not None:
        query = query.where(EventRow.review_state == review_state)
    if reviewed is True:
        query = query.where(EventRow.review_state != "unreviewed")
    elif reviewed is False:
        query = query.where(EventRow.review_state == "unreviewed")
    if job_id is not None:
        query = query.where(EventRow.job_id == str(job_id))
    if cursor:
        try:
            created_at, cursor_id = decode_cursor(cursor)
        except (ValueError, KeyError, TypeError):
            raise ApiError(400, "INVALID_CURSOR", "The pagination cursor is malformed.")
        query = query.where(
            or_(
                EventRow.created_at < _normalise_cursor_time(created_at),
                and_(
                    EventRow.created_at == _normalise_cursor_time(created_at),
                    EventRow.id < cursor_id,
                ),
            )
        )
    rows = session.execute(
        query.order_by(EventRow.created_at.desc(), EventRow.id.desc()).limit(limit + 1)
    ).scalars().all()
    has_more = len(rows) > limit
    rows = rows[:limit]
    sources = _source_details(session, {row.source_id for row in rows})
    items = []
    for row in rows:
        payload = event_api(row)
        payload["source"] = sources.get(row.source_id)
        items.append(payload)
    next_cursor = encode_cursor(rows[-1].created_at, rows[-1].id) if has_more and rows else None
    unreviewed = int(
        session.scalar(
            select(func.count())
            .select_from(EventRow)
            .where(EventRow.deleted_at.is_(None), EventRow.review_state == "unreviewed")
        )
        or 0
    )
    return {
        "items": items,
        "next_cursor": next_cursor,
        "has_more": has_more,
        "unreviewed_count": unreviewed,
    }


@router.get("/events/{event_id}")
def get_event(event_id: UUID, session: Session = Depends(get_session)) -> dict:
    row = _event_or_404(session, event_id)
    payload = _with_source(session, row)
    payload["reviews"] = [review_api(review) for review in sorted(row.reviews, key=lambda item: item.created_at)]
    if row.evidence is not None:
        payload["evidence"]["asset"] = {
            "clip_id": row.evidence.id,
            "bytes": row.evidence.bytes,
            "sha256": row.evidence.sha256,
            "mime_type": row.evidence.mime_type,
            "width": row.evidence.width,
            "height": row.evidence.height,
            "fps": row.evidence.fps,
            "duration_ms": row.evidence.duration_ms,
            "frame_count": row.evidence.frame_count,
            "media_interval_ms": {
                "start": row.evidence.start_pts_ms,
                "end": row.evidence.end_pts_ms,
            },
            "encoder": row.evidence.encoder,
            "retain_until": row.evidence.retain_until.isoformat() if row.evidence.retain_until else None,
        }
    return {"event": payload}


@router.post("/events/{event_id}/reviews", status_code=201)
def create_review(
    event_id: UUID,
    payload: ReviewCreateRequest,
    request: Request,
    session: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    event = _event_or_404(session, event_id)
    actor = resolve_operator(request)
    body = payload.model_dump(mode="json")
    key = (idempotency_key or "").strip()
    route = f"/api/v1/events/{event_id}/reviews"
    if key:
        if len(key) < 8 or len(key) > 100:
            raise ApiError(422, "IDEMPOTENCY_KEY_INVALID", "Idempotency-Key must be 8-100 characters.")
        existing = get_idempotent(session, actor=actor, route=route, key=key, body=body)
        if existing is not None:
            return JSONResponse(
                status_code=existing.status_code,
                content=existing.response_json,
                headers={"Idempotency-Replayed": "true"},
            )
    supersedes = None
    if payload.supersedes_review_id is not None:
        supersedes = session.get(ReviewRow, str(payload.supersedes_review_id))
        if supersedes is None or supersedes.event_id != event.id:
            raise ApiError(
                422,
                "SUPERSEDES_REVIEW_INVALID",
                "A superseding review must reference an existing review of the same event.",
            )
    review = ReviewRow(
        id=new_id(),
        event_id=event.id,
        actor=actor,
        decision=payload.decision,
        reason_codes=list(payload.reason_codes),
        notes=payload.notes.strip(),
        supersedes_review_id=supersedes.id if supersedes is not None else None,
    )
    session.add(review)
    event.review_state = payload.decision
    session.add(
        AuditRow(
            id=new_id(),
            actor=actor,
            action="event_reviewed",
            object_type="event",
            object_id=event.id,
            request_id=request_id(request),
            metadata_json={
                "review_id": review.id,
                "decision": payload.decision,
                "reason_codes": list(payload.reason_codes),
                "supersedes_review_id": review.supersedes_review_id,
            },
        )
    )
    session.flush()
    result = {
        "review": review_api(review),
        "event_id": event.id,
        "review_state": event.review_state,
        "request_id": request_id(request),
        "notice": (
            "Review recorded. This decision is an operational review outcome, not a finding of guilt."
        ),
    }
    if key:
        store_idempotent(
            session,
            actor=actor,
            route=route,
            key=key,
            body=body,
            status_code=201,
            response=result,
        )
    session.commit()
    return JSONResponse(status_code=201, content=result)


def _evidence_response(
    request: Request,
    event_id: UUID,
    *,
    thumbnail: bool,
    session: Session,
    context: AppContext,
) -> FileResponse:
    event = _event_or_404(session, event_id)
    evidence = event.evidence
    if evidence is None or evidence.deleted_at is not None or event.evidence_status != "available":
        raise ApiError(
            410,
            "EVIDENCE_UNAVAILABLE",
            event.evidence_reason or "Evidence media is not available for this event.",
        )
    relative = evidence.thumbnail_path if thumbnail else evidence.object_path
    path = safe_media_path(context.settings, relative)
    if path is None or not path.is_file():
        raise ApiError(
            410,
            "EVIDENCE_MISSING",
            "Evidence media is unavailable on local storage. The event remains in the review queue.",
        )
    suffix = path.suffix.lower()
    media_type = "image/jpeg" if thumbnail or suffix in {".jpg", ".jpeg"} else "video/webm"
    filename = f"shop-evidence-{event.id}{suffix if suffix else ''}"
    headers = {
        "Cache-Control": "private, no-store",
        "Content-Disposition": f'inline; filename="{filename}"',
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; media-src 'self'; img-src 'self'",
        "X-Request-ID": request_id(request),
    }
    if not thumbnail:
        headers["ETag"] = f'"{evidence.sha256}"'
    return FileResponse(
        path,
        media_type=media_type,
        filename=filename,
        content_disposition_type="inline",
        headers=headers,
    )


@router.get("/events/{event_id}/media")
def event_media(
    event_id: UUID,
    request: Request,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> FileResponse:
    return _evidence_response(
        request, event_id, thumbnail=False, session=session, context=context
    )


@router.get("/events/{event_id}/thumbnail")
def event_thumbnail(
    event_id: UUID,
    request: Request,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> FileResponse:
    return _evidence_response(
        request, event_id, thumbnail=True, session=session, context=context
    )


@router.delete("/events/{event_id}", status_code=202)
def delete_event(
    event_id: UUID,
    request: Request,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> dict:
    event = session.get(EventRow, str(event_id))
    if event is None:
        raise ApiError(404, "EVENT_NOT_FOUND", "The event is not available.")
    actor = resolve_operator(request)
    if event.deleted_at is not None:
        return {
            "event_id": event.id,
            "status": "deleted",
            "deleted_at": event.deleted_at.isoformat(),
            "request_id": request_id(request),
        }
    evidence = session.get(EvidenceRow, event.evidence_id) if event.evidence_id else None
    paths: list[Path] = []
    if evidence is not None:
        for relative in (evidence.object_path, evidence.thumbnail_path):
            path = safe_media_path(context.settings, relative)
            if path is not None and path.is_file():
                paths.append(path)
        evidence.deleted_at = utcnow()
        event.evidence_status = "deleted"
        event.evidence_reason = "Deleted by an authorized local operator."
    event.deleted_at = utcnow()
    session.add(
        AuditRow(
            id=new_id(),
            actor=actor,
            action="event_deleted",
            object_type="event",
            object_id=event.id,
            request_id=request_id(request),
            metadata_json={
                "evidence_deleted": evidence is not None,
                "clip_id": evidence.id if evidence is not None else None,
            },
        )
    )
    session.commit()
    unlink_failures = 0
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            unlink_failures += 1
    return {
        "event_id": event.id,
        "status": "deleted",
        "evidence_deleted": evidence is not None,
        "file_cleanup_failures": unlink_failures,
        "request_id": request_id(request),
    }
