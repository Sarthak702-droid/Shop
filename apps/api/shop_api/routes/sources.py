"""Approved webcam source registration, preview and versioned metadata."""

from __future__ import annotations

import hashlib
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request, Response
from fastapi.responses import JSONResponse, Response as PlainResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from shop_api.deps import get_context, get_session
from shop_api.errors import ApiError, request_id
from shop_api.schemas import SourceCreateRequest, SourcePatchRequest
from shop_api.security import resolve_operator
from shop_api.service import AppContext, active_source_job, source_api
from shop_core.persistence.models import AuditRow, JobRow, SourceRow, utcnow
from shop_core.persistence.repository import iso, new_id
from shop_core.sources.media import SourceFault, capture_webcam_snapshot, WebcamSource

router = APIRouter(tags=["sources"])


def _etag(row: SourceRow) -> str:
    # Normalize timezone representation: SQLite returns naive datetimes while
    # an in-session row may still be timezone-aware. The same version must hash
    # identically across create, read and conditional update.
    stamp = row.updated_at.replace(tzinfo=None).isoformat() if row.updated_at else ""
    return hashlib.sha256(row.id.encode() + b":" + stamp.encode()).hexdigest()[:32]


def _require_source(session: Session, source_id: UUID) -> SourceRow:
    row = session.get(SourceRow, str(source_id))
    if row is None or row.kind != "webcam":
        raise ApiError(404, "SOURCE_NOT_FOUND", "The source is not available.")
    return row


@router.post("/sources", status_code=201)
def create_source(
    payload: SourceCreateRequest,
    request: Request,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> JSONResponse:
    settings = context.settings
    camera = settings.camera(payload.camera_key)
    if camera is None:
        raise ApiError(
            422,
            "CAMERA_NOT_APPROVED",
            "Only cameras explicitly approved in local configuration can be registered.",
            details={"approved_camera_keys": [item.key for item in settings.security.approved_cameras]},
        )
    existing = session.execute(
        select(SourceRow).where(
            SourceRow.kind == "webcam", SourceRow.camera_key == payload.camera_key
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise ApiError(
            409,
            "SOURCE_ALREADY_EXISTS",
            "This approved camera already has a local source registration.",
            details={"source_id": existing.id},
        )
    actor = resolve_operator(request)
    row = SourceRow(
        id=new_id(),
        store_id="local-store",
        kind="webcam",
        label=payload.label or camera.label,
        camera_key=camera.key,
        config={"camera_key": camera.key},
        state="idle",
    )
    session.add(row)
    session.add(
        AuditRow(
            id=new_id(),
            actor=actor,
            action="source_created",
            object_type="source",
            object_id=row.id,
            request_id=request_id(request),
            metadata_json={"camera_key": camera.key, "kind": "webcam"},
        )
    )
    session.flush()
    body = {"source": source_api(row, settings), "request_id": request_id(request)}
    return JSONResponse(status_code=201, content=body, headers={"ETag": _etag(row)})


@router.get("/sources")
def list_sources(
    session: Session = Depends(get_session), context: AppContext = Depends(get_context)
) -> dict:
    rows = session.execute(
        select(SourceRow).where(SourceRow.kind == "webcam").order_by(SourceRow.created_at, SourceRow.id)
    ).scalars().all()
    approved = {camera.key: camera for camera in context.settings.security.approved_cameras}
    active_jobs = {
        job.source_id: job
        for job in session.execute(
            select(JobRow)
            .where(
                JobRow.source_id.in_([row.id for row in rows]),
                JobRow.state.in_(("queued", "running", "cancelling")),
            )
            .order_by(JobRow.created_at.desc(), JobRow.id.desc())
        ).scalars()
    }
    return {
        "items": [source_api(row, context.settings, active_jobs.get(row.id)) for row in rows],
        "approved_cameras": [
            {
                "key": camera.key,
                "label": camera.label,
                "device_index": camera.device_index,
                "registered": any(source.camera_key == camera.key for source in rows),
            }
            for camera in approved.values()
        ],
    }


@router.get("/sources/{source_id}")
def get_source(
    source_id: UUID, response: Response, session: Session = Depends(get_session), context: AppContext = Depends(get_context)
) -> dict:
    row = _require_source(session, source_id)
    response.headers["ETag"] = _etag(row)
    return {"source": source_api(row, context.settings, active_source_job(session, row.id))}


@router.patch("/sources/{source_id}")
def patch_source(
    source_id: UUID,
    payload: SourcePatchRequest,
    request: Request,
    response: Response,
    if_match: str | None = Header(default=None, alias="If-Match"),
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> dict:
    row = _require_source(session, source_id)
    if not if_match:
        raise ApiError(428, "IF_MATCH_REQUIRED", "PATCH requires the current ETag in If-Match.")
    current = _etag(row)
    normalized = if_match.strip().strip('"')
    if normalized != current:
        raise ApiError(412, "STALE_VERSION", "The source changed since it was read. Reload and retry.")
    actor = resolve_operator(request)
    row.label = payload.label
    row.updated_at = utcnow()
    session.add(
        AuditRow(
            id=new_id(),
            actor=actor,
            action="source_updated",
            object_type="source",
            object_id=row.id,
            request_id=request_id(request),
            metadata_json={"fields": ["label"]},
        )
    )
    session.flush()
    response.headers["ETag"] = _etag(row)
    return {"source": source_api(row, context.settings), "request_id": request_id(request)}


@router.get("/sources/{source_id}/health")
async def source_health(
    source_id: UUID,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> dict:
    row = _require_source(session, source_id)
    camera = context.settings.camera(row.camera_key or "")
    if camera is None:
        raise ApiError(403, "CAMERA_NOT_APPROVED", "The registered source is no longer approved.")
    probe = await run_in_threadpool(WebcamSource.probe, camera.device_index)
    state = "active" if probe.get("opened") and probe.get("frame_read") else "degraded"
    if probe.get("device_present") is False:
        state = "disconnected"
    return {
        "source_id": row.id,
        "camera_key": row.camera_key,
        "state": state,
        "probe": probe,
        "checked_at": iso(utcnow()),
    }


@router.get("/sources/{source_id}/preview")
async def source_preview(
    source_id: UUID,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
) -> PlainResponse:
    row = _require_source(session, source_id)
    camera = context.settings.camera(row.camera_key or "")
    if camera is None:
        raise ApiError(403, "CAMERA_NOT_APPROVED", "The registered source is no longer approved.")
    try:
        image, metadata = await run_in_threadpool(
            capture_webcam_snapshot,
            camera.device_index,
            privacy_polygon=context.settings.zones.privacy_mask,
        )
    except SourceFault as error:
        raise ApiError(503, error.code, error.message, retryable=True) from error
    return PlainResponse(
        content=image,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Shop-Preview-Backend": str(metadata.get("backend", "unknown")),
            "X-Shop-Captured-At": str(metadata.get("captured_at_utc", "")),
        },
    )
