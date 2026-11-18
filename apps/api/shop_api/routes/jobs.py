"""Durable analysis job creation, inspection and idempotent cancellation."""

from __future__ import annotations

import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from shop_api.deps import get_context, get_session
from shop_api.errors import ApiError, request_id
from shop_api.schemas import JobCreateRequest
from shop_api.security import resolve_operator
from shop_api.service import (
    ACTION_PROFILE,
    ANALYSIS_PROFILE,
    ANALYSIS_PROFILES,
    AppContext,
    body_hash,
    disk_status,
    get_idempotent,
    job_api,
    store_idempotent,
)
from shop_core.persistence.models import (
    AuditRow,
    JobRow,
    JobTransitionRow,
    SourceRow,
    UploadRow,
    utcnow,
)
from shop_core.persistence.repository import job_queue_depth, new_id, transition_job
from shop_core.sources.media import SourceFault, probe_video

router = APIRouter(tags=["jobs"])
ACTIVE_STATES = ("queued", "running", "cancelling")
TERMINAL_STATES = ("completed", "failed", "cancelled")


def _job_or_404(session: Session, job_id: UUID) -> JobRow:
    row = session.get(JobRow, str(job_id))
    if row is None:
        raise ApiError(404, "JOB_NOT_FOUND", "The analysis job is not available.")
    return row


@router.post("/jobs", status_code=202)
def create_job(
    payload: JobCreateRequest,
    request: Request,
    session: Session = Depends(get_session),
    context: AppContext = Depends(get_context),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    settings = context.settings
    actor = resolve_operator(request)
    body = payload.model_dump(mode="json")
    key = (idempotency_key or "").strip()
    if key and (len(key) < 8 or len(key) > 100):
        raise ApiError(422, "IDEMPOTENCY_KEY_INVALID", "Idempotency-Key must be 8-100 characters.")
    existing = (
        get_idempotent(session, actor=actor, route="/api/v1/jobs", key=key, body=body)
        if key
        else None
    )
    if existing is not None:
        return JSONResponse(
            status_code=existing.status_code,
            content=existing.response_json,
            headers={"Idempotency-Replayed": "true"},
        )
    if payload.analysis_profile not in ANALYSIS_PROFILES:
        raise ApiError(
            422,
            "ANALYSIS_PROFILE_UNSUPPORTED",
            "The requested analysis profile is not registered on this host.",
            details={"supported_profile": ANALYSIS_PROFILE, "supported_profiles": list(ANALYSIS_PROFILES)},
        )
    action_job = payload.analysis_profile == ACTION_PROFILE
    if action_job and payload.source.kind != "file":
        raise ApiError(
            422,
            "ANALYSIS_PROFILE_SOURCE_UNSUPPORTED",
            "Shelf-action research analysis supports prerecorded file uploads only.",
        )
    if payload.sampling_fps > settings.analysis.sampling_fps:
        raise ApiError(
            422,
            "SAMPLING_FPS_UNSUPPORTED",
            "The requested cadence exceeds the configured CPU analysis profile.",
            details={"maximum_sampling_fps": settings.analysis.sampling_fps},
        )
    if payload.privacy_mask_version != settings.zones.version:
        raise ApiError(
            422,
            "ZONE_VERSION_MISMATCH",
            "The requested privacy mask version is not the active local version.",
            details={"active_version": settings.zones.version},
        )
    context.require_person_model()
    if action_job:
        context.require_action_model()
    disk = disk_status(settings)
    if disk["below_minimum"]:
        raise ApiError(
            503,
            "STORAGE_LOW",
            "Analysis cannot start while free storage is below the configured reserve.",
            retryable=True,
        )
    depth = job_queue_depth(session)
    capacity = settings.runtime.active_jobs + settings.runtime.queued_jobs
    if depth >= capacity:
        raise ApiError(
            429,
            "QUEUE_FULL",
            "The local analysis queue is full. Wait for an active job to finish and retry.",
            retryable=True,
            details={"depth": depth, "capacity": capacity},
            headers={"Retry-After": "10"},
        )

    kind = payload.source.kind
    upload_id: str | None = None
    source_id: str | None = None
    duration_ms: int | None = None
    if payload.source.kind == "file":
        upload = session.get(UploadRow, str(payload.source.upload_id))
        if upload is None or upload.deleted_at is not None:
            raise ApiError(404, "UPLOAD_NOT_FOUND", "The referenced upload is not available.")
        if upload.status != "ready":
            raise ApiError(422, "UPLOAD_NOT_READY", "The referenced upload has not passed validation.")
        upload_id = upload.id
        duration_ms = (upload.probe or {}).get("duration_ms")
    else:
        source = session.get(SourceRow, str(payload.source.source_id))
        if source is None or source.kind != "webcam" or not source.camera_key:
            raise ApiError(404, "SOURCE_NOT_FOUND", "The referenced webcam source is not available.")
        if settings.camera(source.camera_key) is None:
            raise ApiError(403, "CAMERA_NOT_APPROVED", "The referenced camera is no longer approved.")
        source_id = source.id

    job_id = new_id()
    model_bundle = context.action_bundle() if action_job else context.model_bundle()
    row = JobRow(
        id=job_id,
        kind=kind,
        upload_id=upload_id,
        source_id=source_id,
        state="queued",
        analysis_profile=payload.analysis_profile,
        sampling_fps=payload.sampling_fps,
        privacy_mask_version=payload.privacy_mask_version,
        retain_evidence=payload.retain_evidence,
        model_bundle=model_bundle,
        policy_version="merl_action_episodes_v1" if action_job else settings.policy.version,
        progress={
            "stage": "queued",
            "media_pts_ms": 0,
            "duration_ms": duration_ms,
            "analysed_frames": 0,
            "dropped_frames": 0,
            "events_created": 0,
        },
        metrics={},
        coverage={},
        cancel_requested=False,
    )
    session.add(row)
    session.add(
        JobTransitionRow(
            job_id=job_id, from_state=None, to_state="queued", reason="http_request_accepted"
        )
    )
    session.add(
        AuditRow(
            id=new_id(),
            actor=actor,
            action="job_queued",
            object_type="job",
            object_id=job_id,
            request_id=request_id(request),
            metadata_json={
                "kind": kind,
                "upload_id": upload_id,
                "source_id": source_id,
                "model_bundle": model_bundle,
                "sampling_fps": payload.sampling_fps,
            },
        )
    )
    session.flush()
    job_payload = job_api(session, row)
    job_payload["request_id"] = request_id(request)
    response_body = {"job": job_payload, "request_id": request_id(request)}
    if key:
        store_idempotent(
            session,
            actor=actor,
            route="/api/v1/jobs",
            key=key,
            body=body,
            status_code=202,
            response=response_body,
        )
    session.commit()
    return JSONResponse(
        status_code=202,
        content=response_body,
        headers={
            "Location": f"/api/v1/jobs/{job_id}",
            "Retry-After": "2",
        },
    )


@router.get("/jobs")
def list_jobs(
    limit: int = 50,
    state: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    if limit < 1 or limit > 100:
        raise ApiError(422, "PAGE_LIMIT_INVALID", "limit must be between 1 and 100.")
    query = select(JobRow)
    if state is not None:
        if state not in ACTIVE_STATES + TERMINAL_STATES:
            raise ApiError(422, "JOB_STATE_INVALID", "The requested job state is not recognised.")
        query = query.where(JobRow.state == state)
    rows = session.execute(
        query.order_by(JobRow.created_at.desc(), JobRow.id.desc()).limit(limit)
    ).scalars().all()
    return {"items": [job_api(session, row) for row in rows], "next_cursor": None}


@router.get("/jobs/{job_id}")
def get_job(job_id: UUID, request: Request, session: Session = Depends(get_session)) -> dict:
    row = _job_or_404(session, job_id)
    return {
        "job": job_api(session, row, include_transitions=True),
        "request_id": request_id(request),
    }


@router.post("/jobs/{job_id}/cancel", status_code=202)
def cancel_job(
    job_id: UUID,
    request: Request,
    session: Session = Depends(get_session),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> JSONResponse:
    row = _job_or_404(session, job_id)
    actor = resolve_operator(request)
    key = (idempotency_key or "").strip()
    body = {"job_id": str(job_id), "action": "cancel"}
    if key:
        if len(key) < 8 or len(key) > 100:
            raise ApiError(422, "IDEMPOTENCY_KEY_INVALID", "Idempotency-Key must be 8-100 characters.")
        existing = get_idempotent(
            session, actor=actor, route=f"/api/v1/jobs/{job_id}/cancel", key=key, body=body
        )
        if existing is not None:
            return JSONResponse(
                status_code=existing.status_code,
                content=existing.response_json,
                headers={"Idempotency-Replayed": "true"},
            )
    status_code = 202
    if row.state in TERMINAL_STATES:
        status_code = 200
    elif row.state == "queued":
        changed = transition_job(
            session,
            job_id=row.id,
            from_states=("queued",),
            to_state="cancelled",
            reason="cancelled_before_start",
        )
        if changed:
            row.cancel_requested = True
            session.add(
                AuditRow(
                    id=new_id(),
                    actor=actor,
                    action="job_cancelled",
                    object_type="job",
                    object_id=row.id,
                    request_id=request_id(request),
                    metadata_json={"phase": "queued"},
                )
            )
    elif row.state == "running":
        row.cancel_requested = True
        transition_job(
            session,
            job_id=row.id,
            from_states=("running",),
            to_state="cancelling",
            reason="cancel_requested",
        )
        session.add(
            AuditRow(
                id=new_id(),
                actor=actor,
                action="job_cancel_requested",
                object_type="job",
                object_id=row.id,
                request_id=request_id(request),
                metadata_json={"phase": "running"},
            )
        )
    elif row.state == "cancelling":
        row.cancel_requested = True
    session.flush()
    session.refresh(row)
    payload = {
        "job": job_api(session, row),
        "request_id": request_id(request),
        "message": (
            "Cancellation is complete."
            if row.state in TERMINAL_STATES
            else "Cancellation requested. The worker settles the job after its current frame."
        ),
    }
    if key:
        store_idempotent(
            session,
            actor=actor,
            route=f"/api/v1/jobs/{job_id}/cancel",
            key=key,
            body=body,
            status_code=status_code,
            response=payload,
        )
    session.commit()
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/jobs/{job_id}/action-timeline")
def job_action_timeline(job_id: UUID, context: AppContext = Depends(get_context), session: Session = Depends(get_session)) -> JSONResponse:
    """Per-step shelf-action scores for a completed research job (served from local storage)."""
    import json as _json

    row = _job_or_404(session, job_id)
    if row.analysis_profile != ACTION_PROFILE:
        raise ApiError(404, "ACTION_TIMELINE_NOT_APPLICABLE", "This job did not run shelf-action analysis.")
    info = ((row.coverage or {}).get("action_analysis") or {}).get("timeline") or {}
    if row.state != "completed" or not info.get("path"):
        raise ApiError(409, "ACTION_TIMELINE_NOT_READY", "The action timeline is available after the job completes.", retryable=True)
    path = (context.settings.storage.root / "timelines" / f"{row.id}.json").resolve()
    if not path.is_file() or context.settings.storage.root not in path.parents:
        raise ApiError(410, "ACTION_TIMELINE_MISSING", "The stored action timeline is no longer available.")
    data = path.read_bytes()
    import hashlib as _hashlib

    if info.get("sha256") and _hashlib.sha256(data).hexdigest() != info["sha256"]:
        raise ApiError(500, "ACTION_TIMELINE_INTEGRITY", "The stored action timeline failed its integrity check.")
    return JSONResponse(content={"job_id": row.id, "timeline": _json.loads(data)}, headers={"Cache-Control": "private, max-age=60"})
