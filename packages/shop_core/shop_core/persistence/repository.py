"""Durable state transitions and serialization helpers.

All media writes happen before the database row that references them becomes
visible. State changes use conditional updates so a cancellation cannot race a
completion into two terminal outcomes.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from shop_core.persistence.models import (
    AuditRow,
    EventRow,
    EvidenceRow,
    JobRow,
    JobTransitionRow,
    ReviewRow,
    SourceRow,
    UploadRow,
    utcnow,
)


def new_id() -> str:
    return str(uuid.uuid4())


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def audit(
    session: Session,
    *,
    actor: str,
    action: str,
    object_type: str,
    object_id: str | None,
    request_id: str | None,
    metadata: dict | None = None,
) -> None:
    session.add(
        AuditRow(
            id=new_id(),
            actor=actor,
            action=action,
            object_type=object_type,
            object_id=object_id,
            request_id=request_id,
            metadata_json=metadata or {},
        )
    )


def transition_job(
    session: Session,
    *,
    job_id: str,
    from_states: tuple[str, ...],
    to_state: str,
    reason: str | None = None,
) -> bool:
    job = session.get(JobRow, job_id)
    if job is None or job.state not in from_states:
        return False
    previous = job.state
    result = session.execute(
        update(JobRow)
        .where(JobRow.id == job_id, JobRow.state == previous)
        .values(state=to_state, finished_at=utcnow() if to_state in {"completed", "failed", "cancelled"} else job.finished_at)
    )
    if result.rowcount != 1:
        return False
    session.add(
        JobTransitionRow(
            job_id=job_id,
            from_state=previous,
            to_state=to_state,
            reason=reason,
        )
    )
    return True


def upload_public(row: UploadRow) -> dict[str, Any]:
    return {
        "upload_id": row.id,
        "filename": row.original_filename,
        "bytes": row.bytes,
        "sha256": row.sha256,
        "content_type": row.content_type,
        "status": row.status,
        "probe": row.probe,
        "created_at": iso(row.created_at),
    }


def source_public(row: SourceRow) -> dict[str, Any]:
    return {
        "source_id": row.id,
        "kind": row.kind,
        "label": row.label,
        "camera_key": row.camera_key,
        "state": row.state,
        "created_at": iso(row.created_at),
        "updated_at": iso(row.updated_at),
    }


def job_public(row: JobRow, *, include_transitions: bool = False) -> dict[str, Any]:
    payload = {
        "job_id": row.id,
        "kind": row.kind,
        "status": row.state,
        "source": {
            "kind": row.kind,
            "upload_id": row.upload_id,
            "source_id": row.source_id,
        },
        "analysis_profile": row.analysis_profile,
        "sampling_fps": row.sampling_fps,
        "privacy_mask_version": row.privacy_mask_version,
        "retain_evidence": row.retain_evidence,
        "progress": row.progress or {},
        "metrics": row.metrics or {},
        "coverage": row.coverage or {},
        "model_bundle": row.model_bundle,
        "policy_version": row.policy_version,
        "error": (
            {"code": row.error_code, "message": row.error_message}
            if row.error_code
            else None
        ),
        "cancel_requested": row.cancel_requested,
        "created_at": iso(row.created_at),
        "started_at": iso(row.started_at),
        "finished_at": iso(row.finished_at),
    }
    if include_transitions:
        payload["transitions"] = []
    return payload


def event_public(row: EventRow, *, include_reviews: bool = False) -> dict[str, Any]:
    payload = {
        "schema_version": row.schema_version,
        "event_id": row.id,
        "source_id": row.source_id,
        "session_id": row.session_id,
        "job_id": row.job_id,
        "track_ids": row.track_ids or [],
        "media_interval_ms": {"start": row.start_pts_ms, "end": row.end_pts_ms},
        "event_type": row.event_type,
        "reasons": row.reasons or [],
        "review_score": row.review_score,
        "score_semantics": row.score_semantics,
        "calibration_version": row.calibration_version,
        "quality": row.quality or {},
        "required_evidence": row.required_evidence or [],
        "available_evidence": row.available_evidence or [],
        "actions": row.actions or [],
        "evidence": {
            "status": row.evidence_status,
            "clip_id": row.evidence_id,
            "reason": row.evidence_reason,
        },
        "model_bundle": row.model_bundle,
        "policy_version": row.policy_version,
        "review_state": row.review_state,
        "created_at": iso(row.created_at),
    }
    if include_reviews:
        payload["reviews"] = [review_public(review) for review in sorted(row.reviews, key=lambda r: r.created_at)]
    return payload


def review_public(row: ReviewRow) -> dict[str, Any]:
    return {
        "review_id": row.id,
        "event_id": row.event_id,
        "actor": row.actor,
        "decision": row.decision,
        "reason_codes": row.reason_codes or [],
        "notes": row.notes,
        "supersedes_review_id": row.supersedes_review_id,
        "created_at": iso(row.created_at),
    }


def evidence_public(row: EvidenceRow | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "clip_id": row.id,
        "state": "deleted" if row.deleted_at else "available",
        "bytes": row.bytes,
        "sha256": row.sha256,
        "mime_type": row.mime_type,
        "width": row.width,
        "height": row.height,
        "fps": row.fps,
        "duration_ms": row.duration_ms,
        "frame_count": row.frame_count,
        "media_interval_ms": {"start": row.start_pts_ms, "end": row.end_pts_ms},
        "encoder": row.encoder,
        "retain_until": iso(row.retain_until),
        "created_at": iso(row.created_at),
    }


def encode_cursor(created_at: datetime, row_id: str) -> str:
    payload = json.dumps(
        {"created_at": iso(created_at), "id": row_id}, separators=(",", ":")
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, str]:
    padded = cursor + "=" * (-len(cursor) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded))
    created_at = datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00"))
    return created_at, str(payload["id"])


def job_queue_depth(session: Session) -> int:
    count = session.scalar(
        select(func.count()).select_from(JobRow).where(JobRow.state.in_(("queued", "running", "cancelling")))
    )
    return int(count or 0)


def job_counts(session: Session) -> dict[str, int]:
    rows = session.execute(select(JobRow.state, func.count()).group_by(JobRow.state)).all()
    return {state: int(count) for state, count in rows}


def event_counts(session: Session) -> dict[str, int]:
    rows = session.execute(select(EventRow.review_state, func.count()).group_by(EventRow.review_state)).all()
    return {state: int(count) for state, count in rows}


def recover_interrupted_jobs(session: Session, *, lease_seconds: int = 30) -> list[str]:
    """Explicit restart policy: abandoned runs fail closed and are not replayed.

    Re-running detection from the start would duplicate events. A failed job is
    visible to the operator and can be replayed as a new job with a new session.
    """
    now = utcnow()
    stale = session.execute(
        select(JobRow).where(
            JobRow.state.in_(("running", "cancelling")),
            (JobRow.lease_expires_at.is_(None)) | (JobRow.lease_expires_at < now),
        )
    ).scalars().all()
    recovered: list[str] = []
    for job in stale:
        target = "cancelled" if job.state == "cancelling" else "failed"
        ok = transition_job(
            session,
            job_id=job.id,
            from_states=(job.state,),
            to_state=target,
            reason="WORKER_INTERRUPTED: abandoned lease recovered at startup; replay as a new job.",
        )
        if ok:
            job.error_code = "WORKER_INTERRUPTED"
            job.error_message = (
                "The worker stopped before this job finished. Partial coverage may exist; "
                "start a new job to replay the recording."
            )
            job.cancel_requested = target == "cancelled"
            job.lease_owner = None
            job.lease_expires_at = None
            session.add(
                AuditRow(
                    id=new_id(),
                    actor="system",
                    action="job_recovered",
                    object_type="job",
                    object_id=job.id,
                    request_id=None,
                    metadata_json={"policy": "fail_closed_no_duplicate_replay"},
                )
            )
            recovered.append(job.id)
    return recovered


def claim_next_job(session: Session, *, worker_id: str, lease_seconds: int = 30) -> JobRow | None:
    row = session.execute(
        select(JobRow).where(JobRow.state == "queued").order_by(JobRow.created_at, JobRow.id).limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    now = utcnow()
    result = session.execute(
        update(JobRow)
        .where(JobRow.id == row.id, JobRow.state == "queued")
        .values(
            state="running",
            started_at=now,
            lease_owner=worker_id,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            heartbeat_at=now,
        )
    )
    if result.rowcount != 1:
        return None
    session.add(
        JobTransitionRow(job_id=row.id, from_state="queued", to_state="running", reason="worker_claim")
    )
    session.flush()
    session.refresh(row)
    return row


def evidence_retention_deadline(days: int) -> datetime:
    return utcnow() + timedelta(days=days)
