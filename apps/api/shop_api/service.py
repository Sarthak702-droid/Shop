"""Application services shared by routes: status, serialization, idempotency."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from shop_core.config import Settings
from shop_core.persistence.db import Database
from shop_core.persistence.models import (
    AuditRow,
    EventRow,
    EvidenceRow,
    IdempotencyRow,
    JobRow,
    ReviewRow,
    SourceRow,
    UploadRow,
    utcnow,
)
from shop_core.persistence.repository import iso, job_counts, event_counts, job_queue_depth
from shop_core.registry import ModelStatus, host_profile, inspect_model, sha256_file, verify_cpu_runtime

APP_VERSION = "0.1.0"
ANALYSIS_PROFILE = "cpu_baseline_v1"
ACTION_PROFILE = "merl_shelf_actions_research_v1"
ANALYSIS_PROFILES = (ANALYSIS_PROFILE, ACTION_PROFILE)
MAX_PAGE_SIZE = 100


@dataclass
class AppContext:
    settings: Settings
    db: Database
    started_at: datetime = field(default_factory=utcnow)
    started_monotonic: float = field(default_factory=time.monotonic)
    _model_cache: dict[str, Any] | None = None
    _model_cache_at: float = 0.0

    def models(self) -> dict[str, ModelStatus]:
        now = time.monotonic()
        if self._model_cache is not None and now - self._model_cache_at < 5.0:
            return self._model_cache
        statuses = {
            name: inspect_model(self.settings, name)
            for name in ("person_detector", "retail_detector", "pose", "temporal")
        }
        self._model_cache = statuses
        self._model_cache_at = now
        return statuses

    def model_bundle(self) -> str:
        person = self.models()["person_detector"]
        return f"{person.model_id}:{person.version}|policy:{self.settings.policy.version}"

    def action_bundle(self) -> str:
        temporal = self.models()["temporal"]
        person = self.models()["person_detector"]
        return f"{temporal.model_id}:{temporal.version}|{person.model_id}:{person.version}|policy:merl_action_episodes_v1"[:100]

    def require_action_model(self) -> ModelStatus:
        status = self.models()["temporal"]
        if not status.available:
            from shop_api.errors import ApiError

            raise ApiError(
                503,
                "ACTION_MODEL_UNAVAILABLE",
                "Shelf-action research analysis is unavailable because its checkpoint failed its checks.",
                retryable=False,
                details={"reason": status.reason, "code": status.details.get("code")},
            )
        return status

    def require_person_model(self) -> ModelStatus:
        status = self.models()["person_detector"]
        if not status.available:
            from shop_api.errors import ApiError

            raise ApiError(
                503,
                "MODEL_UNAVAILABLE",
                "Person analysis is unavailable because the approved detector failed its checks.",
                retryable=False,
                details={"reason": status.reason, "code": status.details.get("code")},
            )
        return status


def worker_heartbeat(settings: Settings) -> dict[str, Any]:
    path = settings.storage.root / "worker-heartbeat.json"
    if not path.is_file():
        return {
            "status": "not_started",
            "fresh": False,
            "last_seen_at": None,
            "detail": "No worker heartbeat exists. Analysis jobs will remain queued.",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(str(payload["updated_at"]).replace("Z", "+00:00"))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {
            "status": "invalid",
            "fresh": False,
            "last_seen_at": None,
            "detail": "The worker heartbeat could not be read.",
        }
    age = (datetime.now(UTC) - updated.astimezone(UTC)).total_seconds()
    fresh = age <= 20.0
    raw_state = str(payload.get("state", "unknown"))
    status = raw_state if fresh else "stale"
    return {
        "status": status,
        "fresh": fresh,
        "last_seen_at": iso(updated),
        "age_seconds": round(max(0.0, age), 1),
        "worker_id": payload.get("worker_id"),
        "job_id": payload.get("job_id"),
        "reason": payload.get("reason"),
        "message": payload.get("message"),
        "detail": (
            "Analysis worker heartbeat is current."
            if fresh and raw_state not in {"blocked", "stopped"}
            else "Analysis worker is not currently available."
        ),
    }


def disk_status(settings: Settings) -> dict[str, Any]:
    usage = shutil.disk_usage(settings.storage.root)
    return {
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_fraction": round(usage.free / usage.total, 4) if usage.total else None,
        "minimum_free_bytes": settings.storage.min_free_bytes,
        "below_minimum": usage.free < settings.storage.min_free_bytes,
    }


def safe_media_path(settings: Settings, relative_path: str) -> Path | None:
    try:
        path = (settings.storage.root / relative_path).resolve()
    except (OSError, ValueError):
        return None
    try:
        path.relative_to(settings.storage.root.resolve())
    except ValueError:
        return None
    return path


def thumbnail_url(row: EventRow) -> str | None:
    if row.evidence_status != "available" or not row.evidence_id or row.deleted_at:
        return None
    return f"/api/v1/events/{row.id}/thumbnail"


def media_url(row: EventRow) -> str | None:
    if row.evidence_status != "available" or not row.evidence_id or row.deleted_at:
        return None
    return f"/api/v1/events/{row.id}/media"


def event_api(row: EventRow, *, include_reviews: bool = False) -> dict[str, Any]:
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
            "thumbnail_url": thumbnail_url(row),
            "media_url": media_url(row),
        },
        "model_bundle": row.model_bundle,
        "policy_version": row.policy_version,
        "review_state": row.review_state,
        "created_at": iso(row.created_at),
        "deleted_at": iso(row.deleted_at),
    }
    if include_reviews:
        payload["reviews"] = [
            review_api(review) for review in sorted(row.reviews, key=lambda item: item.created_at)
        ]
    return payload


def review_api(row: ReviewRow) -> dict[str, Any]:
    return {
        "review_id": row.id,
        "event_id": row.event_id,
        "actor": row.actor,
        "decision": row.decision,
        "reason_codes": row.reason_codes or [],
        "notes": row.notes,
        "action_feedback": row.action_feedback,
        "supersedes_review_id": row.supersedes_review_id,
        "created_at": iso(row.created_at),
    }


def active_source_job(session: Session, source_id: str) -> JobRow | None:
    return session.execute(
        select(JobRow)
        .where(
            JobRow.source_id == source_id,
            JobRow.state.in_(("queued", "running", "cancelling")),
        )
        .order_by(JobRow.created_at.desc(), JobRow.id.desc())
        .limit(1)
    ).scalar_one_or_none()


def source_api(row: SourceRow, settings: Settings, job: JobRow | None = None) -> dict[str, Any]:
    camera = settings.camera(row.camera_key) if row.camera_key else None
    derived_state = row.state
    active_job = None
    if job is not None:
        # Payload video/reviewer-facing state: a running job is actively using
        # the camera, while queued/cancelling work is reported without pretending
        # the source is already streaming.
        derived_state = "active" if job.state == "running" else job.state
        active_job = {
            "job_id": job.id,
            "status": job.state,
            "cancel_requested": job.cancel_requested,
        }
    return {
        "source_id": row.id,
        "kind": row.kind,
        "label": row.label,
        "camera_key": row.camera_key,
        "state": derived_state,
        "approved_camera": camera is not None,
        "active_job": active_job,
        "camera": (
            {"label": camera.label, "device_index": camera.device_index} if camera is not None else None
        ),
        "health": {
            "last_known_state": row.state,
            "live_checked": False,
            "preview_url": f"/api/v1/sources/{row.id}/preview" if row.kind == "webcam" else None,
        },
        "created_at": iso(row.created_at),
        "updated_at": iso(row.updated_at),
    }


def upload_api(row: UploadRow) -> dict[str, Any]:
    return {
        "upload_id": row.id,
        "filename": row.original_filename,
        "bytes": row.bytes,
        "sha256": row.sha256,
        "content_type": row.content_type,
        "status": row.status,
        "probe": row.probe or {},
        "created_at": iso(row.created_at),
        "deleted_at": iso(row.deleted_at),
    }


def event_count_for_job(session: Session, job_id: str) -> int:
    return int(
        session.scalar(
            select(func.count()).select_from(EventRow).where(
                EventRow.job_id == job_id, EventRow.deleted_at.is_(None)
            )
        )
        or 0
    )


def job_api(session: Session, row: JobRow, *, include_transitions: bool = False) -> dict[str, Any]:
    from shop_core.persistence.repository import job_public

    payload = job_public(row)
    payload["events_count"] = event_count_for_job(session, row.id)
    source_id = row.source_id
    source_label = None
    if source_id:
        source = session.get(SourceRow, source_id)
        source_label = source.label if source else None
    payload["source"]["label"] = source_label
    if include_transitions:
        from shop_core.persistence.models import JobTransitionRow

        transitions = session.execute(
            select(JobTransitionRow)
            .where(JobTransitionRow.job_id == row.id)
            .order_by(JobTransitionRow.created_at, JobTransitionRow.id)
        ).scalars().all()
        payload["transitions"] = [
            {
                "from_state": item.from_state,
                "to_state": item.to_state,
                "reason": item.reason,
                "created_at": iso(item.created_at),
            }
            for item in transitions
        ]
    return payload


def storage_and_queue(session: Session, settings: Settings, context: AppContext) -> dict[str, Any]:
    from shop_core.persistence.repository import job_counts

    worker = worker_heartbeat(settings)
    disk = disk_status(settings)
    return {
        "worker": worker,
        "disk": disk,
        "queue": {
            "depth": job_queue_depth(session),
            "active_limit": settings.runtime.active_jobs,
            "queued_limit": settings.runtime.queued_jobs,
            "counts_by_state": job_counts(session),
        },
        "models": {name: status.to_public() for name, status in context.models().items()},
    }


def body_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def get_idempotent(
    session: Session, *, actor: str, route: str, key: str, body: dict[str, Any]
) -> IdempotencyRow | None:
    existing = session.execute(
        select(IdempotencyRow).where(
            IdempotencyRow.actor == actor,
            IdempotencyRow.route == route,
            IdempotencyRow.key == key,
        )
    ).scalar_one_or_none()
    if existing is None:
        return None
    if existing.body_hash != body_hash(body):
        from shop_api.errors import ApiError

        raise ApiError(
            409,
            "IDEMPOTENCY_KEY_REUSED",
            "This Idempotency-Key was already used with a different request body.",
        )
    return existing


def store_idempotent(
    session: Session,
    *,
    actor: str,
    route: str,
    key: str,
    body: dict[str, Any],
    status_code: int,
    response: dict[str, Any],
) -> None:
    session.add(
        IdempotencyRow(
            actor=actor,
            route=route,
            key=key,
            body_hash=body_hash(body),
            response_json=response,
            status_code=status_code,
        )
    )


def model_capability_payload(context: AppContext) -> dict[str, Any]:
    statuses = context.models()
    return {
        "bundle": context.model_bundle(),
        "person": statuses["person_detector"].to_public(),
        "retail": statuses["retail_detector"].to_public(),
        "pose": statuses["pose"].to_public(),
        "temporal": statuses["temporal"].to_public(),
        "policy": {
            "version": context.settings.policy.version,
            "score_semantics": context.settings.policy.score_semantics,
            "baseline_event_type": "person_activity_observed",
            "no_activity_score_available": True,
            "detail": (
                "The baseline emits descriptive person-activity segments with a null review score. "
                "It is not a theft classifier and has no calibrated action probability."
            ),
        },
    }


def analysis_profiles_payload(context: AppContext) -> list[dict[str, Any]]:
    temporal = context.models()["temporal"]
    person = context.models()["person_detector"]
    return [
        {
            "profile": ANALYSIS_PROFILE,
            "label": "Person activity (baseline)",
            "available": person.available,
            "sources": ["file", "webcam"],
            "event_type": "person_activity_observed",
            "research_mode": False,
            "detail": "Detects and tracks people at 5 Hz and publishes descriptive dwell segments.",
        },
        {
            "profile": ACTION_PROFILE,
            "label": "Shelf actions (MERL research model)",
            "available": person.available and temporal.available,
            "sources": ["file"],
            "event_type": "shelf_interaction_observed",
            "research_mode": True,
            "classes": temporal.details.get("class_order"),
            "model": f"{temporal.model_id}:{temporal.version}",
            "reason": None if temporal.available else temporal.reason,
            "detail": (
                "Causal TCN over merl_features_v1 predicts five observable shelf interactions. Trained on one "
                "overhead mock-store camera; uncalibrated scores; never a theft or intent score."
            ),
        },
    ]


def capabilities_payload(context: AppContext) -> dict[str, Any]:
    settings = context.settings
    cpu = verify_cpu_runtime()
    return {
        "service": {"name": "shop-api", "version": APP_VERSION, "api_version": "v1"},
        "host": host_profile(),
        "cpu_verification": {
            "passed": cpu.passed,
            "checks": cpu.checks,
            "label": "actual_measured_result",
        },
        "sources": {
            "file_upload": {"available": True, "accepted_inputs": ["video containers decodable locally"]},
            "webcam": {
                "available": True,
                "approved_cameras": [
                    {"key": camera.key, "label": camera.label, "device_index": camera.device_index}
                    for camera in settings.security.approved_cameras
                ],
            },
            "rtsp": {"available": False, "reason": "RTSP is deferred to P2 and disabled in this build."},
            "audio": {"available": False, "reason": "Audio is disabled by policy."},
        },
        "models": model_capability_payload(context),
        "limits": {
            "active_jobs": settings.runtime.active_jobs,
            "queued_jobs": settings.runtime.queued_jobs,
            "max_upload_bytes": settings.storage.max_upload_bytes,
            "sampling_fps_max": settings.analysis.sampling_fps,
            "long_edge_px": settings.analysis.long_edge_px,
            "evidence_retention_days": settings.storage.evidence_days,
            "metadata_retention_days": settings.storage.metadata_days,
            "analysis_profile": ANALYSIS_PROFILE,
            "analysis_profiles": analysis_profiles_payload(context),
            "label": "proposed_starting_setting",
        },
        "security": {
            "bind": settings.security.bind_host,
            "bearer_token_configured": settings.api_token is not None,
            "reviewer_allowlist": list(settings.security.reviewers),
            "local_only": True,
        },
        "features": {
            "descriptive_activity_events": True,
            "human_review_required": True,
            "automatic_enforcement": False,
            "identity_recognition": False,
            "appearance_matching": False,
            "retail_object_detection": context.models()["retail_detector"].available,
            "pose": context.models()["pose"].available,
            "temporal_activity_model": context.models()["temporal"].available,
        },
    }


def readiness_payload(context: AppContext) -> tuple[bool, dict[str, Any]]:
    settings = context.settings
    checks: list[dict[str, Any]] = []
    try:
        with context.db.session() as session:
            session.execute(select(1))
            migrations = session.execute(
                select(func.count()).select_from(JobRow)
            ).scalar()
        checks.append(
            {
                "name": "database",
                "status": "pass",
                "detail": "SQLite connection, foreign keys and schema are queryable.",
                "observed": {"jobs": int(migrations or 0)},
            }
        )
    except Exception:
        checks.append(
            {
                "name": "database",
                "status": "fail",
                "detail": "SQLite could not be queried.",
                "observed": None,
            }
        )
    disk = disk_status(settings)
    checks.append(
        {
            "name": "storage",
            "status": "pass" if not disk["below_minimum"] else "fail",
            "detail": (
                "Storage has at least the configured free-space reserve."
                if not disk["below_minimum"]
                else "Free storage is below the configured reserve."
            ),
            "observed": {
                "free_bytes": disk["free_bytes"],
                "minimum_free_bytes": disk["minimum_free_bytes"],
            },
        }
    )
    person = context.models()["person_detector"]
    checks.append(
        {
            "name": "person_detector",
            "status": "pass" if person.available else "fail",
            "detail": person.reason,
            "observed": {
                "model_id": person.model_id,
                "version": person.version,
                "commercial_status": person.commercial_status,
            },
        }
    )
    worker = worker_heartbeat(settings)
    checks.append(
        {
            "name": "analysis_worker",
            "status": "pass" if worker["fresh"] and worker["status"] not in {"blocked", "stopped"} else "fail",
            "detail": worker["detail"],
            "observed": {
                "status": worker["status"],
                "last_seen_at": worker["last_seen_at"],
            },
        }
    )
    ready = all(item["status"] == "pass" for item in checks)
    return ready, {
        "status": "ready" if ready else "degraded",
        "ready": ready,
        "checks": checks,
        "worker": worker,
        "checked_at": iso(utcnow()),
    }


def metrics_payload(session: Session, context: AppContext) -> dict[str, Any]:
    settings = context.settings
    upload_count = int(session.scalar(select(func.count()).select_from(UploadRow).where(UploadRow.deleted_at.is_(None))) or 0)
    event_total = int(session.scalar(select(func.count()).select_from(EventRow).where(EventRow.deleted_at.is_(None))) or 0)
    reviews_today = int(
        session.scalar(
            select(func.count()).select_from(ReviewRow).where(
                ReviewRow.created_at >= datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            )
        )
        or 0
    )
    latest_events = session.execute(
        select(EventRow)
        .where(EventRow.deleted_at.is_(None))
        .options(selectinload(EventRow.evidence))
        .order_by(EventRow.created_at.desc(), EventRow.id.desc())
        .limit(5)
    ).scalars().all()
    worker = worker_heartbeat(settings)
    disk = disk_status(settings)
    return {
        "summary": {
            "uploads": upload_count,
            "events_total": event_total,
            "events_by_review_state": {
                "unreviewed": 0,
                "benign": 0,
                "review_needed": 0,
                "inconclusive": 0,
                **event_counts(session),
            },
            "reviews_today": reviews_today,
        },
        "jobs": {
            "queue_depth": job_queue_depth(session),
            "counts_by_state": job_counts(session),
            "active_limit": settings.runtime.active_jobs,
        },
        "worker": worker,
        "storage": disk,
        "models": {name: status.to_public() for name, status in context.models().items()},
        "recent_events": [event_api(row) for row in latest_events],
        "uptime_seconds": round(time.monotonic() - context.started_monotonic, 1),
        "label": "actual_measured_result",
    }
