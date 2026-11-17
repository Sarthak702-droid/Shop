"""Worker entry point: claims durable jobs and runs CPU inference.

Run separately from the API:

    .venv/bin/python -m shop_worker

The API never executes model inference in its request handlers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from shop_core.config import Settings, load_settings
from shop_core.contracts import EventRecord
from shop_core.detection.person import PersonDetector
from shop_core.evidence.writer import EvidenceClip, EvidenceWriter
from shop_core.persistence.db import Database
from shop_core.persistence.models import AuditRow, EventRow, EvidenceRow, JobRow, SourceRow, UploadRow, utcnow
from shop_core.persistence.repository import (
    claim_next_job,
    evidence_retention_deadline,
    iso,
    job_public,
    new_id,
    recover_interrupted_jobs,
    transition_job,
)
from shop_core.pipeline.runner import (
    PipelineSink,
    run_file_analysis,
    run_webcam_analysis,
)
from shop_core.registry import ModelUnavailable
from shop_core.sources.media import WebcamSource

LOG_FORMAT = '{"time":"%(asctime)s","level":"%(levelname)s","service":"shop-worker","message":"%(message)s"}'


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def write_worker_heartbeat(settings: Settings, payload: dict) -> None:
    path = settings.storage.root / "worker-heartbeat.json"
    payload = {**payload, "updated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


class DatabaseSink(PipelineSink):
    def __init__(
        self,
        db: Database,
        settings: Settings,
        *,
        job_id: str,
        worker_id: str,
        retain_evidence: bool,
    ) -> None:
        self.db = db
        self.settings = settings
        self.job_id = job_id
        self.worker_id = worker_id
        self.retain_evidence = retain_evidence
        self._cancel_checked_at = 0.0
        self._cancelled = False
        self._last_heartbeat = 0.0

    def _touch_lease(self, session) -> None:  # noqa: ANN001
        session.execute(
            JobRow.__table__.update()
            .where(JobRow.id == self.job_id)
            .values(
                lease_expires_at=utcnow() + timedelta(seconds=45),
                heartbeat_at=utcnow(),
            )
        )

    def should_cancel(self) -> bool:
        now = time.monotonic()
        if now - self._cancel_checked_at < 1.0:
            return self._cancelled
        self._cancel_checked_at = now
        with self.db.session() as session:
            row = session.execute(
                select(JobRow.cancel_requested, JobRow.state).where(JobRow.id == self.job_id)
            ).one_or_none()
            if row is None:
                self._cancelled = True
            else:
                cancel_requested, state = row
                self._cancelled = bool(cancel_requested) or state in {"cancelling", "cancelled"}
        return self._cancelled

    def report_progress(self, progress: dict) -> None:
        with self.db.session() as session:
            session.execute(
                JobRow.__table__.update()
                .where(JobRow.id == self.job_id)
                .values(progress=progress, heartbeat_at=utcnow())
            )
        self._heartbeat_file("running", progress)

    def event_started(self, record: EventRecord) -> None:
        with self.db.session() as session:
            exists = session.get(EventRow, record.event_id)
            if exists is not None:
                return
            session.add(
                EventRow(
                    id=record.event_id,
                    job_id=self.job_id,
                    source_id=record.source_id,
                    session_id=record.session_id,
                    event_type=record.event_type,
                    reasons=list(record.reasons),
                    start_pts_ms=record.start_pts_ms,
                    end_pts_ms=record.end_pts_ms,
                    track_ids=list(record.track_ids),
                    review_score=record.review_score,
                    score_semantics=record.score_semantics,
                    calibration_version=record.calibration_version,
                    quality=record.quality,
                    required_evidence=list(record.required_evidence),
                    available_evidence=list(record.available_evidence),
                    model_bundle=record.model_bundle,
                    policy_version=record.policy_version,
                    schema_version=record.schema_version,
                    review_state="unreviewed",
                    evidence_status="pending",
                )
            )
            session.add(
                AuditRow(
                    id=new_id(),
                    actor="system",
                    action="event_created",
                    object_type="event",
                    object_id=record.event_id,
                    request_id=None,
                    metadata_json={"job_id": self.job_id, "policy_version": record.policy_version},
                )
            )

    def event_evidence_ready(
        self, event_id: str, clip: EvidenceClip | None, reason: str | None
    ) -> None:
        with self.db.session() as session:
            event = session.get(EventRow, event_id)
            if event is None:
                return
            if clip is None:
                status = "unavailable" if reason and "RETENTION_DISABLED" in reason else "failed"
                event.evidence_status = status
                event.evidence_reason = (reason or "Evidence unavailable.")[:200]
                session.add(
                    AuditRow(
                        id=new_id(),
                        actor="system",
                        action="evidence_unavailable",
                        object_type="event",
                        object_id=event_id,
                        request_id=None,
                        metadata_json={"reason": event.evidence_reason},
                    )
                )
                return
            if not self.retain_evidence:
                clip.path.unlink(missing_ok=True)
                clip.path.with_suffix(".jpg").unlink(missing_ok=True)
                event.evidence_status = "unavailable"
                event.evidence_reason = "EVIDENCE_RETENTION_DISABLED: clip was generated but not retained."
                return
            object_path = clip.path.relative_to(self.settings.storage.root).as_posix()
            thumbnail_path = clip.path.with_suffix(".jpg").relative_to(self.settings.storage.root).as_posix()
            evidence = EvidenceRow(
                id=clip.clip_id,
                event_id=event_id,
                object_path=object_path,
                thumbnail_path=thumbnail_path,
                sha256=clip.sha256,
                bytes=clip.bytes,
                mime_type=clip.mime_type,
                width=clip.width,
                height=clip.height,
                fps=clip.fps,
                duration_ms=clip.duration_ms,
                frame_count=clip.frame_count,
                start_pts_ms=clip.start_pts_ms,
                end_pts_ms=clip.end_pts_ms,
                encoder=clip.encoder,
                generate_ms=clip.generate_ms,
                retain_until=evidence_retention_deadline(self.settings.storage.evidence_days),
            )
            session.add(evidence)
            event.evidence_status = "available"
            event.evidence_id = clip.clip_id
            event.evidence_reason = None
            session.add(
                AuditRow(
                    id=new_id(),
                    actor="system",
                    action="evidence_published",
                    object_type="event",
                    object_id=event_id,
                    request_id=None,
                    metadata_json={"clip_id": clip.clip_id, "sha256": clip.sha256},
                )
            )

    def heartbeat(self) -> None:
        with self.db.session() as session:
            self._touch_lease(session)
        self._heartbeat_file("running", None)

    def _heartbeat_file(self, state: str, progress: dict | None) -> None:
        now = time.monotonic()
        if progress is not None and now - self._last_heartbeat < 4.0:
            return
        self._last_heartbeat = now
        write_worker_heartbeat(
            self.settings,
            {
                "worker_id": self.worker_id,
                "state": state,
                "job_id": self.job_id,
                "media_pts_ms": (progress or {}).get("media_pts_ms"),
            },
        )


def _resolve_job_source(session, job: JobRow) -> tuple[Path | None, dict]:
    if job.upload_id:
        upload = session.get(UploadRow, job.upload_id)
        if upload is None or upload.status != "ready" or upload.deleted_at is not None:
            raise ValueError("UPLOAD_UNAVAILABLE")
        return Path(upload.stored_path), upload.probe or {}
    source = session.get(SourceRow, job.source_id) if job.source_id else None
    if source is None or source.kind != "webcam":
        raise ValueError("SOURCE_UNAVAILABLE")
    return None, source.config or {}


def _file_source_for_upload(session, upload_id: str) -> str:
    existing = session.execute(
        select(SourceRow).where(SourceRow.kind == "file", SourceRow.camera_key == upload_id)
    ).scalar_one_or_none()
    if existing is not None:
        return existing.id
    upload = session.get(UploadRow, upload_id)
    filename = upload.original_filename if upload is not None else "file upload"
    source = SourceRow(
        id=new_id(),
        store_id="local-store",
        kind="file",
        label=filename[:100],
        camera_key=upload_id,
        config={"upload_id": upload_id},
        state="idle",
    )
    session.add(source)
    session.flush()
    return source.id


def run_job(db: Database, settings: Settings, job_id: str, worker_id: str, detector: PersonDetector) -> None:
    with db.session() as session:
        job = session.get(JobRow, job_id)
        if job is None or job.state != "running":
            return
        upload_path, source_info = _resolve_job_source(session, job)
        if upload_path is not None:
            upload_path = settings.storage.root / upload_path
        persist_evidence = bool(job.retain_evidence)
        sampling_fps = job.sampling_fps
        job_kind = job.kind
        job_source_id = job.source_id
        upload_id = job.upload_id

    sink = DatabaseSink(
        db,
        settings,
        job_id=job_id,
        worker_id=worker_id,
        retain_evidence=persist_evidence,
    )
    writer = EvidenceWriter(settings.evidence_dir)
    session_id = str(uuid.uuid4())
    result = None
    try:
        if job_kind == "file":
            if upload_path is None or upload_id is None:
                raise ValueError("UPLOAD_UNAVAILABLE")
            with db.session() as session:
                upload = session.get(UploadRow, upload_id)
                if upload is None:
                    raise ValueError("UPLOAD_UNAVAILABLE")
                duration_ms = (upload.probe or {}).get("duration_ms")
                source_id = _file_source_for_upload(session, upload_id)
            result = run_file_analysis(
                settings=settings,
                detector=detector,
                path=upload_path,
                duration_ms=duration_ms,
                job_id=job_id,
                source_id=source_id,
                session_id=session_id,
                sink=sink,
                evidence_writer=writer,
                generate_evidence=persist_evidence,
                sampling_fps=sampling_fps,
            )
        else:
            camera_key = source_info.get("camera_key")
            camera_config = settings.camera(str(camera_key)) if camera_key else None
            if camera_config is None or job_source_id is None:
                raise ValueError("CAMERA_NOT_APPROVED")
            camera = WebcamSource(
                camera_config.device_index,
                source_id=str(job_source_id),
                session_id=session_id,
                queue_frames=settings.analysis.live_queue_frames,
            )
            result = run_webcam_analysis(
                settings=settings,
                detector=detector,
                camera=camera,
                job_id=job_id,
                source_id=str(job_source_id),
                session_id=session_id,
                sink=sink,
                evidence_writer=writer,
                generate_evidence=persist_evidence,
                sampling_fps=sampling_fps,
            )
    except ValueError as error:
        code = str(error)
        _finish_failed(db, job_id, code, "The job source could not be resolved for analysis.")
        return
    except Exception as error:  # noqa: BLE001 - worker must settle the job
        logging.exception("job execution crashed")
        _finish_failed(
            db,
            job_id,
            getattr(error, "code", "WORKER_JOB_FAILED"),
            getattr(error, "message", "The worker could not complete this job."),
        )
        return

    if result is None:
        _finish_failed(db, job_id, "WORKER_JOB_FAILED", "The worker returned no result.")
        return

    metrics = result.metrics.to_public()
    with db.session() as session:
        job = session.get(JobRow, job_id)
        if job is None:
            return
        job.metrics = metrics
        job.coverage = result.coverage
        if result.notes:
            job.progress = {**(job.progress or {}), "notes": result.notes}
        job.lease_owner = None
        job.lease_expires_at = None
        job.heartbeat_at = utcnow()

    cancelled = sink.should_cancel()
    if result.status == "failed":
        _finish_failed(db, job_id, result.error_code or "PIPELINE_FAILED", result.error_message or "Analysis failed.")
        return
    if cancelled or result.status == "cancelled":
        with db.session() as session:
            transition_job(session, job_id=job_id, from_states=("running", "cancelling"), to_state="cancelled", reason="cancelled")
            session.add(
                AuditRow(
                    id=new_id(),
                    actor="system",
                    action="job_cancelled",
                    object_type="job",
                    object_id=job_id,
                    request_id=None,
                    metadata_json={"events_created": result.events_created},
                )
            )
        logging.info("job %s cancelled", job_id)
        return
    with db.session() as session:
        changed = transition_job(
            session,
            job_id=job_id,
            from_states=("running",),
            to_state="completed",
            reason="analysis_completed",
        )
        if not changed:
            current = session.get(JobRow, job_id)
            if current is not None and current.cancel_requested:
                transition_job(
                    session,
                    job_id=job_id,
                    from_states=("running", "cancelling"),
                    to_state="cancelled",
                    reason="cancel_won_before_completion",
                )
        session.add(
            AuditRow(
                id=new_id(),
                actor="system",
                action="job_completed",
                object_type="job",
                object_id=job_id,
                request_id=None,
                metadata_json={
                    "events_created": result.events_created,
                    "evidence_failures": result.evidence_failures,
                    "analysis_coverage": result.metrics.analysis_coverage,
                },
            )
        )
    logging.info(
        "job %s completed events=%s evidence_failures=%s wall_ms=%s",
        job_id,
        result.events_created,
        result.evidence_failures,
        result.metrics.wall_time_ms,
    )


def _finish_failed(db: Database, job_id: str, code: str, message: str) -> None:
    with db.session() as session:
        changed = transition_job(
            session,
            job_id=job_id,
            from_states=("running", "cancelling"),
            to_state="failed",
            reason=code,
        )
        if changed:
            job = session.get(JobRow, job_id)
            if job is not None:
                job.error_code = code[:64]
                job.error_message = message[:500]
                job.lease_owner = None
                job.lease_expires_at = None
            session.add(
                AuditRow(
                    id=new_id(),
                    actor="system",
                    action="job_failed",
                    object_type="job",
                    object_id=job_id,
                    request_id=None,
                    metadata_json={"code": code[:64]},
                )
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shop CPU inference worker")
    parser.add_argument("--config", default=None)
    parser.add_argument("--once", action="store_true", help="process at most one queued job then exit")
    parser.add_argument("--job-id", default=None, help="run a specific queued job")
    args = parser.parse_args(argv)
    configure_logging()
    settings = load_settings(args.config)
    settings.ensure_directories()
    db = Database(settings)
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    logging.info("worker %s starting", worker_id)

    try:
        detector = PersonDetector.load(settings)
    except ModelUnavailable as error:
        logging.error("person detector unavailable: %s %s", error.code, error.message)
        write_worker_heartbeat(
            settings,
            {"worker_id": worker_id, "state": "blocked", "reason": error.code, "message": error.message},
        )
        return 3

    write_worker_heartbeat(settings, {"worker_id": worker_id, "state": "idle", "job_id": None})
    with db.session() as session:
        recovered = recover_interrupted_jobs(session)
    if recovered:
        logging.warning("recovered interrupted jobs as failed: %s", ", ".join(recovered))

    running = True

    def stop(_signum, _frame) -> None:  # noqa: ANN001
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        claimed: JobRow | None = None
        with db.session() as session:
            if args.job_id:
                job = session.get(JobRow, args.job_id)
                if job is not None and job.state == "queued":
                    from shop_core.persistence.repository import transition_job as _transition

                    _transition(session, job_id=job.id, from_states=("queued",), to_state="running", reason="worker_claim")
                    job.started_at = utcnow()
                    job.lease_owner = worker_id
                    job.lease_expires_at = utcnow() + timedelta(seconds=45)
                    job.heartbeat_at = utcnow()
                    claimed = job
            else:
                claimed = claim_next_job(session, worker_id=worker_id, lease_seconds=45)
        if claimed is None:
            if args.once or args.job_id:
                break
            write_worker_heartbeat(settings, {"worker_id": worker_id, "state": "idle", "job_id": None})
            time.sleep(1.0)
            continue
        write_worker_heartbeat(settings, {"worker_id": worker_id, "state": "running", "job_id": claimed.id})
        logging.info("claimed job %s", claimed.id)
        run_job(db, settings, claimed.id, worker_id, detector)
        write_worker_heartbeat(settings, {"worker_id": worker_id, "state": "idle", "job_id": None})
        if args.once or args.job_id:
            break
    write_worker_heartbeat(settings, {"worker_id": worker_id, "state": "stopped", "job_id": None})
    db.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
