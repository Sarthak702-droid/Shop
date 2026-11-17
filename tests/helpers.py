"""Test fixtures shared by the API integration tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from shop_core.config import Settings
from shop_core.persistence.models import (
    EventRow,
    EvidenceRow,
    JobRow,
    SourceRow,
    UploadRow,
    utcnow,
)
from shop_core.persistence.repository import new_id


def write_evidence_files(settings: Settings, event_id: str) -> tuple[str, str, bytes, bytes]:
    """Create a small placeholder evidence pair on local managed storage."""
    folder = settings.storage.root / "evidence" / event_id
    folder.mkdir(parents=True, exist_ok=True)
    clip_bytes = b"\x1aE\xdf\xa3shop-test-webm-placeholder"
    thumb_bytes = b"\xff\xd8\xff\xe0shop-test-jpeg-placeholder\xff\xd9"
    clip = folder / "clip.webm"
    thumb = folder / "clip.jpg"
    clip.write_bytes(clip_bytes)
    thumb.write_bytes(thumb_bytes)
    return (
        clip.relative_to(settings.storage.root).as_posix(),
        thumb.relative_to(settings.storage.root).as_posix(),
        clip_bytes,
        thumb_bytes,
    )


def seed_event(
    app,  # noqa: ANN001
    *,
    review_state: str = "unreviewed",
    evidence: bool = True,
    retain_until: datetime | None = None,
    created_at: datetime | None = None,
    label: str = "seed.avi",
    event_type: str = "person_activity_observed",
) -> dict:
    """Insert one completed file job and its descriptive event."""
    context = app.state.context
    settings = context.settings
    event_id = new_id()
    clip_bytes = b""
    thumb_bytes = b""
    with context.db.session() as session:
        upload = UploadRow(
            id=new_id(),
            owner="test",
            original_filename=label,
            stored_path="uploads/seed.avi",
            sha256="0" * 64,
            bytes=1,
            status="ready",
            probe={"duration_ms": 6000, "width": 96, "height": 64},
        )
        session.add(upload)
        session.flush()
        job = JobRow(
            id=new_id(),
            kind="file",
            upload_id=upload.id,
            state="completed",
            analysis_profile="cpu_baseline_v1",
            sampling_fps=5.0,
            privacy_mask_version="zones_default_v1",
            retain_evidence=True,
            model_bundle="test_bundle",
            policy_version="policy_baseline_v1",
            progress={"stage": "completed"},
            metrics={},
            coverage={"analysis_coverage": 1.0},
            finished_at=created_at or utcnow(),
        )
        session.add(job)
        session.flush()
        source = SourceRow(
            id=new_id(),
            store_id="local-store",
            kind="file",
            label=label,
            camera_key=upload.id,
            config={"upload_id": upload.id},
            state="idle",
        )
        session.add(source)
        session.flush()
        event = EventRow(
            id=event_id,
            job_id=job.id,
            source_id=source.id,
            session_id=new_id(),
            event_type=event_type,
            reasons=["person_track_observed"],
            start_pts_ms=0,
            end_pts_ms=6000,
            track_ids=["t0001"],
            review_score=None,
            score_semantics=None,
            calibration_version=None,
            quality={
                "policy": "baseline_person_activity",
                "observed_samples": 30,
                "observed_duration_ms": 6000,
                "observed_fraction": 1.0,
                "predicted_gaps": 0,
                "crowded_max_persons": 1,
                "source_discontinuity": False,
                "no_activity_score_available": True,
                "warnings": [],
            },
            required_evidence=["person_detections", "source_coverage"],
            available_evidence=["person_detections", "track_observations"],
            model_bundle="test_bundle",
            policy_version="policy_baseline_v1",
            review_state=review_state,
            created_at=created_at or utcnow(),
        )
        session.add(event)
        if evidence:
            clip_id = new_id()
            object_path, thumbnail_path, clip_bytes, thumb_bytes = write_evidence_files(
                settings, event_id
            )
            session.add(
                EvidenceRow(
                    id=clip_id,
                    event_id=event_id,
                    object_path=object_path,
                    thumbnail_path=thumbnail_path,
                    sha256="a" * 64,
                    bytes=len(clip_bytes),
                    mime_type="video/webm",
                    width=96,
                    height=64,
                    fps=10.0,
                    duration_ms=2000,
                    frame_count=20,
                    start_pts_ms=0,
                    end_pts_ms=2000,
                    encoder="VP80",
                    generate_ms=1.5,
                    retain_until=retain_until,
                )
            )
            event.evidence_status = "available"
            event.evidence_id = clip_id
        else:
            event.evidence_status = "unavailable"
            event.evidence_reason = "EVIDENCE_UNAVAILABLE: no retained clip."
        session.flush()
    return {
        "event_id": event_id,
        "job_id": job.id,
        "clip_bytes": clip_bytes,
        "thumb_bytes": thumb_bytes,
    }
