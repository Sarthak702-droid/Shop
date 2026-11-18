"""Durable job state machine, recovery policy and schema constraints."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from shop_core.persistence.db import Database
from shop_core.persistence.migrate import database_head, upgrade_database
from shop_core.persistence.models import EventRow, JobRow, JobTransitionRow, ReviewRow, UploadRow, utcnow
from shop_core.persistence.repository import (
    claim_next_job,
    decode_cursor,
    encode_cursor,
    new_id,
    recover_interrupted_jobs,
    transition_job,
)


@pytest.fixture()
def db(settings):
    database = Database(settings)
    upgrade_database(database.engine)
    yield database
    database.dispose()


def add_upload(session, **overrides) -> UploadRow:
    values = {
        "id": new_id(),
        "owner": "test",
        "original_filename": "clip.avi",
        "stored_path": "uploads/clip.avi",
        "sha256": "0" * 64,
        "bytes": 10,
        "status": "ready",
        "probe": {"duration_ms": 1000},
    }
    values.update(overrides)
    row = UploadRow(**values)
    session.add(row)
    session.flush()
    return row


def add_job(session, upload_id: str, **overrides) -> JobRow:
    values = {
        "id": new_id(),
        "kind": "file",
        "upload_id": upload_id,
        "state": "queued",
        "analysis_profile": "cpu_baseline_v1",
        "sampling_fps": 5.0,
        "privacy_mask_version": "zones_default_v1",
        "retain_evidence": True,
        "model_bundle": "bundle",
        "policy_version": "policy_baseline_v1",
        "progress": {},
        "metrics": {},
        "coverage": {},
    }
    values.update(overrides)
    row = JobRow(**values)
    session.add(row)
    session.flush()
    return row


def test_migration_installs_schema_and_version(db) -> None:
    head = database_head(db.engine)
    assert head is not None
    assert head == "b7c1d2e3f4a5"
    with db.session() as session:
        assert session.execute(select(UploadRow)).scalars().all() == []


def test_claim_only_one_worker_wins(db) -> None:
    with db.session() as session:
        upload = add_upload(session)
        job = add_job(session, upload.id)
    with db.session() as session:
        claimed = claim_next_job(session, worker_id="worker-a")
        assert claimed is not None
        assert claimed.id == job.id
        assert claimed.state == "running"
        assert claimed.lease_owner == "worker-a"
        assert claim_next_job(session, worker_id="worker-b") is None
    with db.session() as session:
        transitions = session.execute(
            select(JobTransitionRow).where(JobTransitionRow.job_id == job.id)
        ).scalars().all()
        assert [(item.from_state, item.to_state) for item in transitions] == [("queued", "running")]


def test_conditional_terminal_transition_cannot_run_twice(db) -> None:
    with db.session() as session:
        upload = add_upload(session)
        job = add_job(session, upload.id, state="running")
    with db.session() as session:
        assert transition_job(
            session, job_id=job.id, from_states=("running",), to_state="completed", reason="done"
        )
        assert not transition_job(
            session, job_id=job.id, from_states=("running",), to_state="cancelled", reason="late"
        )
    with db.session() as session:
        row = session.get(JobRow, job.id)
        assert row.state == "completed"
        assert row.finished_at is not None


def test_cancel_before_start_is_terminal(db) -> None:
    with db.session() as session:
        upload = add_upload(session)
        job = add_job(session, upload.id)
    with db.session() as session:
        assert transition_job(
            session,
            job_id=job.id,
            from_states=("queued",),
            to_state="cancelled",
            reason="cancelled_before_start",
        )
    with db.session() as session:
        row = session.get(JobRow, job.id)
        assert row.state == "cancelled"
        assert row.finished_at is not None


def test_recovery_fails_closed_without_replay(db) -> None:
    stale = utcnow() - timedelta(minutes=5)
    with db.session() as session:
        upload = add_upload(session)
        running = add_job(
            session,
            upload.id,
            state="running",
            lease_owner="dead-worker",
            lease_expires_at=stale,
        )
        cancelling = add_job(
            session,
            upload.id,
            state="cancelling",
            cancel_requested=True,
            lease_owner="dead-worker",
            lease_expires_at=stale,
        )
        queued = add_job(session, upload.id)
        live = add_job(
            session,
            upload.id,
            state="running",
            lease_owner="live-worker",
            lease_expires_at=utcnow() + timedelta(minutes=1),
        )
    with db.session() as session:
        recovered = recover_interrupted_jobs(session)
    assert set(recovered) == {running.id, cancelling.id}
    with db.session() as session:
        failed = session.get(JobRow, running.id)
        cancelled = session.get(JobRow, cancelling.id)
        assert failed.state == "failed"
        assert failed.error_code == "WORKER_INTERRUPTED"
        assert "replay" in failed.error_message.lower()
        assert cancelled.state == "cancelled"
        assert session.get(JobRow, queued.id).state == "queued"
        assert session.get(JobRow, live.id).state == "running"


def test_cursor_round_trip() -> None:
    created_at = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=UTC)
    cursor = encode_cursor(created_at, "row-id")
    decoded_at, decoded_id = decode_cursor(cursor)
    assert decoded_at == created_at
    assert decoded_id == "row-id"
    with pytest.raises(Exception):
        decode_cursor("not-a-cursor")


def test_schema_rejects_invalid_job_and_event_rows(db) -> None:
    with db.session() as session:
        with pytest.raises(IntegrityError):
            session.add(
                JobRow(
                    id=new_id(),
                    kind="file",
                    state="queued",
                    analysis_profile="cpu_baseline_v1",
                    sampling_fps=5.0,
                    privacy_mask_version="zones_default_v1",
                    retain_evidence=True,
                    model_bundle="bundle",
                    policy_version="policy",
                    progress={},
                    metrics={},
                    coverage={},
                )
            )
            session.flush()
        session.rollback()

    with db.session() as session:
        upload = add_upload(session)
        job = add_job(session, upload.id)
        session.add(
            EventRow(
                id=new_id(),
                job_id=job.id,
                source_id="source",
                session_id="session",
                event_type="person_activity_observed",
                reasons=[],
                start_pts_ms=0,
                end_pts_ms=1000,
                track_ids=[],
                review_score=1.5,
                quality={},
                required_evidence=[],
                available_evidence=[],
                model_bundle="bundle",
                policy_version="policy",
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()

    with db.session() as session:
        upload = add_upload(session)
        job = add_job(session, upload.id)
        event = EventRow(
            id=new_id(),
            job_id=job.id,
            source_id="source",
            session_id="session",
            event_type="person_activity_observed",
            reasons=[],
            start_pts_ms=0,
            end_pts_ms=1000,
            track_ids=[],
            review_score=None,
            quality={},
            required_evidence=[],
            available_evidence=[],
            model_bundle="bundle",
            policy_version="policy",
        )
        session.add(event)
        session.flush()
        session.add(
            ReviewRow(
                id=new_id(),
                event_id=event.id,
                actor="local-reviewer",
                decision="guilty",
                reason_codes=[],
                notes="not a supported outcome",
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()
