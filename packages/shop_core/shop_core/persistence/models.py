"""SQLAlchemy models for durable local state.

Media paths are server-generated and stored relative to the storage root.
Review history is append-only. Job terminal states are guarded by a CHECK
constraint that mirrors the application state machine.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class SourceRow(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    store_id: Mapped[str] = mapped_column(String(36), default="local-store", nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    camera_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    state: Mapped[str] = mapped_column(String(32), default="idle", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    jobs: Mapped[list["JobRow"]] = relationship(back_populates="source")

    __table_args__ = (
        CheckConstraint("kind IN ('file','webcam','rtsp')", name="ck_sources_kind"),
        CheckConstraint("state IN ('idle','active','degraded','disconnected')", name="ck_sources_state"),
    )


class UploadRow(Base):
    __tablename__ = "uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner: Mapped[str] = mapped_column(String(100), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="validating", nullable=False)
    probe: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    jobs: Mapped[list["JobRow"]] = relationship(back_populates="upload")

    __table_args__ = (
        CheckConstraint("status IN ('validating','ready','rejected','deleted')", name="ck_uploads_status"),
        CheckConstraint("bytes > 0", name="ck_uploads_bytes"),
        Index("ix_uploads_created", "created_at", "id"),
    )


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    upload_id: Mapped[str | None] = mapped_column(ForeignKey("uploads.id"), nullable=True)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.id"), nullable=True)
    state: Mapped[str] = mapped_column(String(24), default="queued", nullable=False)
    analysis_profile: Mapped[str] = mapped_column(String(100), nullable=False)
    sampling_fps: Mapped[float] = mapped_column(Float, nullable=False)
    privacy_mask_version: Mapped[str] = mapped_column(String(100), nullable=False)
    retain_evidence: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    model_bundle: Mapped[str] = mapped_column(String(100), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(100), nullable=False)
    progress: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    coverage: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(100), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    upload: Mapped[UploadRow | None] = relationship(back_populates="jobs")
    source: Mapped[SourceRow | None] = relationship(back_populates="jobs")
    events: Mapped[list["EventRow"]] = relationship(back_populates="job")

    __table_args__ = (
        CheckConstraint(
            "(upload_id IS NOT NULL AND source_id IS NULL) OR (upload_id IS NULL AND source_id IS NOT NULL)",
            name="ck_jobs_one_source",
        ),
        CheckConstraint("kind IN ('file','webcam')", name="ck_jobs_kind"),
        CheckConstraint(
            "state IN ('queued','running','cancelling','completed','failed','cancelled')",
            name="ck_jobs_state",
        ),
        CheckConstraint("sampling_fps > 0", name="ck_jobs_sampling"),
        Index("ix_jobs_state_created", "state", "created_at"),
        Index("ix_jobs_created", "created_at", "id"),
    )


class JobTransitionRow(Base):
    __tablename__ = "job_transitions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    from_state: Mapped[str | None] = mapped_column(String(24), nullable=True)
    to_state: Mapped[str] = mapped_column(String(24), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    __table_args__ = (Index("ix_job_transitions_job", "job_id", "created_at"),)


class EventRow(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    source_id: Mapped[str] = mapped_column(String(36), nullable=False)
    session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    reasons: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    start_pts_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_pts_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    track_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    review_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_semantics: Mapped[str | None] = mapped_column(String(64), nullable=True)
    calibration_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    quality: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    required_evidence: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    available_evidence: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    # Research shelf-action segments inside this event (empty for person-activity events).
    actions: Mapped[list] = mapped_column(JSON, default=list, nullable=False, server_default="[]")
    model_bundle: Mapped[str] = mapped_column(String(100), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(100), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(16), default="1.0", nullable=False)
    review_state: Mapped[str] = mapped_column(String(24), default="unreviewed", nullable=False)
    evidence_status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False)
    evidence_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)
    evidence_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job: Mapped[JobRow] = relationship(back_populates="events")
    evidence: Mapped["EvidenceRow | None"] = relationship(back_populates="event", uselist=False)
    reviews: Mapped[list["ReviewRow"]] = relationship(back_populates="event")

    __table_args__ = (
        CheckConstraint("end_pts_ms > start_pts_ms", name="ck_events_interval"),
        CheckConstraint("start_pts_ms >= 0", name="ck_events_start"),
        CheckConstraint(
            "review_score IS NULL OR (review_score >= 0 AND review_score <= 1)",
            name="ck_events_score",
        ),
        CheckConstraint(
            "review_state IN ('unreviewed','benign','review_needed','inconclusive')",
            name="ck_events_review_state",
        ),
        CheckConstraint(
            "evidence_status IN ('pending','available','unavailable','failed','deleted','expired')",
            name="ck_events_evidence_status",
        ),
        Index("ix_events_created", "created_at", "id"),
        Index("ix_events_job", "job_id", "created_at"),
        Index("ix_events_review", "review_state", "created_at"),
    )


class EvidenceRow(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(
        ForeignKey("events.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    object_path: Mapped[str] = mapped_column(String(512), nullable=False)
    thumbnail_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    fps: Mapped[float] = mapped_column(Float, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_count: Mapped[int] = mapped_column(Integer, nullable=False)
    start_pts_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    end_pts_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    encoder: Mapped[str] = mapped_column(String(64), nullable=False)
    generate_ms: Mapped[float] = mapped_column(Float, nullable=False)
    retain_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    event: Mapped[EventRow] = relationship(back_populates="evidence")

    __table_args__ = (
        CheckConstraint("bytes > 0", name="ck_evidence_bytes"),
        CheckConstraint("end_pts_ms > start_pts_ms", name="ck_evidence_interval"),
        Index("ix_evidence_retention", "retain_until"),
    )


class ReviewRow(Base):
    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), nullable=False)
    actor: Mapped[str] = mapped_column(String(100), nullable=False)
    decision: Mapped[str] = mapped_column(String(24), nullable=False)
    reason_codes: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Reviewer judgement of predicted shelf actions: {"verdict", "observed_actions"}.
    action_feedback: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    supersedes_review_id: Mapped[str | None] = mapped_column(
        ForeignKey("reviews.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    event: Mapped[EventRow] = relationship(back_populates="reviews")
    supersedes: Mapped["ReviewRow | None"] = relationship(remote_side=[id])

    __table_args__ = (
        CheckConstraint(
            "decision IN ('benign','review_needed','inconclusive')", name="ck_reviews_decision"
        ),
        CheckConstraint("length(notes) <= 2000", name="ck_reviews_notes"),
        Index("ix_reviews_event", "event_id", "created_at"),
    )


class AuditRow(Base):
    __tablename__ = "audit"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    object_type: Mapped[str] = mapped_column(String(64), nullable=False)
    object_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    __table_args__ = (Index("ix_audit_created", "created_at", "id"),)


class IdempotencyRow(Base):
    __tablename__ = "idempotency"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(100), nullable=False)
    route: Mapped[str] = mapped_column(String(200), nullable=False)
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("actor", "route", "key", name="uq_idempotency_actor_route_key"),
    )
