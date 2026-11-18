"""Versioned data contracts shared by sources, models, policy, API and review UI.

These types are deliberately framework-light. The API layer converts them to
Pydantic models; storage converts them to rows. Media time is always integer
milliseconds relative to the source session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np

FRAME_CONTRACT_VERSION = "shop_frame_v1"
EVENT_SCHEMA_VERSION = "1.0"
FEATURE_VERSION = "shop_features_v1"
FEATURE_COUNT = 145
FEATURE_WINDOW_SAMPLES = 29
FEATURE_SAMPLING_HZ = 5.0
ACTION_LABELS = (
    "reach_to_shelf",
    "product_pickup_observed",
    "return_to_shelf",
    "hand_near_bag",
    "item_visibility_lost",
)
COCO_KEYPOINT_ORDER = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

PtsProvenance = Literal["media_pts", "frame_index_nominal_fps", "monotonic_capture"]


@dataclass(frozen=True)
class CoordinateTransform:
    """Invertible mapping from model/crop coordinates to original pixels."""

    original_width: int
    original_height: int
    model_width: int
    model_height: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0
    crop_origin_x: float = 0.0
    crop_origin_y: float = 0.0
    version: str = "identity_v1"

    def to_original(self, x: float, y: float) -> tuple[float, float]:
        ox = (x - self.pad_x) / self.scale_x + self.crop_origin_x
        oy = (y - self.pad_y) / self.scale_y + self.crop_origin_y
        return ox, oy

    def to_model(self, x: float, y: float) -> tuple[float, float]:
        mx = (x - self.crop_origin_x) * self.scale_x + self.pad_x
        my = (y - self.crop_origin_y) * self.scale_y + self.pad_y
        return mx, my


@dataclass
class FrameEnvelope:
    """One decoded frame plus its timestamp provenance.

    Ownership: ``image_bgr`` is a uint8 HxWx3 array owned by the producer for
    the duration of the processing callback. Consumers must copy before
    retaining it beyond the callback.
    """

    source_id: str
    session_id: str
    sequence: int
    pts_ms: int
    pts_provenance: PtsProvenance
    received_monotonic_ns: int
    width: int
    height: int
    image_bgr: np.ndarray
    captured_at_utc: str | None = None
    color_space: Literal["BGR"] = "BGR"
    transform: CoordinateTransform | None = None
    valid: bool = True
    discontinuity: bool = False
    decode_ms: float | None = None
    contract_version: str = FRAME_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.image_bgr.dtype != np.uint8 or self.image_bgr.ndim != 3:
            raise ValueError("frame image must be uint8 HxWx3")
        h, w = self.image_bgr.shape[:2]
        if (h, w) != (self.height, self.width):
            raise ValueError("frame dimensions do not match the envelope")
        if self.pts_ms < 0:
            raise ValueError("media timestamp cannot be negative")

    def identity_transform(self) -> CoordinateTransform:
        return CoordinateTransform(
            original_width=self.width,
            original_height=self.height,
            model_width=self.width,
            model_height=self.height,
        )


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    model_id: str
    model_version: str
    role: Literal["person", "retail"] = "person"

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.box
        if not all(np.isfinite(v) for v in self.box):
            raise ValueError("detection coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("detection box must have positive area")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError("detection score must be in [0, 1]")
        if self.role not in {"person", "retail"}:
            raise ValueError("detection role must be person or retail")


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


@dataclass
class TrackObservation:
    """A tracker output at one analyzed media timestamp."""

    track_id: str
    box: tuple[float, float, float, float]
    status: TrackStatus
    predicted: bool
    score: float
    age_ms: int
    hits: int
    missed: int
    mean_observed_score: float
    observed: bool = True


@dataclass(frozen=True)
class MediaProbe:
    """Actual decoder observations, not the filename or container header alone."""

    container: str
    codec: str
    width: int
    height: int
    nominal_fps: float | None
    frame_count: int | None
    duration_ms: int | None
    first_frame_decoded: bool
    pts_provenance: PtsProvenance
    probe_backend: str
    notes: list[str] = field(default_factory=list)


@dataclass
class AnalysisMetrics:
    """Measured values for one run. Never populated with publisher metadata."""

    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    wall_time_ms: int = 0
    media_duration_ms: int | None = None
    analyzed_frames: int = 0
    expected_samples: int | None = None
    dropped_frames: int = 0
    decode_gaps: int = 0
    detection_failures: int = 0
    person_detections: int = 0
    confirmed_tracks: int = 0
    decode_ms: list[float] = field(default_factory=list)
    preprocess_ms: list[float] = field(default_factory=list)
    detect_ms: list[float] = field(default_factory=list)
    track_ms: list[float] = field(default_factory=list)
    evidence_ms: list[float] = field(default_factory=list)
    peak_rss_kb: int = 0
    inference_threads: int = 0
    analysis_coverage: float | None = None
    valid_analysis_fps: float | None = None
    real_time_factor: float | None = None

    def to_public(self) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float] | None:
            if not values:
                return None
            ordered = sorted(values)
            p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
            return {
                "mean_ms": round(sum(ordered) / len(ordered), 3),
                "p50_ms": round(ordered[len(ordered) // 2], 3),
                "p95_ms": round(ordered[p95_index], 3),
                "samples": len(ordered),
            }

        return {
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "wall_time_ms": self.wall_time_ms,
            "media_duration_ms": self.media_duration_ms,
            "analyzed_frames": self.analyzed_frames,
            "expected_samples": self.expected_samples,
            "dropped_frames": self.dropped_frames,
            "decode_gaps": self.decode_gaps,
            "detection_failures": self.detection_failures,
            "person_detections": self.person_detections,
            "confirmed_tracks": self.confirmed_tracks,
            "stage_timing": {
                "decode": stats(self.decode_ms),
                "preprocess": stats(self.preprocess_ms),
                "detect": stats(self.detect_ms),
                "track": stats(self.track_ms),
                "evidence": stats(self.evidence_ms),
            },
            "peak_rss_kb": self.peak_rss_kb,
            "inference_threads": self.inference_threads,
            "analysis_coverage": self.analysis_coverage,
            "valid_analysis_fps": self.valid_analysis_fps,
            "real_time_factor": self.real_time_factor,
            "label": "actual_measured_result",
        }


@dataclass(frozen=True)
class EvidenceStatus:
    status: Literal["available", "unavailable", "pending", "failed", "deleted", "expired"]
    reason: str | None = None
    clip_id: str | None = None
    frame_count: int | None = None
    duration_ms: int | None = None


@dataclass
class EventRecord:
    """Descriptive activity segment for human review.

    ``review_score`` stays ``None`` when no validated score exists. A missing
    score is never coerced to zero, and no field represents criminal guilt.
    """

    event_id: str
    source_id: str
    session_id: str
    job_id: str
    event_type: str
    reasons: list[str]
    start_pts_ms: int
    end_pts_ms: int
    track_ids: list[str]
    review_score: float | None
    score_semantics: str | None
    calibration_version: str | None
    quality: dict[str, Any]
    required_evidence: list[str]
    available_evidence: list[str]
    model_bundle: str
    policy_version: str
    schema_version: str = EVENT_SCHEMA_VERSION
    review_state: Literal["unreviewed", "benign", "review_needed", "inconclusive"] = "unreviewed"
    evidence: EvidenceStatus = field(default_factory=lambda: EvidenceStatus(status="pending"))
    created_at_utc: str = ""
    # Research shelf-action segments (``shelf_interaction_observed`` events only).
    actions: list[dict[str, Any]] = field(default_factory=list)
