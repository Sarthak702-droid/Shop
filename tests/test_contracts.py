"""Versioned frame, model, feature and event contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from shop_core.contracts import (
    ACTION_LABELS,
    COCO_KEYPOINT_ORDER,
    EVENT_SCHEMA_VERSION,
    FEATURE_COUNT,
    FEATURE_SAMPLING_HZ,
    FEATURE_VERSION,
    FEATURE_WINDOW_SAMPLES,
    FRAME_CONTRACT_VERSION,
    AnalysisMetrics,
    CoordinateTransform,
    Detection,
    EventRecord,
    EvidenceStatus,
    FrameEnvelope,
    TrackObservation,
    TrackStatus,
)


def make_frame(**overrides) -> FrameEnvelope:
    values = {
        "source_id": "source-1",
        "session_id": "session-1",
        "sequence": 0,
        "pts_ms": 0,
        "pts_provenance": "media_pts",
        "received_monotonic_ns": 1,
        "width": 4,
        "height": 3,
        "image_bgr": np.zeros((3, 4, 3), dtype=np.uint8),
    }
    values.update(overrides)
    return FrameEnvelope(**values)


def test_feature_contract_declares_145_values_over_29_samples_at_5hz() -> None:
    assert FEATURE_VERSION == "shop_features_v1"
    assert FEATURE_COUNT == 145
    assert FEATURE_WINDOW_SAMPLES == 29
    assert FEATURE_SAMPLING_HZ == 5.0
    assert len(ACTION_LABELS) == 5
    assert len(COCO_KEYPOINT_ORDER) == 17
    assert len(set(ACTION_LABELS)) == 5


def test_frame_envelope_accepts_valid_image_and_identity_transform() -> None:
    frame = make_frame()
    assert frame.contract_version == FRAME_CONTRACT_VERSION
    assert frame.color_space == "BGR"
    transform = frame.identity_transform()
    assert transform.to_original(1.0, 2.0) == (1.0, 2.0)


def test_frame_envelope_rejects_invalid_payloads() -> None:
    with pytest.raises(ValueError):
        make_frame(image_bgr=np.zeros((3, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        make_frame(image_bgr=np.zeros((3, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        make_frame(width=5)
    with pytest.raises(ValueError):
        make_frame(height=4)
    with pytest.raises(ValueError):
        make_frame(pts_ms=-1)


def test_coordinate_transform_round_trip() -> None:
    transform = CoordinateTransform(
        original_width=1920,
        original_height=1080,
        model_width=640,
        model_height=360,
        scale_x=0.5,
        scale_y=0.5,
        pad_x=12.0,
        pad_y=4.0,
        crop_origin_x=100.0,
        crop_origin_y=50.0,
    )
    model = transform.to_model(300.0, 200.0)
    back = transform.to_original(*model)
    assert back == pytest.approx((300.0, 200.0))


def test_detection_validation() -> None:
    detection = Detection(
        box=(1.0, 2.0, 3.0, 4.0),
        score=0.9,
        class_id=1,
        class_name="person",
        model_id="m",
        model_version="v1",
    )
    assert detection.role == "person"
    with pytest.raises(ValueError):
        Detection((1, 2, 1, 4), 0.9, 1, "person", "m", "v1")
    with pytest.raises(ValueError):
        Detection((1, 2, 3, 4), 1.5, 1, "person", "m", "v1")
    with pytest.raises(ValueError):
        Detection((1, 2, 3, 4), 0.5, 1, "person", "m", "v1", role="not-a-role")


def test_track_observation_predicted_flag_is_independent() -> None:
    observation = TrackObservation(
        track_id="t0001",
        box=(0, 0, 10, 20),
        status=TrackStatus.CONFIRMED,
        predicted=True,
        score=0.8,
        age_ms=500,
        hits=3,
        missed=1,
        mean_observed_score=0.7,
        observed=False,
    )
    assert observation.predicted is True
    assert observation.observed is False


def make_event(**overrides) -> EventRecord:
    values = {
        "event_id": "event-1",
        "source_id": "source-1",
        "session_id": "session-1",
        "job_id": "job-1",
        "event_type": "person_activity_observed",
        "reasons": ["person_track_observed"],
        "start_pts_ms": 0,
        "end_pts_ms": 6000,
        "track_ids": ["t0001"],
        "review_score": None,
        "score_semantics": None,
        "calibration_version": None,
        "quality": {},
        "required_evidence": ["person_detections"],
        "available_evidence": ["person_detections"],
        "model_bundle": "bundle",
        "policy_version": "policy_baseline_v1",
    }
    values.update(overrides)
    return EventRecord(**values)


def test_event_record_keeps_missing_score_null() -> None:
    event = make_event()
    assert event.review_score is None
    assert event.review_state == "unreviewed"
    assert event.schema_version == EVENT_SCHEMA_VERSION
    assert event.evidence == EvidenceStatus(status="pending")
    # Review outcome is a separate field and never inferred from a score.
    reviewed = make_event(review_state="benign")
    assert reviewed.review_score is None
    assert reviewed.review_state == "benign"


def test_event_record_allows_explicit_uncalibrated_score() -> None:
    event = make_event(review_score=0.42, score_semantics="uncalibrated_ranking")
    assert event.review_score == 0.42
    assert event.score_semantics == "uncalibrated_ranking"


def test_analysis_metrics_public_payload_is_labeled_measured() -> None:
    metrics = AnalysisMetrics(
        wall_time_ms=1000,
        analyzed_frames=10,
        decode_ms=[1.0, 2.0, 3.0],
    )
    public = metrics.to_public()
    assert public["label"] == "actual_measured_result"
    assert public["stage_timing"]["decode"]["samples"] == 3
    assert public["stage_timing"]["detect"] is None

    broken = AnalysisMetrics()
    public = broken.to_public()
    assert public["stage_timing"]["decode"] is None
    assert public["analyzed_frames"] == 0
