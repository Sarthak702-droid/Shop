"""Baseline event policy: hysteresis, null scores and honest quality flags."""

from __future__ import annotations

import pytest

from shop_core.contracts import TrackObservation, TrackStatus
from shop_core.events.policy import (
    EVENT_TYPE_PERSON_ACTIVITY,
    REASON_PERSON_TRACK,
    BaselineActivityPolicy,
)


def observation(
    track_id: str = "t0001",
    *,
    observed: bool = True,
    status: TrackStatus = TrackStatus.CONFIRMED,
    score: float = 0.9,
    pts_ms: int = 0,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        box=(0.0, 0.0, 10.0, 20.0),
        status=status,
        predicted=not observed,
        score=score,
        age_ms=pts_ms,
        hits=3,
        missed=0 if observed else 2,
        mean_observed_score=score,
        observed=observed,
    )


def policy(**overrides) -> BaselineActivityPolicy:
    values = {
        "version": "policy_test_v1",
        "sampling_fps": 5.0,
        "min_dwell_ms": 1000,
        "gap_split_ms": 3000,
        "cooldown_ms": 10000,
        "max_event_duration_ms": 60000,
        "max_events": 200,
    }
    values.update(overrides)
    return BaselineActivityPolicy(**values)


def ingest(
    p: BaselineActivityPolicy,
    pts_ms: int,
    tracks: list[TrackObservation],
    *,
    person_count: int = 1,
    source_discontinuity: bool = False,
):
    return p.ingest(
        pts_ms=pts_ms,
        tracks=tracks,
        person_count=person_count,
        model_bundle="bundle",
        source_discontinuity=source_discontinuity,
    )


def test_short_activity_does_not_emit_an_event() -> None:
    p = policy()
    for pts in (0, 200, 400, 600):
        assert ingest(p, pts, [observation(pts_ms=pts)]) == []
    assert p.flush(model_bundle="bundle", at_pts_ms=600) == []


def test_dwell_emits_descriptive_event_with_null_score() -> None:
    p = policy()
    for pts in (0, 200, 400, 600, 800, 1000):
        ingest(p, pts, [observation(pts_ms=pts)])
    events = p.flush(model_bundle="bundle", at_pts_ms=1000)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == EVENT_TYPE_PERSON_ACTIVITY
    assert event.reasons == [REASON_PERSON_TRACK]
    assert event.review_score is None
    assert event.score_semantics is None
    assert event.start_pts_ms == 0
    assert event.end_pts_ms == 1200
    assert event.track_ids == ["t0001"]
    assert event.quality["policy"] == "baseline_person_activity"
    assert event.quality["no_activity_score_available"] is True
    assert event.quality["observed_fraction"] == pytest.approx(1.0)
    assert event.required_evidence == ["person_detections", "source_coverage"]


def test_predicted_gaps_reduce_observed_fraction() -> None:
    p = policy(min_dwell_ms=0)
    for pts in (0, 200, 400):
        ingest(p, pts, [observation(pts_ms=pts)])
    for pts in (600, 800):
        ingest(p, pts, [observation(observed=False, pts_ms=pts)])
    ingest(p, 1000, [observation(pts_ms=1000)])
    events = p.flush(model_bundle="bundle", at_pts_ms=1000)
    assert len(events) == 1
    quality = events[0].quality
    assert quality["observed_samples"] == 4
    assert quality["predicted_gaps"] == 2
    assert quality["observed_fraction"] == pytest.approx(4 / 6, abs=1e-4)
    assert "intermittent_track_observations" in quality["warnings"]


def test_tentative_tracks_never_open_events() -> None:
    p = policy(min_dwell_ms=0)
    for pts in (0, 200, 400):
        ingest(p, pts, [observation(status=TrackStatus.TENTATIVE, pts_ms=pts)])
    assert p.flush(model_bundle="bundle") == []


def test_source_gap_splits_a_segment_and_emits_prior_event() -> None:
    p = policy(min_dwell_ms=0, gap_split_ms=500, cooldown_ms=0)
    ingest(p, 0, [observation(pts_ms=0)])
    ingest(p, 200, [observation(pts_ms=200)])
    events = ingest(p, 1000, [observation(pts_ms=1000)])
    assert len(events) == 1
    assert events[0].start_pts_ms == 0
    assert events[0].end_pts_ms == 400
    ingested = ingest(p, 1200, [observation(pts_ms=1200)])
    assert ingested == []
    events = p.flush(model_bundle="bundle")
    assert len(events) == 1
    assert events[0].start_pts_ms == 1200


def test_source_discontinuity_closes_segment() -> None:
    p = policy(min_dwell_ms=0)
    ingest(p, 0, [observation(pts_ms=0)])
    ingest(p, 200, [observation(pts_ms=200)])
    events = ingest(p, 400, [observation(pts_ms=400)], source_discontinuity=True)
    assert len(events) == 1
    assert events[0].quality["source_discontinuity"] is True
    assert "source_discontinuity" in events[0].quality["warnings"]


def test_cooldown_prevents_immediate_second_event() -> None:
    p = policy(min_dwell_ms=0, gap_split_ms=500, cooldown_ms=1000)
    ingest(p, 0, [observation(pts_ms=0)])
    ingest(p, 200, [observation(pts_ms=200)])
    events = ingest(p, 1000, [observation(pts_ms=1000)])
    assert len(events) == 1
    # The previous event ended at 400 ms. 1100 is still inside the cooldown.
    assert ingest(p, 1100, [observation(pts_ms=1100)]) == []
    assert ingest(p, 1200, [observation(pts_ms=1200)]) == []
    # 1400 - 400 = 1000 ms, so a new segment may open.
    ingest(p, 1400, [observation(pts_ms=1400)])
    events = p.flush(model_bundle="bundle")
    assert len(events) == 1
    assert events[0].start_pts_ms == 1400


def test_maximum_event_duration_forces_a_boundary() -> None:
    p = policy(min_dwell_ms=0, max_event_duration_ms=1000, cooldown_ms=0)
    emitted = []
    for pts in range(0, 1201, 200):
        emitted.extend(ingest(p, pts, [observation(pts_ms=pts)]))
    assert len(emitted) == 1
    assert emitted[0].start_pts_ms == 0
    assert emitted[0].end_pts_ms == 1200
    # Remaining activity opens a new segment which flush closes.
    rest = p.flush(model_bundle="bundle")
    assert len(rest) == 1
    assert rest[0].start_pts_ms == 1200


def test_crowding_warning_is_explicit() -> None:
    p = policy(min_dwell_ms=0)
    for pts in (0, 200, 400):
        ingest(p, pts, [observation(pts_ms=pts)], person_count=6)
    events = p.flush(model_bundle="bundle")
    assert "crowded_scene_identity_ambiguity" in events[0].quality["warnings"]
    assert events[0].quality["crowded_max_persons"] == 6


def test_event_cap_is_enforced() -> None:
    p = policy(min_dwell_ms=0, max_events=1)
    for pts in (0, 200, 400):
        track_a = observation("t0001", pts_ms=pts)
        track_b = observation("t0002", pts_ms=pts)
        ingest(p, pts, [track_a, track_b])
    events = p.flush(model_bundle="bundle")
    assert len(events) == 1


def test_active_segments_report_open_windows() -> None:
    p = policy()
    ingest(p, 0, [observation(pts_ms=0)])
    assert p.active_segments() == {"t0001": 0}
    p.flush(model_bundle="bundle")
    assert p.active_segments() == {}
