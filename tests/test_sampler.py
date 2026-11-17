"""Sampling by media PTS, gap accounting and discontinuity resets."""

from __future__ import annotations

import numpy as np
import pytest

from shop_core.contracts import FrameEnvelope
from shop_core.scheduling.sampler import SamplingScheduler


def frame(pts_ms: int, *, valid: bool = True) -> FrameEnvelope:
    return FrameEnvelope(
        source_id="source",
        session_id="session",
        sequence=pts_ms,
        pts_ms=pts_ms,
        pts_provenance="media_pts",
        received_monotonic_ns=pts_ms * 1_000_000,
        width=4,
        height=3,
        image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
        valid=valid,
    )


def test_scheduler_accepts_only_frames_at_or_after_due_timestamp() -> None:
    scheduler = SamplingScheduler(5.0)
    assert scheduler.accept(frame(0)) is True
    assert scheduler.accept(frame(100)) is False
    assert scheduler.accept(frame(200)) is True
    assert scheduler.accept(frame(300)) is False
    assert scheduler.accept(frame(400)) is True
    assert scheduler.stats.accepted == 3
    assert scheduler.stats.skipped_schedule == 2
    assert scheduler.stats.missing_samples == 0
    assert scheduler.coverage == pytest.approx(1.0)


def test_scheduler_counts_missing_samples_and_coverage() -> None:
    scheduler = SamplingScheduler(5.0)
    assert scheduler.accept(frame(0)) is True
    assert scheduler.accept(frame(400)) is True
    assert scheduler.accept(frame(800)) is True
    assert scheduler.stats.accepted == 3
    assert scheduler.stats.missing_samples == 2
    assert scheduler.stats.expected_samples == 5
    assert scheduler.coverage == pytest.approx(0.6)
    assert scheduler.stats.first_pts_ms == 0
    assert scheduler.stats.last_pts_ms == 800


def test_long_gap_is_flagged_for_tracker_reset() -> None:
    scheduler = SamplingScheduler(5.0, gap_reset_ms=2500)
    assert scheduler.accept(frame(0)) is True
    late = frame(3000)
    assert scheduler.accept(late) is True
    assert late.discontinuity is True
    assert scheduler.stats.discontinuity_events == 1
    assert scheduler.stats.missing_samples == 14
    assert scheduler.stats.expected_samples == 16


def test_invalid_frames_are_not_counted_as_samples() -> None:
    scheduler = SamplingScheduler(5.0)
    assert scheduler.accept(frame(0, valid=False)) is False
    assert scheduler.stats.accepted == 0
    assert scheduler.stats.expected_samples is None
    assert scheduler.coverage is None


def test_first_frame_timestamp_may_start_later_than_zero() -> None:
    scheduler = SamplingScheduler(5.0)
    assert scheduler.accept(frame(1200)) is True
    assert scheduler.accept(frame(1399)) is False
    assert scheduler.accept(frame(1400)) is True
    assert scheduler.stats.missing_samples == 0
    assert scheduler.stats.first_pts_ms == 1200


def test_scheduler_validates_fps() -> None:
    with pytest.raises(ValueError):
        SamplingScheduler(0)
    with pytest.raises(ValueError):
        SamplingScheduler(-5)
