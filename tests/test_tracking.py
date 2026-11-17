"""Kalman filter and person tracker behavior, including elapsed-time gaps."""

from __future__ import annotations

import numpy as np
import pytest

from shop_core.contracts import Detection, TrackStatus
from shop_core.tracking.kalman import (
    KalmanBoxFilter,
    process_noise,
    transition_matrix,
)
from shop_core.tracking.tracker import PersonTracker, TrackerConfig


def detection(box: tuple[float, float, float, float], score: float = 0.9) -> Detection:
    return Detection(
        box=box,
        score=score,
        class_id=1,
        class_name="person",
        model_id="test_detector",
        model_version="v1",
    )


def test_transition_and_process_noise_shapes_and_errors() -> None:
    f = transition_matrix(0.5)
    assert f.shape == (8, 8)
    assert f[0, 4] == pytest.approx(0.5)
    q = process_noise(0.5, 2.0)
    assert q.shape == (8, 8)
    assert np.allclose(q, q.T)
    assert np.all(np.linalg.eigvalsh(q) >= -1e-9)
    with pytest.raises(ValueError):
        transition_matrix(-0.1)
    with pytest.raises(ValueError):
        process_noise(-0.1, 1.0)
    with pytest.raises(ValueError):
        process_noise(0.1, -1.0)


def test_kalman_prediction_moves_with_estimated_velocity() -> None:
    kalman = KalmanBoxFilter((0, 0, 20, 40), measurement_variance=1.0)
    for x in (20.0, 40.0, 60.0, 80.0):
        kalman.predict(0.1, spectral_density=1.0)
        kalman.update(np.array([x + 10.0, 20.0, 20.0, 40.0]))
    assert kalman.covariance_is_valid()
    before = kalman.box
    kalman.predict(0.1, spectral_density=1.0)
    after = kalman.box
    assert after[0] > before[0]
    assert after[0] - before[0] == pytest.approx(20.0, abs=6.0)


def test_kalman_rejects_invalid_variance_and_measurement() -> None:
    with pytest.raises(ValueError):
        KalmanBoxFilter((0, 0, 10, 10), measurement_variance=0)
    kalman = KalmanBoxFilter((0, 0, 10, 10), measurement_variance=1.0)
    with pytest.raises(ValueError):
        kalman.update(np.array([0.0, 0.0]))


def test_tracker_confirms_after_configured_hits() -> None:
    tracker = PersonTracker()
    box = (10.0, 10.0, 50.0, 90.0)
    statuses = []
    for pts in (0, 100, 200):
        update = tracker.update([detection(box)], pts)
        statuses.append(update.observations[0].status)
    assert statuses[0] is TrackStatus.TENTATIVE
    assert statuses[1] is TrackStatus.TENTATIVE
    assert statuses[2] is TrackStatus.CONFIRMED
    assert tracker.active_track_ids == ["t0001"]


def test_tracker_survives_short_gap_and_removes_after_horizon() -> None:
    tracker = PersonTracker()
    box = (10.0, 10.0, 50.0, 90.0)
    for pts in (0, 100, 200):
        tracker.update([detection(box)], pts)
    update = tracker.update([], 500)
    assert len(update.observations) == 1
    observation = update.observations[0]
    assert observation.status is TrackStatus.CONFIRMED
    assert observation.predicted is True
    assert observation.observed is False
    # 300 ms since last observation, still within the confirmed horizon.
    assert observation.missed == 1

    # Elapsed media time, not call count, drives loss.
    update = tracker.update([], 1400)
    assert update.observations[0].status is TrackStatus.LOST

    update = tracker.update([], 3000)
    assert update.observations == []
    assert update.removed_tracks == 1
    assert tracker.active_track_ids == []


def test_tentative_track_is_removed_quickly() -> None:
    tracker = PersonTracker()
    tracker.update([detection((0, 0, 20, 40))], 0)
    update = tracker.update([], 300)
    assert update.removed_tracks == 1
    assert tracker.active_track_ids == []


def test_tracker_does_not_match_a_gated_out_detection() -> None:
    tracker = PersonTracker()
    tracker.update([detection((0, 0, 20, 40))], 0)
    update = tracker.update([detection((500, 500, 520, 540))], 100)
    assert update.new_tracks == 1
    assert update.matched_tracks == 0
    assert len(update.observations) == 2


def test_tracker_predicts_motion_during_a_short_gap() -> None:
    tracker = PersonTracker()
    for index, pts in enumerate((0, 100, 200, 300)):
        tracker.update([detection((index * 20.0, 0.0, index * 20.0 + 20.0, 40.0))], pts)
    update = tracker.update([], 400)
    assert len(update.observations) == 1
    predicted = update.observations[0]
    assert predicted.predicted is True
    center_x = (predicted.box[0] + predicted.box[2]) / 2.0
    # Observed centers were 10, 30, 50, 70 at 10 fps; the next time step is ~90.
    assert center_x > 75.0


def test_discontinuity_starts_a_new_identity_namespace() -> None:
    tracker = PersonTracker()
    tracker.update([detection((0, 0, 20, 40))], 0)
    update = tracker.update([detection((0, 0, 20, 40))], 5000, discontinuity=True)
    assert [item.track_id for item in update.observations] == ["t0002"]
    assert tracker.active_track_ids == ["t0002"]


def test_tracker_rejects_invalid_configuration_and_time() -> None:
    with pytest.raises(ValueError):
        TrackerConfig(iou_weight=0.5, motion_weight=0.6)
    with pytest.raises(ValueError):
        TrackerConfig(gate_chi2=0)
    tracker = PersonTracker()
    with pytest.raises(ValueError):
        tracker.update([], -1)
    tracker.update([detection((0, 0, 20, 40))], 0)
    tracker.reset()
    assert tracker.active_track_ids == []
