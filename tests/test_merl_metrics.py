import numpy as np
import pytest

from ml.merl.dataset import step_labels
from ml.merl.metrics import average_precision, best_f1_threshold, f1_at, runs, segment_detection, tiou


def test_average_precision_perfect_and_worst():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert average_precision(y * 1.0, y) == pytest.approx(1.0)
    worst = average_precision(1.0 - y, y)
    assert 0 < worst < 0.6


def test_average_precision_known_value():
    # ranking: 1(pos) 2(neg) 3(pos) -> AP = (1/1 + 2/3) / 2
    assert average_precision(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1])) == pytest.approx((1 + 2 / 3) / 2)


def test_no_positives_is_nan():
    assert np.isnan(average_precision(np.array([0.1, 0.2]), np.array([0, 0])))


def test_threshold_and_f1():
    s = np.array([0.9, 0.8, 0.3, 0.2])
    y = np.array([1, 1, 0, 0])
    thr, f1 = best_f1_threshold(s, y)
    assert f1 == pytest.approx(1.0) and 0.3 < thr <= 0.8
    assert f1_at(s, y, thr)["f1"] == pytest.approx(1.0)


def test_runs_and_iou():
    assert runs(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]
    assert runs(np.array([])) == []
    assert tiou((0, 9), (5, 14)) == pytest.approx(5 / 15)
    assert tiou((0, 1), (3, 4)) == 0.0


def test_segment_detection_matches_once_per_ground_truth():
    labels = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0])
    scores = np.array([0, .9, .9, .9, 0, 0, .1, .1, 0])
    res = segment_detection([(scores, labels)], threshold=0.5)
    assert (res.predicted, res.ground_truth) == (1, 2)
    assert res.precision == 1.0 and res.recall == 0.5 and res.ap == pytest.approx(0.5)


def test_step_labels_use_centre_frame():
    frames = np.array([0, 6, 12])  # centres 3, 9, 15
    y = step_labels(frames, [(0, 8, 10), (4, 0, 3)])
    assert y[:, 0].tolist() == [0, 1, 0]
    assert y[:, 4].tolist() == [1, 0, 0]
