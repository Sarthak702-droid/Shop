"""Deterministic geometry tests: continuous coordinates, no +1 pixel bias."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shop_core.geometry import (
    box_area,
    box_center,
    box_from_center,
    clip_box,
    cosine_distance,
    cosine_similarity,
    intersection_area,
    iou,
    normalized_box,
    normalized_point_to_box_distance,
    point_in_polygon,
    polygon_area,
    temporal_iou,
    validate_box,
)


def test_iou_identical_and_disjoint() -> None:
    box = (10.0, 20.0, 30.0, 60.0)
    assert iou(box, box) == pytest.approx(1.0)
    assert iou(box, (100.0, 100.0, 120.0, 140.0)) == 0.0


def test_iou_matches_continuous_hand_calculation() -> None:
    # Intersection 5x5=25, union 100+100-25=175 -> 1/7
    assert iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0)) == pytest.approx(1 / 7)
    # Half-open area: touching edges have zero intersection.
    assert intersection_area((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_one_pixel_box_uses_continuous_area() -> None:
    # A 1x1 box has area 1.0, not 4.0 from the +1 pixel convention.
    assert box_area((4.0, 4.0, 5.0, 5.0)) == pytest.approx(1.0)
    assert iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_validate_box_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        validate_box((0, 0, 1))
    with pytest.raises(ValueError):
        validate_box((0, 0, 0, 5))
    with pytest.raises(ValueError):
        validate_box((0, 0, 5, 5, 6))
    with pytest.raises(ValueError):
        validate_box((0, 0, math.inf, 5))


def test_clip_box_and_degenerate_rejection() -> None:
    assert clip_box((-5, -5, 50, 50), 40, 30) == (0.0, 0.0, 40, 30)
    with pytest.raises(ValueError):
        clip_box((-10, -10, 0, 0), 40, 30)


def test_center_helpers_are_inverses() -> None:
    box = (10.0, 20.0, 30.0, 60.0)
    cx, cy = box_center(box)
    assert box_from_center(cx, cy, 20.0, 40.0) == pytest.approx(box)


def test_point_in_polygon_and_polygon_area() -> None:
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert point_in_polygon((5.0, 5.0), square) is True
    assert point_in_polygon((15.0, 5.0), square) is False
    assert point_in_polygon((5.0, 5.0), square[:2]) is False
    assert polygon_area(square) == pytest.approx(100.0)
    assert polygon_area(square[:2]) == 0.0


def test_point_to_box_distance_is_zero_inside() -> None:
    assert normalized_point_to_box_distance((5, 5), (0, 0, 10, 10), scale=10) == 0.0
    # 5 px outside a 10 px body scale -> 0.5 body widths.
    assert normalized_point_to_box_distance((15, 5), (0, 0, 10, 10), scale=10) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        normalized_point_to_box_distance((0, 0), (0, 0, 10, 10), scale=0)


def test_normalized_box_rejects_zero_dimensions() -> None:
    assert normalized_box((10, 5, 20, 15), 100, 50) == pytest.approx((0.1, 0.1, 0.2, 0.3))
    with pytest.raises(ValueError):
        normalized_box((0, 0, 1, 1), 0, 10)


def test_cosine_similarity_distinguishes_zero_norm_from_orthogonal() -> None:
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine_similarity(a, np.zeros(2)) is None
    assert cosine_distance(a, np.zeros(2)) is None
    assert cosine_distance(a, np.array([1.0, 0.0])) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cosine_similarity(a, np.array([1.0, 0.0, 0.0]))


def test_temporal_iou() -> None:
    assert temporal_iou(0, 1000, 0, 1000) == pytest.approx(1.0)
    assert temporal_iou(0, 1000, 2000, 3000) == 0.0
    assert temporal_iou(0, 2000, 1000, 3000) == pytest.approx(1000 / 3000)
    with pytest.raises(ValueError):
        temporal_iou(10, 10, 0, 1000)
