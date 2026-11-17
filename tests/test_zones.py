"""Privacy-mask and analysis-ROI behavior."""

from __future__ import annotations

import numpy as np

from shop_core.zones import apply_privacy_mask, inside_analysis_roi, mask_contains


def test_privacy_mask_blackens_only_the_configured_normalized_region() -> None:
    image = np.full((100, 200, 3), 220, dtype=np.uint8)
    polygon = [[0.0, 0.0], [0.25, 0.0], [0.25, 1.0], [0.0, 1.0]]
    result = apply_privacy_mask(image, polygon)
    assert result is image
    assert np.all(image[:, :49] == 0)
    assert np.all(image[:, 51:] == 220)
    assert mask_contains(image, polygon, (10, 50)) is True
    assert mask_contains(image, polygon, (150, 50)) is False


def test_privacy_mask_accepts_polygon_and_is_noop_when_empty() -> None:
    image = np.full((40, 40, 3), 7, dtype=np.uint8)
    assert apply_privacy_mask(image, []) is image
    assert np.all(image == 7)
    diamond = [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]]
    apply_privacy_mask(image, diamond)
    assert tuple(image[20, 20]) == (0, 0, 0)
    assert tuple(image[0, 0]) == (7, 7, 7)


def test_analysis_roi_keeps_detections_by_center() -> None:
    roi = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    assert inside_analysis_roi((10, 10, 30, 30), roi, 100, 100) is True
    assert inside_analysis_roi((70, 10, 90, 30), roi, 100, 100) is False
    assert inside_analysis_roi((70, 10, 90, 30), [], 100, 100) is True


def test_empty_or_short_polygons_are_safe() -> None:
    image = np.full((10, 10, 3), 5, dtype=np.uint8)
    apply_privacy_mask(image, [[0.0, 0.0], [1.0, 0.0]])
    assert np.all(image == 5)
    assert mask_contains(image, [], (1, 1)) is False
