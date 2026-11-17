"""Privacy-mask and analysis-zone application.

Zone coordinates are normalized to the original image and versioned in local
configuration. A privacy mask is rasterized before detection and before any
frame is retained as evidence, so masked pixels never reach the detector or a
reviewer. The analysis ROI only filters detections; it is not a claim that the
remaining view is sufficient for a product-level conclusion.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from shop_core.geometry import point_in_polygon


def _pixel_polygon(
    polygon: Sequence[Sequence[float]], width: int, height: int
) -> np.ndarray:
    points = []
    for x, y in polygon:
        px = int(round(float(x) * max(0, width - 1)))
        py = int(round(float(y) * max(0, height - 1)))
        points.append((max(0, min(width - 1, px)), max(0, min(height - 1, py))))
    return np.asarray(points, dtype=np.int32)


def apply_privacy_mask(image: np.ndarray, polygon: Sequence[Sequence[float]]) -> np.ndarray:
    """Fill a normalized privacy polygon with opaque black, in place.

    Empty configuration is a no-op. The polygon is closed by OpenCV. Returns
    the same array so callers can use the result directly.
    """
    if not polygon:
        return image
    if image.ndim < 2 or image.shape[0] < 1 or image.shape[1] < 1:
        return image
    height, width = image.shape[:2]
    points = _pixel_polygon(polygon, width, height)
    if len(points) < 3:
        return image
    cv2.fillPoly(image, [points], (0, 0, 0))
    return image


def mask_contains(image: np.ndarray, polygon: Sequence[Sequence[float]], point: tuple[float, float]) -> bool:
    """Whether a pixel point is inside the normalized privacy polygon."""
    if not polygon or image.ndim < 2:
        return False
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return False
    x, y = point
    return point_in_polygon((x / width, y / height), polygon)


def inside_analysis_roi(
    box: Sequence[float], polygon: Sequence[Sequence[float]], width: int, height: int
) -> bool:
    """Keep a detection when its center lies inside the configured analysis ROI.

    An empty ROI means the full frame is eligible. The center test is a
    deliberate, simple policy: it avoids partial-edge boxes from being treated
    as fully in-zone, and it does not claim that a center-in-zone person is
    necessarily visible enough for product evidence.
    """
    if not polygon:
        return True
    if width <= 0 or height <= 0:
        return False
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height)
    return point_in_polygon(center, polygon)
