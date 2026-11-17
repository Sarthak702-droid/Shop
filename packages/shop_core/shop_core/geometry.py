"""Deterministic geometry used by detection, tracking and association."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

Box = tuple[float, float, float, float]


def validate_box(box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    values = tuple(float(v) for v in box)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("box coordinates must be finite")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive area")
    return values  # type: ignore[return-value]


def clip_box(box: Sequence[float], width: int, height: int) -> Box:
    x1, y1, x2, y2 = validate_box(box)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("clipped box is degenerate")
    return (x1, y1, x2, y2)


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Continuous-coordinate IoU with no extra +1 pixel convention."""
    validate_box(a)
    validate_box(b)
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_center(box: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_from_center(cx: float, cy: float, width: float, height: float) -> Box:
    half_w = max(width, 1e-6) / 2.0
    half_h = max(height, 1e-6) / 2.0
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def normalized_point_to_box_distance(
    point: tuple[float, float], box: Sequence[float], scale: float
) -> float:
    """Distance from a point to a box's nearest edge, normalized by body scale."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("body scale must be positive and finite")
    px, py = point
    x1, y1, x2, y2 = validate_box(box)
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy) / scale


def point_in_polygon(point: tuple[float, float], polygon: Iterable[Sequence[float]]) -> bool:
    """Even-odd ray casting for normalized or pixel polygons."""
    vertices = [(float(x), float(y)) for x, y in polygon]
    if len(vertices) < 3:
        return False
    x, y = point
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > y) != (yj > y):
            denom = yj - yi
            if denom == 0:
                j = i
                continue
            intersect_x = (xj - xi) * (y - yi) / denom + xi
            if x < intersect_x:
                inside = not inside
        j = i
    return inside


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    vertices = list(polygon)
    if len(vertices) < 3:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(vertices, vertices[1:] + vertices[:1]):
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def normalized_box(box: Sequence[float], width: int, height: int) -> Box:
    x1, y1, x2, y2 = validate_box(box)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    """Return None for a zero-norm embedding: missing evidence, not distance zero."""
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("embeddings must be one-dimensional with equal shape")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    similarity = cosine_similarity(a, b)
    return None if similarity is None else 1.0 - similarity


def temporal_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    if end_a <= start_a or end_b <= start_b:
        raise ValueError("temporal intervals must have positive duration")
    overlap = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - overlap
    return overlap / union if union > 0 else 0.0
