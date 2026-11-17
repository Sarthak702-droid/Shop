"""Frame- and segment-level metrics for per-step action scores.

* Frame AP: average precision of per-step scores against per-step labels (all-points, sklearn style).
* Segment detection: thresholded runs become predicted segments, scored by their mean probability,
  then greedily matched to ground-truth runs at temporal IoU >= ``iou`` (VOC-style AP plus P/R/F1).

Everything is pure NumPy so results are reproducible without extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = labels[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / positives)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 (used on validation only). Returns (threshold, f1)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    if labels.sum() == 0:
        return 0.5, float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, hits = scores[order], labels[order]
    tp = np.cumsum(hits)
    fp = np.cumsum(~hits)
    f1 = 2 * tp / (tp + fp + labels.sum())
    # only cut between distinct scores
    valid = np.r_[s[1:] != s[:-1], True]
    i = int(np.argmax(np.where(valid, f1, -1)))
    return float(s[i]), float(f1[i])


def f1_at(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    pred = np.asarray(scores).ravel() >= threshold
    labels = np.asarray(labels).ravel().astype(bool)
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def runs(binary: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    b = np.asarray(binary).astype(np.int8).ravel()
    if b.size == 0:
        return []
    edges = np.diff(np.r_[0, b, 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def tiou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    return inter / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter)


@dataclass
class SegmentResult:
    ap: float
    precision: float
    recall: float
    f1: float
    predicted: int
    ground_truth: int


def segment_detection(
    videos: list[tuple[np.ndarray, np.ndarray]], threshold: float, iou: float = 0.5, min_len: int = 1
) -> SegmentResult:
    """``videos``: list of (scores[T], labels[T]) for one class."""
    preds: list[tuple[float, int, tuple[int, int]]] = []
    gts: dict[int, list[tuple[int, int]]] = {}
    for vi, (scores, labels) in enumerate(videos):
        gts[vi] = runs(labels)
        for seg in runs(np.asarray(scores) >= threshold):
            if seg[1] - seg[0] + 1 >= min_len:
                preds.append((float(np.mean(scores[seg[0] : seg[1] + 1])), vi, seg))
    total_gt = sum(len(v) for v in gts.values())
    preds.sort(key=lambda p: -p[0])
    used = {vi: [False] * len(g) for vi, g in gts.items()}
    hits = []
    for _score, vi, seg in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gts[vi]):
            if not used[vi][j]:
                o = tiou(seg, g)
                if o > best:
                    best, best_j = o, j
        if best >= iou:
            used[vi][best_j] = True
            hits.append(True)
        else:
            hits.append(False)
    tp = int(sum(hits))
    p = tp / len(preds) if preds else 0.0
    r = tp / total_gt if total_gt else 0.0
    if total_gt and hits:
        h = np.asarray(hits)
        prec = np.cumsum(h) / np.arange(1, len(h) + 1)
        ap = float((prec * h).sum() / total_gt)
    else:
        ap = 0.0 if total_gt else float("nan")
    return SegmentResult(ap, p, r, 2 * p * r / (p + r) if p + r else 0.0, len(preds), total_gt)
