"""Reviewed Kalman + IoU + Hungarian person tracker.

This is an independent implementation of the published constant-velocity /
IoU / assignment method described in the project PRD and ML guide. It does not
copy GPL tracker source. Identities are local to one source session and are
never matched across stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from shop_core.contracts import Detection, TrackObservation, TrackStatus
from shop_core.geometry import iou
from shop_core.tracking.kalman import KalmanBoxFilter


@dataclass
class TrackerConfig:
    iou_weight: float = 0.60
    motion_weight: float = 0.40
    gate_chi2: float = 9.4877
    process_noise: float = 12.0
    measurement_noise: float = 18.0
    confirm_hits: int = 3
    max_prediction_ms: int = 1000
    tentative_max_prediction_ms: int = 250
    remove_after_ms: int = 2500

    def __post_init__(self) -> None:
        if abs(self.iou_weight + self.motion_weight - 1.0) > 1e-6:
            raise ValueError("association weights must sum to 1.0")
        if self.gate_chi2 <= 0:
            raise ValueError("gate must be positive")
        if self.process_noise <= 0 or self.measurement_noise <= 0:
            raise ValueError("noise parameters must be positive")


@dataclass
class _Track:
    track_id: str
    filter: KalmanBoxFilter
    first_seen_ms: int
    last_seen_ms: int
    last_detection: Detection
    hits: int = 1
    misses: int = 0
    status: TrackStatus = TrackStatus.TENTATIVE
    score_sum: float = 0.0
    score_count: int = 0
    observed_ms: int = 0
    last_observed_ms: int = 0

    @property
    def mean_score(self) -> float:
        return self.score_sum / self.score_count if self.score_count else self.last_detection.score

    def register_observation(self, detection: Detection, pts_ms: int) -> None:
        if self.score_count:
            self.observed_ms += max(0, pts_ms - self.last_observed_ms)
        self.last_observed_ms = pts_ms
        self.filter.update(_measurement(detection.box))
        self.last_detection = detection
        self.last_seen_ms = pts_ms
        self.hits += 1
        self.misses = 0
        self.score_sum += detection.score
        self.score_count += 1


@dataclass
class TrackerUpdate:
    observations: list[TrackObservation] = field(default_factory=list)
    matched_tracks: int = 0
    new_tracks: int = 0
    removed_tracks: int = 0
    gated_out: int = 0


def _measurement(box: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dtype=float)


class PersonTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1
        self._last_pts_ms: int | None = None

    @property
    def active_track_ids(self) -> list[str]:
        return sorted(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._last_pts_ms = None

    def update(
        self,
        detections: list[Detection],
        pts_ms: int,
        *,
        discontinuity: bool = False,
    ) -> TrackerUpdate:
        if pts_ms < 0:
            raise ValueError("media timestamp cannot be negative")
        result = TrackerUpdate()
        if discontinuity:
            self.reset()

        dt_seconds = 0.0
        if self._last_pts_ms is not None and pts_ms > self._last_pts_ms:
            dt_seconds = (pts_ms - self._last_pts_ms) / 1000.0
        self._last_pts_ms = pts_ms

        tracks = list(self._tracks.values())
        for track in tracks:
            if dt_seconds > 0:
                track.filter.predict(dt_seconds, self.config.process_noise)

        matches: dict[int, int] = {}
        gated_matrix: np.ndarray | None = None
        if tracks and detections:
            gated_matrix = np.full((len(tracks), len(detections)), np.inf)
            for ti, track in enumerate(tracks):
                for di, detection in enumerate(detections):
                    measurement = _measurement(detection.box)
                    try:
                        mahalanobis = track.filter.mahalanobis_squared(measurement)
                    except np.linalg.LinAlgError:
                        result.gated_out += 1
                        continue
                    if not np.isfinite(mahalanobis) or mahalanobis > self.config.gate_chi2:
                        result.gated_out += 1
                        continue
                    overlap = iou(track.filter.box, detection.box)
                    motion_cost = min(mahalanobis / self.config.gate_chi2, 1.0)
                    gated_matrix[ti, di] = (
                        self.config.iou_weight * (1.0 - overlap)
                        + self.config.motion_weight * motion_cost
                    )
            valid = np.isfinite(gated_matrix)
            if valid.any():
                assignment_cost = np.where(valid, gated_matrix, 1e6)
                rows, cols = linear_sum_assignment(assignment_cost)
                for row, col in zip(rows, cols):
                    if np.isfinite(gated_matrix[row, col]):
                        matches[int(row)] = int(col)

        used_detections: set[int] = set()
        for track_index, detection_index in matches.items():
            track = tracks[track_index]
            detection = detections[detection_index]
            track.register_observation(detection, pts_ms)
            used_detections.add(detection_index)
            result.matched_tracks += 1
            if track.status == TrackStatus.TENTATIVE and track.hits >= self.config.confirm_hits:
                track.status = TrackStatus.CONFIRMED

        for track_index, track in enumerate(tracks):
            if track_index in matches:
                continue
            track.misses += 1
            elapsed = pts_ms - track.last_seen_ms
            if track.status == TrackStatus.TENTATIVE:
                if elapsed > self.config.tentative_max_prediction_ms:
                    self._remove(track)
                    result.removed_tracks += 1
            elif track.status in (TrackStatus.CONFIRMED, TrackStatus.LOST):
                if elapsed > self.config.max_prediction_ms:
                    track.status = TrackStatus.LOST
                if elapsed > self.config.remove_after_ms:
                    self._remove(track)
                    result.removed_tracks += 1

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            self._create_track(detection, pts_ms)
            result.new_tracks += 1

        for track in self._tracks.values():
            predicted = track.misses > 0
            result.observations.append(
                TrackObservation(
                    track_id=track.track_id,
                    box=track.filter.box,
                    status=track.status,
                    predicted=predicted,
                    score=track.mean_score,
                    age_ms=max(0, pts_ms - track.first_seen_ms),
                    hits=track.hits,
                    missed=track.misses,
                    mean_observed_score=track.mean_score,
                    observed=not predicted,
                )
            )
        result.observations.sort(key=lambda obs: obs.track_id)
        return result

    def _create_track(self, detection: Detection, pts_ms: int) -> None:
        track_id = f"t{self._next_id:04d}"
        self._next_id += 1
        filter_ = KalmanBoxFilter(
            detection.box,
            measurement_variance=self.config.measurement_noise,
        )
        track = _Track(
            track_id=track_id,
            filter=filter_,
            first_seen_ms=pts_ms,
            last_seen_ms=pts_ms,
            last_detection=detection,
            score_sum=detection.score,
            score_count=1,
            last_observed_ms=pts_ms,
        )
        self._tracks[track_id] = track

    def _remove(self, track: _Track) -> None:
        track.status = TrackStatus.REMOVED
        self._tracks.pop(track.track_id, None)
