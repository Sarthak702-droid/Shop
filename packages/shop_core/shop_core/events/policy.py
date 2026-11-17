"""Versioned, descriptive baseline event policy.

The baseline emits observable person-activity segments. It does not use pose,
product visibility, bag proximity or any shoplifting score. A segment is a
request for human context review, not a finding. ``review_score`` stays null
because no validated score exists for this policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shop_core.contracts import TrackObservation

EVENT_TYPE_PERSON_ACTIVITY = "person_activity_observed"
REASON_PERSON_TRACK = "person_track_observed"


@dataclass
class EventCandidate:
    event_type: str
    reasons: list[str]
    start_pts_ms: int
    end_pts_ms: int
    track_ids: list[str]
    review_score: float | None
    score_semantics: str | None
    quality: dict
    required_evidence: list[str]
    available_evidence: list[str]
    model_bundle: str
    policy_version: str
    observed_ms: int


@dataclass
class _Segment:
    track_id: str
    start_pts_ms: int
    last_observed_pts_ms: int
    observed_samples: int
    score_sum: float
    max_score: float
    predicted_gaps: int
    crowded_max: int
    discontinuity_seen: bool = False
    frame_refs: list[object] = field(default_factory=list)


class BaselineActivityPolicy:
    """Deterministic segmentation with hysteresis, cooldown and explicit quality."""

    def __init__(
        self,
        *,
        version: str,
        sampling_fps: float,
        min_dwell_ms: int,
        gap_split_ms: int,
        cooldown_ms: int,
        max_event_duration_ms: int = 60000,
        max_events: int = 200,
    ) -> None:
        self.version = version
        self.sampling_fps = sampling_fps
        self.interval_ms = int(round(1000.0 / sampling_fps))
        self.min_dwell_ms = min_dwell_ms
        self.gap_split_ms = max(gap_split_ms, self.interval_ms)
        self.cooldown_ms = cooldown_ms
        self.max_event_duration_ms = max(max_event_duration_ms, self.min_dwell_ms + self.interval_ms)
        self.max_events = max_events
        self._segments: dict[str, _Segment] = {}
        self._last_event_end: dict[str, int] = {}
        self.emitted = 0

    def ingest(
        self,
        *,
        pts_ms: int,
        tracks: list[TrackObservation],
        person_count: int,
        model_bundle: str,
        source_discontinuity: bool = False,
    ) -> list[EventCandidate]:
        completed: list[EventCandidate] = []
        visible: set[str] = set()
        for track in tracks:
            if not track.observed or track.status.value not in {"confirmed", "tentative"}:
                continue
            if track.status.value != "confirmed":
                continue
            visible.add(track.track_id)
            segment = self._segments.get(track.track_id)
            if segment is None:
                last_end = self._last_event_end.get(track.track_id)
                if last_end is not None and pts_ms - last_end < self.cooldown_ms:
                    continue
                segment = _Segment(
                    track_id=track.track_id,
                    start_pts_ms=pts_ms,
                    last_observed_pts_ms=pts_ms,
                    observed_samples=1,
                    score_sum=track.mean_observed_score,
                    max_score=track.score,
                    predicted_gaps=0,
                    crowded_max=person_count,
                    discontinuity_seen=source_discontinuity,
                )
                self._segments[track.track_id] = segment
            else:
                gap = pts_ms - segment.last_observed_pts_ms
                if gap > self.gap_split_ms or source_discontinuity:
                    segment.discontinuity_seen = (
                        segment.discontinuity_seen or source_discontinuity
                    )
                    completed.append(self._close(segment, end_pts_ms=segment.last_observed_pts_ms, model_bundle=model_bundle))
                    del self._segments[track.track_id]
                    continue
                segment.last_observed_pts_ms = pts_ms
                segment.observed_samples += 1
                segment.score_sum += track.mean_observed_score
                segment.max_score = max(segment.max_score, track.score)
                segment.crowded_max = max(segment.crowded_max, person_count)
                segment.discontinuity_seen = segment.discontinuity_seen or source_discontinuity
            if pts_ms - segment.start_pts_ms >= self.max_event_duration_ms:
                completed.append(self._close(segment, end_pts_ms=pts_ms, model_bundle=model_bundle))
                del self._segments[track.track_id]
        for track in tracks:
            if track.observed or track.status.value != "confirmed":
                continue
            segment = self._segments.get(track.track_id)
            if segment is not None:
                segment.predicted_gaps += 1
        for track_id in list(self._segments):
            segment = self._segments[track_id]
            if track_id in visible:
                continue
            gap = pts_ms - segment.last_observed_pts_ms
            if gap > self.gap_split_ms:
                completed.append(
                    self._close(segment, end_pts_ms=segment.last_observed_pts_ms, model_bundle=model_bundle)
                )
                del self._segments[track_id]
        return [item for item in completed if item is not None]

    def flush(self, *, model_bundle: str, at_pts_ms: int | None = None) -> list[EventCandidate]:
        ended: list[EventCandidate] = []
        for segment in list(self._segments.values()):
            ended.append(
                self._close(
                    segment,
                    end_pts_ms=segment.last_observed_pts_ms,
                    model_bundle=model_bundle,
                )
            )
        self._segments.clear()
        return [item for item in ended if item is not None]

    def active_segments(self) -> dict[str, int]:
        """Track ID to start timestamp for segments that may still open an event."""
        return {track_id: segment.start_pts_ms for track_id, segment in self._segments.items()}

    def _close(self, segment: _Segment, *, end_pts_ms: int, model_bundle: str) -> EventCandidate | None:
        span_ms = max(self.interval_ms, segment.last_observed_pts_ms - segment.start_pts_ms + self.interval_ms)
        observed_span = max(self.interval_ms, segment.observed_samples * self.interval_ms)
        event_end_ms = max(segment.start_pts_ms + self.interval_ms, end_pts_ms + self.interval_ms)
        if span_ms < self.min_dwell_ms:
            self._last_event_end[segment.track_id] = event_end_ms
            return None
        if self.emitted >= self.max_events:
            self._last_event_end[segment.track_id] = event_end_ms
            return None
        self.emitted += 1
        self._last_event_end[segment.track_id] = event_end_ms
        mean_score = segment.score_sum / max(1, segment.observed_samples)
        observed_fraction = min(1.0, observed_span / span_ms)
        quality = {
            "policy": "baseline_person_activity",
            "mean_person_score": round(mean_score, 4),
            "max_person_score": round(segment.max_score, 4),
            "observed_samples": segment.observed_samples,
            "observed_duration_ms": observed_span,
            "observed_fraction": round(observed_fraction, 4),
            "predicted_gaps": segment.predicted_gaps,
            "crowded_max_persons": segment.crowded_max,
            "source_discontinuity": segment.discontinuity_seen,
            "no_activity_score_available": True,
            "warnings": [],
        }
        if observed_fraction < 0.7:
            quality["warnings"].append("intermittent_track_observations")
        if segment.crowded_max > 4:
            quality["warnings"].append("crowded_scene_identity_ambiguity")
        if segment.discontinuity_seen:
            quality["warnings"].append("source_discontinuity")
        return EventCandidate(
            event_type=EVENT_TYPE_PERSON_ACTIVITY,
            reasons=[REASON_PERSON_TRACK],
            start_pts_ms=segment.start_pts_ms,
            end_pts_ms=event_end_ms,
            track_ids=[segment.track_id],
            review_score=None,
            score_semantics=None,
            quality=quality,
            required_evidence=["person_detections", "source_coverage"],
            available_evidence=["person_detections", "track_observations"],
            model_bundle=model_bundle,
            policy_version=self.version,
            observed_ms=observed_span,
        )
