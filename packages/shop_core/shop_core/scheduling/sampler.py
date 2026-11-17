"""Timestamp-aware sampling and coverage accounting."""

from __future__ import annotations

from dataclasses import dataclass

from shop_core.contracts import FrameEnvelope


@dataclass
class SamplerStats:
    accepted: int = 0
    skipped_schedule: int = 0
    missing_samples: int = 0
    discontinuity_events: int = 0
    first_pts_ms: int | None = None
    last_pts_ms: int | None = None
    expected_samples: int | None = None


class SamplingScheduler:
    """Select at most one analysis tick per target interval using media PTS.

    The scheduler never substitutes processing wall time for media time. Gaps
    are counted; a long gap is flagged so the tracker can reset its identity
    namespace rather than bridging unrelated observations.
    """

    def __init__(self, sampling_fps: float, *, gap_reset_ms: int = 2500) -> None:
        if sampling_fps <= 0:
            raise ValueError("sampling FPS must be positive")
        self.interval_ms = 1000.0 / sampling_fps
        self.gap_reset_ms = gap_reset_ms
        self.stats = SamplerStats()
        self._next_due_ms: float | None = None
        self._last_accepted_pts: int | None = None

    def accept(self, frame: FrameEnvelope) -> bool:
        if not frame.valid:
            return False
        if self._next_due_ms is None:
            self._next_due_ms = float(frame.pts_ms)
            self.stats.first_pts_ms = frame.pts_ms
        # Only a frame at or after its due media timestamp may be analyzed. A
        # frame arriving early is skipped and counted; accepting it would make
        # the effective cadence exceed the configured sampling rate.
        if frame.pts_ms < self._next_due_ms:
            self.stats.skipped_schedule += 1
            return False
        gap = 0
        if self._last_accepted_pts is not None:
            elapsed = frame.pts_ms - self._last_accepted_pts
            gap = max(0, int(round(elapsed / self.interval_ms)) - 1)
            self.stats.missing_samples += gap
            if elapsed > self.gap_reset_ms:
                self.stats.discontinuity_events += 1
                frame.discontinuity = True
        self._last_accepted_pts = frame.pts_ms
        self.stats.last_pts_ms = frame.pts_ms
        self.stats.accepted += 1
        self.stats.expected_samples = (self.stats.missing_samples + self.stats.accepted) or None
        self._next_due_ms = max(self._next_due_ms + self.interval_ms, float(frame.pts_ms))
        while self._next_due_ms <= frame.pts_ms:
            self._next_due_ms += self.interval_ms
        return True

    @property
    def coverage(self) -> float | None:
        expected = self.stats.expected_samples
        if not expected or expected <= 0:
            return None
        return min(1.0, self.stats.accepted / expected)
