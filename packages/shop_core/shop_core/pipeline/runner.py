"""CPU analysis pipeline for prerecorded files and live webcam sessions.

The pipeline owns source timing, sampling, detection, tracking, descriptive
policy, progress and evidence preparation. Database writes are delegated to a
sink so expensive work never runs inside an HTTP handler.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from shop_core.config import Settings
from shop_core.contracts import (
    AnalysisMetrics,
    Detection,
    EventRecord,
    EvidenceStatus,
    TrackObservation,
)
from shop_core.detection.person import PersonDetector
from shop_core.events.policy import BaselineActivityPolicy, EventCandidate
from shop_core.evidence.writer import EvidenceClip, EvidenceError, EvidenceFrame, EvidenceWriter
from shop_core.scheduling.sampler import SamplingScheduler
from shop_core.sources.media import FileVideoSource, SourceFault, WebcamSource
from shop_core.tracking.tracker import PersonTracker, TrackerConfig
from shop_core.zones import apply_privacy_mask, inside_analysis_roi


@dataclass
class SampleRecord:
    pts_ms: int
    detections: list[Detection]
    tracks: list[TrackObservation]
    discontinuity: bool = False


class PipelineSink(Protocol):
    def should_cancel(self) -> bool: ...

    def report_progress(self, progress: dict) -> None: ...

    def event_started(self, record: EventRecord) -> None: ...

    def event_evidence_ready(
        self, event_id: str, clip: EvidenceClip | None, reason: str | None
    ) -> None: ...

    def heartbeat(self) -> None: ...


@dataclass
class RunResult:
    status: str
    error_code: str | None
    error_message: str | None
    metrics: AnalysisMetrics
    coverage: dict
    events_created: int
    evidence_failures: int
    notes: list[str] = field(default_factory=list)


class InMemorySink:
    """Sink used by tests and the benchmark harness."""

    def __init__(self) -> None:
        self.progress: dict = {}
        self.events: list[EventRecord] = []
        self.evidence: dict[str, tuple[EvidenceClip | None, str | None]] = {}
        self.cancelled = False

    def should_cancel(self) -> bool:
        return self.cancelled

    def report_progress(self, progress: dict) -> None:
        self.progress = progress

    def event_started(self, record: EventRecord) -> None:
        self.events.append(record)

    def event_evidence_ready(self, event_id: str, clip: EvidenceClip | None, reason: str | None) -> None:
        self.evidence[event_id] = (clip, reason)

    def heartbeat(self) -> None:
        pass


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def peak_rss_kb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def build_tracker(settings: Settings) -> PersonTracker:
    cfg = settings.analysis
    return PersonTracker(
        TrackerConfig(
            iou_weight=cfg.tracker_iou_weight,
            motion_weight=cfg.tracker_motion_weight,
            gate_chi2=cfg.tracker_gate_chi2,
            process_noise=cfg.tracker_process_noise,
            measurement_noise=cfg.tracker_measurement_noise,
            confirm_hits=cfg.tracker_confirm_hits,
            max_prediction_ms=cfg.tracker_max_prediction_ms,
            tentative_max_prediction_ms=cfg.tracker_tentative_max_prediction_ms,
            remove_after_ms=cfg.tracker_remove_after_ms,
        )
    )


def build_policy(settings: Settings, sampling_fps: float | None = None) -> BaselineActivityPolicy:
    return BaselineActivityPolicy(
        version=settings.policy.version,
        sampling_fps=sampling_fps or settings.analysis.sampling_fps,
        min_dwell_ms=settings.analysis.event_min_dwell_ms,
        gap_split_ms=settings.analysis.event_gap_split_ms,
        cooldown_ms=settings.policy.cooldown_ms,
        max_events=settings.analysis.max_events_per_job,
    )


def model_bundle_label(settings: Settings, detector: PersonDetector) -> str:
    return f"{detector.model_id}:{detector.version}|policy:{settings.policy.version}"


def read_window_frames(
    path: Path,
    *,
    start_ms: int,
    end_ms: int,
    sampling_fps: float,
    observations: list[SampleRecord],
    privacy_polygon: list[list[float]] | None = None,
    max_long_edge: int | None = None,
) -> list[EvidenceFrame]:
    """Re-decode a media window for evidence.

    Seeking is approximate in many backends, so decoding begins a few seconds
    before the requested start and frames are selected by their own timestamps.
    ``max_long_edge`` downscales frames as they are read (boxes are rescaled with
    them) so long windows do not hold full-resolution frames in memory.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise EvidenceError("EVIDENCE_SOURCE_UNAVAILABLE", "Source media could not be reopened for evidence.")
    interval = 1000.0 / sampling_fps
    seek_to = max(0, start_ms - 3000)
    capture.set(cv2.CAP_PROP_POS_MSEC, float(seek_to))
    next_due = float(start_ms)
    frames: list[EvidenceFrame] = []
    last_pts = -1
    observation_index = 0
    ordered = sorted(observations, key=lambda item: item.pts_ms)
    try:
        while True:
            ok, image = capture.read()
            if not ok or image is None:
                break
            pts = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if pts <= 0:
                pts = float(max(last_pts + interval, start_ms))
            pts_ms = int(round(pts))
            if pts_ms < start_ms - interval:
                last_pts = pts_ms
                continue
            if pts_ms > end_ms + interval * 1.5:
                break
            if pts_ms + interval * 0.5 < next_due:
                last_pts = pts_ms
                continue
            while observation_index + 1 < len(ordered) and ordered[observation_index + 1].pts_ms <= pts_ms:
                observation_index += 1
            observation = ordered[observation_index] if ordered else None
            if observation is None or abs(observation.pts_ms - pts_ms) > max(250.0, interval * 2):
                observation = None
            if privacy_polygon:
                image = apply_privacy_mask(image, privacy_polygon)
            detections = observation.detections if observation else []
            tracks = observation.tracks if observation else []
            if max_long_edge and max(image.shape[:2]) > max_long_edge:
                scale = max_long_edge / max(image.shape[:2])
                image = cv2.resize(image, (max(2, int(round(image.shape[1] * scale))), max(2, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
                detections = [replace(d, box=tuple(v * scale for v in d.box)) for d in detections]
                tracks = [replace(t, box=tuple(v * scale for v in t.box)) for t in tracks]
            frames.append(
                EvidenceFrame(
                    pts_ms=pts_ms,
                    image_bgr=image,
                    detections=detections,
                    tracks=tracks,
                )
            )
            next_due = max(next_due + interval, float(pts_ms))
            while next_due <= pts_ms:
                next_due += interval
            last_pts = pts_ms
    finally:
        capture.release()
    if not frames:
        raise EvidenceError(
            "EVIDENCE_WINDOW_EMPTY",
            "No decodable frames were found in the requested evidence window.",
        )
    return frames


def _progress_payload(
    *,
    metrics: AnalysisMetrics,
    coverage: float | None,
    events_created: int,
    stage: str,
    last_pts_ms: int | None,
    duration_ms: int | None,
    source_stats: dict,
    queue: dict | None = None,
) -> dict:
    payload = {
        "stage": stage,
        "media_pts_ms": last_pts_ms,
        "duration_ms": duration_ms,
        "analysed_frames": metrics.analyzed_frames,
        "dropped_frames": metrics.dropped_frames,
        "expected_samples": metrics.expected_samples,
        "analysis_coverage": None if coverage is None else round(coverage, 4),
        "persons_detected": metrics.person_detections,
        "events_created": events_created,
        "decode_gaps": metrics.decode_gaps,
        "source": source_stats,
    }
    if queue is not None:
        payload["queue"] = queue
    return payload


def _event_identity(
    settings: Settings,
    source_id: str,
    session_id: str,
    job_id: str,
    candidate: EventCandidate,
) -> EventRecord:
    return EventRecord(
        event_id=str(uuid.uuid4()),
        source_id=source_id,
        session_id=session_id,
        job_id=job_id,
        event_type=candidate.event_type,
        reasons=candidate.reasons,
        start_pts_ms=candidate.start_pts_ms,
        end_pts_ms=candidate.end_pts_ms,
        track_ids=candidate.track_ids,
        review_score=candidate.review_score,
        score_semantics=candidate.score_semantics,
        calibration_version=None,
        quality=candidate.quality,
        required_evidence=candidate.required_evidence,
        available_evidence=candidate.available_evidence,
        model_bundle=candidate.model_bundle,
        policy_version=candidate.policy_version,
        evidence=EvidenceStatus(status="pending"),
        created_at_utc=now_iso(),
    )


def _finalize_metrics(
    metrics: AnalysisMetrics,
    *,
    started_wall: float,
    media_span_ms: int | None,
) -> None:
    metrics.finished_at_utc = now_iso()
    metrics.wall_time_ms = int((time.perf_counter() - started_wall) * 1000.0)
    if metrics.wall_time_ms > 0:
        metrics.valid_analysis_fps = round(metrics.analyzed_frames / (metrics.wall_time_ms / 1000.0), 3)
    if media_span_ms and media_span_ms > 0:
        metrics.real_time_factor = round(metrics.wall_time_ms / media_span_ms, 3)
    metrics.peak_rss_kb = peak_rss_kb()


def run_file_analysis(
    *,
    settings: Settings,
    detector: PersonDetector,
    path: Path,
    duration_ms: int | None,
    job_id: str,
    source_id: str,
    session_id: str,
    sink: PipelineSink,
    evidence_writer: EvidenceWriter | None = None,
    generate_evidence: bool = True,
    sampling_fps: float | None = None,
) -> RunResult:
    sampling_fps = sampling_fps or settings.analysis.sampling_fps
    if sampling_fps <= 0 or sampling_fps > settings.analysis.sampling_fps:
        raise ValueError("requested sampling cadence exceeds the configured analysis profile")
    started_wall = time.perf_counter()
    metrics = AnalysisMetrics(
        started_at_utc=now_iso(),
        media_duration_ms=duration_ms,
        inference_threads=settings.runtime.inference_threads,
    )
    source = FileVideoSource(path, source_id=source_id, session_id=session_id)
    tracker = build_tracker(settings)
    policy = build_policy(settings, sampling_fps)
    scheduler = SamplingScheduler(
        sampling_fps,
        gap_reset_ms=settings.analysis.tracker_remove_after_ms,
    )
    samples: list[SampleRecord] = []
    records: list[EventRecord] = []
    records_by_id: dict[str, EventRecord] = {}
    candidates_by_id: dict[str, EventCandidate] = {}
    seen_confirmed: set[str] = set()
    evidence_failures: dict[str, str] = {}
    status = "completed"
    error_code: str | None = None
    error_message: str | None = None
    notes: list[str] = []
    last_progress = 0.0
    last_heartbeat = 0.0
    actual_last_pts: int | None = None
    model_bundle = model_bundle_label(settings, detector)
    writer = evidence_writer or EvidenceWriter(settings.evidence_dir)

    try:
        source.open()
        for frame in source.frames(should_stop=sink.should_cancel):
            if frame.discontinuity:
                metrics.decode_gaps += 1
            if not scheduler.accept(frame):
                continue
            if frame.decode_ms is not None:
                metrics.decode_ms.append(frame.decode_ms)
            apply_privacy_mask(frame.image_bgr, settings.zones.privacy_mask)
            try:
                detections = detector.detect(frame.image_bgr)
            except RuntimeError as error:
                metrics.detection_failures += 1
                raise SourceFault("DETECTOR_FAILED", "Person detector inference failed.") from error
            if settings.zones.analysis_roi:
                detections = [
                    detection
                    for detection in detections
                    if inside_analysis_roi(
                        detection.box,
                        settings.zones.analysis_roi,
                        frame.width,
                        frame.height,
                    )
                ]
            metrics.detect_ms.append(detector.last_timing.detect_ms)
            metrics.preprocess_ms.append(detector.last_timing.preprocess_ms)
            track_started = time.perf_counter()
            update = tracker.update(detections, frame.pts_ms, discontinuity=frame.discontinuity)
            metrics.track_ms.append((time.perf_counter() - track_started) * 1000.0)
            metrics.analyzed_frames += 1
            metrics.person_detections += len(detections)
            source.stats.frames_sampled += 1
            actual_last_pts = frame.pts_ms
            for observation in update.observations:
                if observation.status.value == "confirmed" and observation.track_id not in seen_confirmed:
                    seen_confirmed.add(observation.track_id)
            samples.append(
                SampleRecord(
                    pts_ms=frame.pts_ms,
                    detections=detections,
                    tracks=update.observations,
                    discontinuity=frame.discontinuity,
                )
            )
            completed = policy.ingest(
                pts_ms=frame.pts_ms,
                tracks=update.observations,
                person_count=len(detections),
                model_bundle=model_bundle,
                source_discontinuity=frame.discontinuity,
            )
            for candidate in completed:
                record = _event_identity(settings, source_id, session_id, job_id, candidate)
                records.append(record)
                records_by_id[record.event_id] = record
                candidates_by_id[record.event_id] = candidate
                sink.event_started(record)
            elapsed = time.perf_counter()
            if elapsed - last_progress >= 1.0:
                sink.report_progress(
                    _progress_payload(
                        metrics=metrics,
                        coverage=scheduler.coverage,
                        events_created=len(records),
                        stage="analysing",
                        last_pts_ms=frame.pts_ms,
                        duration_ms=duration_ms,
                        source_stats=source.stats.to_public(),
                    )
                )
                last_progress = elapsed
            if elapsed - last_heartbeat >= 5.0:
                sink.heartbeat()
                last_heartbeat = elapsed
        for candidate in policy.flush(model_bundle=model_bundle):
            record = _event_identity(settings, source_id, session_id, job_id, candidate)
            records.append(record)
            records_by_id[record.event_id] = record
            candidates_by_id[record.event_id] = candidate
            sink.event_started(record)
        if source.stats.decode_incomplete:
            notes.append("Decoder reached EOF before the declared frame count; analysis coverage is partial.")
        if sink.should_cancel():
            status = "cancelled"
    except SourceFault as error:
        status = "failed"
        error_code = error.code
        error_message = error.message
        for candidate in policy.flush(model_bundle=model_bundle):
            record = _event_identity(settings, source_id, session_id, job_id, candidate)
            records.append(record)
            records_by_id[record.event_id] = record
            candidates_by_id[record.event_id] = candidate
            sink.event_started(record)
    except Exception as error:  # noqa: BLE001 - failures are explicit and recorded
        status = "failed"
        error_code = getattr(error, "code", "PIPELINE_FAILED")
        error_message = getattr(error, "message", "Analysis pipeline failed unexpectedly.")
        notes.append(f"failure_type={type(error).__name__}")
    finally:
        source.close()

    if status == "failed":
        for record in records:
            evidence_failures[record.event_id] = "ANALYSIS_FAILED: evidence window was not generated."
            sink.event_evidence_ready(record.event_id, None, evidence_failures[record.event_id])
    elif not generate_evidence:
        for record in records:
            reason = "EVIDENCE_RETENTION_DISABLED: this job requested no retained evidence media."
            evidence_failures[record.event_id] = reason
            sink.event_evidence_ready(record.event_id, None, reason)
    else:
        for record in records:
            candidate = candidates_by_id[record.event_id]
            if duration_ms and duration_ms > 0:
                media_end = max(duration_ms, actual_last_pts or 0)
            else:
                media_end = actual_last_pts or record.end_pts_ms
            if status == "cancelled" and actual_last_pts is not None:
                media_end = min(media_end, actual_last_pts)
            window_start = max(0, min(record.start_pts_ms - settings.analysis.evidence_context_before_ms, media_end - 1))
            window_end = max(window_start + 1, min(record.end_pts_ms + settings.analysis.evidence_context_after_ms, media_end))
            if window_end <= window_start:
                evidence_failures[record.event_id] = "EVIDENCE_WINDOW_EMPTY: no media range was available."
                sink.event_evidence_ready(record.event_id, None, evidence_failures[record.event_id])
                continue
            evidence_started = time.perf_counter()
            try:
                frames = read_window_frames(
                    path,
                    start_ms=window_start,
                    end_ms=window_end,
                    sampling_fps=sampling_fps,
                    observations=samples,
                    privacy_polygon=settings.zones.privacy_mask,
                )
                clip = writer.write_clip(
                    event_id=record.event_id,
                    frames=frames,
                    start_pts_ms=frames[0].pts_ms,
                    end_pts_ms=max(frames[0].pts_ms + 1, frames[-1].pts_ms + int(1000 / sampling_fps)),
                )
                metrics.evidence_ms.append(clip.generate_ms)
                sink.event_evidence_ready(record.event_id, clip, None)
            except EvidenceError as error:
                metrics.evidence_ms.append((time.perf_counter() - evidence_started) * 1000.0)
                reason = f"{error.code}: {error.message}"
                evidence_failures[record.event_id] = reason
                sink.event_evidence_ready(record.event_id, None, reason)
            except Exception:  # noqa: BLE001 - evidence failure must be reported, not hidden
                metrics.evidence_ms.append((time.perf_counter() - evidence_started) * 1000.0)
                reason = "EVIDENCE_GENERATION_FAILED: clip could not be generated."
                evidence_failures[record.event_id] = reason
                sink.event_evidence_ready(record.event_id, None, reason)

    metrics.confirmed_tracks = len(seen_confirmed)
    media_span_ms = None
    if actual_last_pts is not None and scheduler.stats.first_pts_ms is not None:
        media_span_ms = max(1, actual_last_pts - scheduler.stats.first_pts_ms)
    _finalize_metrics(metrics, started_wall=started_wall, media_span_ms=media_span_ms)
    metrics.expected_samples = scheduler.stats.expected_samples
    metrics.analysis_coverage = scheduler.coverage
    if status == "cancelled":
        notes.append("Cancellation settled after the last analysed media timestamp; later media was not analysed.")
    coverage = {
        "analysis_coverage": metrics.analysis_coverage,
        "expected_samples": metrics.expected_samples,
        "analyzed_samples": metrics.analyzed_frames,
        "missing_samples": scheduler.stats.missing_samples,
        "decode_gaps": metrics.decode_gaps,
        "decode_incomplete": source.stats.decode_incomplete,
        "measured_media_span_ms": media_span_ms,
        "declared_duration_ms": duration_ms,
        "source_stats": source.stats.to_public(),
        "evidence_generation": {
            "requested": len(records),
            "failures": len(evidence_failures),
            "failure_reasons": evidence_failures,
        },
        "label": "actual_measured_result",
    }
    sink.report_progress(
        _progress_payload(
            metrics=metrics,
            coverage=metrics.analysis_coverage,
            events_created=len(records),
            stage="finished",
            last_pts_ms=actual_last_pts,
            duration_ms=duration_ms,
            source_stats=source.stats.to_public(),
        )
    )
    return RunResult(
        status=status,
        error_code=error_code,
        error_message=error_message,
        metrics=metrics,
        coverage=coverage,
        events_created=len(records),
        evidence_failures=len(evidence_failures),
        notes=notes,
    )


@dataclass
class LiveBuffer:
    track_id: str
    start_pts_ms: int
    frames: list[tuple[int, bytes, int]]
    record: EventRecord | None = None
    finalize_at_ms: int | None = None
    truncated: bool = False

    @property
    def key(self) -> str:
        return f"{self.track_id}:{self.start_pts_ms}"


class LiveEvidenceStore:
    """Bounded in-memory evidence cache for live sessions.

    JPEG bytes are encoded once per analysed sample and shared by reference
    between the context ring and open segment buffers. A live source has no
    recording to re-decode later, so this cache is the only evidence source.
    """

    def __init__(
        self,
        *,
        before_ms: int,
        after_ms: int,
        interval_ms: int,
        max_buffers: int = 8,
        long_edge: int = 960,
    ) -> None:
        self.before_ms = before_ms
        self.after_ms = after_ms
        self.interval_ms = interval_ms
        self.max_buffers = max_buffers
        self.long_edge = long_edge
        ring_seconds = (before_ms + after_ms) / 1000.0
        self.ring: deque[tuple[int, bytes, int]] = deque(
            maxlen=max(10, int(ring_seconds / (interval_ms / 1000.0)) + 10)
        )
        self.buffers: dict[str, LiveBuffer] = {}
        self.skipped: dict[str, str] = {}

    def add_frame(self, pts_ms: int, image: np.ndarray, sample_index: int) -> None:
        height, width = image.shape[:2]
        scale = self.long_edge / max(height, width)
        if scale < 1.0:
            image = cv2.resize(image, (max(2, int(width * scale)), max(2, int(height * scale))))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return
        item = (pts_ms, encoded.tobytes(), sample_index)
        self.ring.append(item)
        for buffer in self.buffers.values():
            buffer.frames.append(item)

    def sync_segments(self, active: dict[str, int]) -> None:
        for track_id, start_pts_ms in active.items():
            key = f"{track_id}:{start_pts_ms}"
            if key in self.buffers or key in self.skipped:
                continue
            if len(self.buffers) >= self.max_buffers:
                self.skipped[key] = "LIVE_EVIDENCE_CONCURRENCY_LIMIT: too many simultaneous segments."
                continue
            seed = [item for item in self.ring if item[0] >= start_pts_ms - self.before_ms]
            self.buffers[key] = LiveBuffer(
                track_id=track_id,
                start_pts_ms=start_pts_ms,
                frames=seed,
            )

    def attach(self, record: EventRecord) -> str | None:
        track_id = record.track_ids[0] if record.track_ids else ""
        key = f"{track_id}:{record.start_pts_ms}"
        buffer = self.buffers.get(key)
        if buffer is None:
            return self.skipped.get(
                key, "LIVE_EVIDENCE_UNAVAILABLE: the segment buffer was not retained."
            )
        buffer.record = record
        buffer.finalize_at_ms = record.end_pts_ms + self.after_ms
        return None

    def ready(self, now_pts_ms: int) -> list[LiveBuffer]:
        ready = [
            buffer
            for buffer in self.buffers.values()
            if buffer.record is not None
            and (buffer.truncated or (buffer.finalize_at_ms is not None and now_pts_ms >= buffer.finalize_at_ms))
        ]
        for buffer in ready:
            self.buffers.pop(buffer.key, None)
        return ready

    def finish_all(self) -> list[LiveBuffer]:
        ready = [buffer for buffer in self.buffers.values() if buffer.record is not None]
        for buffer in ready:
            buffer.truncated = True
            self.buffers.pop(buffer.key, None)
        return ready

    def pending_count(self) -> int:
        return sum(1 for buffer in self.buffers.values() if buffer.record is not None)


def _decoded_frames(buffer: LiveBuffer, samples: list[SampleRecord]) -> list[EvidenceFrame]:
    frames: list[EvidenceFrame] = []
    for pts_ms, jpeg, sample_index in buffer.frames:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        sample = samples[sample_index] if 0 <= sample_index < len(samples) else None
        frames.append(
            EvidenceFrame(
                pts_ms=pts_ms,
                image_bgr=image,
                detections=sample.detections if sample else [],
                tracks=sample.tracks if sample else [],
            )
        )
    return frames


def run_webcam_analysis(
    *,
    settings: Settings,
    detector: PersonDetector,
    camera: WebcamSource,
    job_id: str,
    source_id: str,
    session_id: str,
    sink: PipelineSink,
    evidence_writer: EvidenceWriter | None = None,
    max_duration_ms: int = 30 * 60 * 1000,
    generate_evidence: bool = True,
    sampling_fps: float | None = None,
) -> RunResult:
    sampling_fps = sampling_fps or settings.analysis.sampling_fps
    if sampling_fps <= 0 or sampling_fps > settings.analysis.sampling_fps:
        raise ValueError("requested sampling cadence exceeds the configured analysis profile")
    started_wall = time.perf_counter()
    metrics = AnalysisMetrics(
        started_at_utc=now_iso(),
        inference_threads=settings.runtime.inference_threads,
    )
    tracker = build_tracker(settings)
    policy = build_policy(settings, sampling_fps)
    interval_ms = int(round(1000.0 / sampling_fps))
    live_store = LiveEvidenceStore(
        before_ms=settings.analysis.evidence_context_before_ms,
        after_ms=settings.analysis.evidence_context_after_ms,
        interval_ms=interval_ms,
    )
    writer = evidence_writer or EvidenceWriter(settings.evidence_dir)
    samples: list[SampleRecord] = []
    records: list[EventRecord] = []
    seen_confirmed: set[str] = set()
    evidence_failures: dict[str, str] = {}
    status = "completed"
    error_code: str | None = None
    error_message: str | None = None
    notes: list[str] = []
    model_bundle = model_bundle_label(settings, detector)
    last_progress = 0.0
    last_heartbeat = 0.0
    last_sampled_pts: int | None = None
    started_pts: int | None = None

    def finalize(buffer: LiveBuffer) -> None:
        if buffer.record is None:
            return
        if not generate_evidence:
            reason = "EVIDENCE_RETENTION_DISABLED: this job requested no retained evidence media."
            evidence_failures[buffer.record.event_id] = reason
            sink.event_evidence_ready(buffer.record.event_id, None, reason)
            return
        frames = _decoded_frames(buffer, samples)
        if not frames:
            reason = "LIVE_EVIDENCE_EMPTY: no cached live frames were available."
            evidence_failures[buffer.record.event_id] = reason
            sink.event_evidence_ready(buffer.record.event_id, None, reason)
            return
        started = time.perf_counter()
        suffix = " (context truncated at session end)" if buffer.truncated else ""
        try:
            clip = writer.write_clip(
                event_id=buffer.record.event_id,
                frames=frames,
                start_pts_ms=frames[0].pts_ms,
                end_pts_ms=max(frames[0].pts_ms + 1, frames[-1].pts_ms + interval_ms),
            )
            metrics.evidence_ms.append(clip.generate_ms)
            sink.event_evidence_ready(buffer.record.event_id, clip, None)
        except EvidenceError as error:
            metrics.evidence_ms.append((time.perf_counter() - started) * 1000.0)
            reason = f"{error.code}: {error.message}{suffix}"
            evidence_failures[buffer.record.event_id] = reason
            sink.event_evidence_ready(buffer.record.event_id, None, reason)
        except Exception:  # noqa: BLE001
            metrics.evidence_ms.append((time.perf_counter() - started) * 1000.0)
            reason = f"EVIDENCE_GENERATION_FAILED: clip could not be generated.{suffix}"
            evidence_failures[buffer.record.event_id] = reason
            sink.event_evidence_ready(buffer.record.event_id, None, reason)

    def accept_candidates(candidates: list[EventCandidate]) -> None:
        for candidate in candidates:
            record = _event_identity(settings, source_id, session_id, job_id, candidate)
            records.append(record)
            sink.event_started(record)
            reason = live_store.attach(record)
            if reason:
                evidence_failures[record.event_id] = reason
                sink.event_evidence_ready(record.event_id, None, reason)

    try:
        camera.open()
        while not sink.should_cancel():
            frame = camera.next_frame(timeout=0.25)
            elapsed_wall = time.perf_counter()
            for buffer in live_store.ready(last_sampled_pts or 0):
                finalize(buffer)
            if frame is None:
                if camera.disconnected:
                    raise SourceFault("CAMERA_DISCONNECTED", "Camera stopped delivering frames.")
                if elapsed_wall - last_heartbeat >= 5.0:
                    sink.heartbeat()
                    last_heartbeat = elapsed_wall
                continue
            if started_pts is None:
                started_pts = frame.pts_ms
            if last_sampled_pts is not None:
                gap = frame.pts_ms - last_sampled_pts
                if gap < interval_ms * 0.8:
                    camera.stats.dropped_schedule += 1
                    continue
                if gap > settings.analysis.tracker_remove_after_ms:
                    metrics.dropped_frames += int(gap // interval_ms)
                    metrics.decode_gaps += 1
                    frame.discontinuity = True
            if frame.pts_ms - (started_pts or 0) > max_duration_ms:
                notes.append("Live safety duration reached; the session stopped cleanly.")
                break
            last_sampled_pts = frame.pts_ms
            apply_privacy_mask(frame.image_bgr, settings.zones.privacy_mask)
            try:
                detections = detector.detect(frame.image_bgr)
            except RuntimeError as error:
                metrics.detection_failures += 1
                raise SourceFault("DETECTOR_FAILED", "Person detector inference failed.") from error
            if settings.zones.analysis_roi:
                detections = [
                    detection
                    for detection in detections
                    if inside_analysis_roi(
                        detection.box,
                        settings.zones.analysis_roi,
                        frame.width,
                        frame.height,
                    )
                ]
            metrics.detect_ms.append(detector.last_timing.detect_ms)
            metrics.preprocess_ms.append(detector.last_timing.preprocess_ms)
            track_started = time.perf_counter()
            update = tracker.update(detections, frame.pts_ms, discontinuity=frame.discontinuity)
            metrics.track_ms.append((time.perf_counter() - track_started) * 1000.0)
            metrics.analyzed_frames += 1
            metrics.person_detections += len(detections)
            for observation in update.observations:
                if observation.status.value == "confirmed" and observation.track_id not in seen_confirmed:
                    seen_confirmed.add(observation.track_id)
            sample_index = len(samples)
            samples.append(
                SampleRecord(
                    pts_ms=frame.pts_ms,
                    detections=detections,
                    tracks=update.observations,
                    discontinuity=frame.discontinuity,
                )
            )
            live_store.add_frame(frame.pts_ms, frame.image_bgr, sample_index)
            completed = policy.ingest(
                pts_ms=frame.pts_ms,
                tracks=update.observations,
                person_count=len(detections),
                model_bundle=model_bundle,
                source_discontinuity=frame.discontinuity,
            )
            live_store.sync_segments(policy.active_segments())
            accept_candidates(completed)
            for buffer in live_store.ready(frame.pts_ms):
                finalize(buffer)
            if elapsed_wall - last_progress >= 1.0:
                sink.report_progress(
                    _progress_payload(
                        metrics=metrics,
                        coverage=None,
                        events_created=len(records),
                        stage="live",
                        last_pts_ms=frame.pts_ms,
                        duration_ms=None,
                        source_stats=camera.stats.to_public(),
                        queue={
                            "capacity": camera.queue_frames,
                            "dropped_stale": camera.dropped_stale,
                            "connected": not camera.disconnected,
                            "evidence_pending": live_store.pending_count(),
                        },
                    )
                )
                last_progress = elapsed_wall
            if elapsed_wall - last_heartbeat >= 5.0:
                sink.heartbeat()
                last_heartbeat = elapsed_wall
        if sink.should_cancel():
            status = "cancelled"
    except SourceFault as error:
        status = "failed"
        error_code = error.code
        error_message = error.message
    except Exception as error:  # noqa: BLE001
        status = "failed"
        error_code = getattr(error, "code", "LIVE_PIPELINE_FAILED")
        error_message = getattr(error, "message", "Live analysis failed unexpectedly.")
        notes.append(f"failure_type={type(error).__name__}")
    finally:
        try:
            accept_candidates(policy.flush(model_bundle=model_bundle))
        except Exception:  # noqa: BLE001
            notes.append("policy_flush_failed")
        for buffer in live_store.finish_all():
            finalize(buffer)
        camera.close()

    metrics.confirmed_tracks = len(seen_confirmed)
    _finalize_metrics(metrics, started_wall=started_wall, media_span_ms=None)
    coverage = {
        "analysis_coverage": None,
        "analyzed_samples": metrics.analyzed_frames,
        "expected_samples": None,
        "live_drops": camera.dropped_stale,
        "measured_media_span_ms": (
            None
            if last_sampled_pts is None or started_pts is None
            else max(0, last_sampled_pts - started_pts)
        ),
        "source_stats": camera.stats.to_public(),
        "evidence_generation": {
            "requested": len(records),
            "failures": len(evidence_failures),
            "failure_reasons": evidence_failures,
        },
        "label": "actual_measured_result",
        "note": "Live coverage has no finite declared duration. Freshness drops are counted separately.",
    }
    sink.report_progress(
        _progress_payload(
            metrics=metrics,
            coverage=None,
            events_created=len(records),
            stage="finished",
            last_pts_ms=last_sampled_pts,
            duration_ms=None,
            source_stats=camera.stats.to_public(),
            queue={
                "capacity": camera.queue_frames,
                "dropped_stale": camera.dropped_stale,
                "connected": not camera.disconnected,
            },
        )
    )
    return RunResult(
        status=status,
        error_code=error_code,
        error_message=error_message,
        metrics=metrics,
        coverage=coverage,
        events_created=len(records),
        evidence_failures=len(evidence_failures),
        notes=notes,
    )
