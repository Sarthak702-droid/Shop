"""Source adapters. File and webcam share one frame contract; RTSP is deferred."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from shop_core.contracts import FrameEnvelope, PtsProvenance
from shop_core.zones import apply_privacy_mask


class SourceFault(RuntimeError):
    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class SourceStats:
    frames_decoded: int = 0
    frames_sampled: int = 0
    dropped_stale: int = 0
    dropped_schedule: int = 0
    gaps: int = 0
    reconnects: int = 0
    end_reason: str | None = None
    decode_incomplete: bool = False
    notes: list[str] = field(default_factory=list)

    def to_public(self) -> dict:
        return {
            "frames_decoded": self.frames_decoded,
            "frames_sampled": self.frames_sampled,
            "dropped_stale": self.dropped_stale,
            "dropped_schedule": self.dropped_schedule,
            "gaps": self.gaps,
            "reconnects": self.reconnects,
            "end_reason": self.end_reason,
            "decode_incomplete": self.decode_incomplete,
            "notes": list(self.notes),
        }


def capture_webcam_snapshot(
    device_index: int,
    *,
    max_long_edge: int = 960,
    privacy_polygon: list[list[float]] | None = None,
) -> tuple[bytes, dict]:
    """Capture one local preview JPEG and release the device immediately.

    Opening a second handle while the analysis worker owns the camera may fail;
    callers receive a typed fault rather than a fabricated preview.
    """
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        capture.release()
        raise SourceFault(
            "CAMERA_UNAVAILABLE",
            "Camera could not be opened. It may be disconnected, busy or permission-restricted.",
        )
    try:
        ok, image = capture.read()
        if not ok or image is None:
            raise SourceFault("CAMERA_NO_FRAME", "Camera opened but did not deliver a decodable frame.")
        height, width = image.shape[:2]
        scale = min(1.0, max_long_edge / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        apply_privacy_mask(image, privacy_polygon or [])
        encoded_ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not encoded_ok:
            raise SourceFault("CAMERA_ENCODE_FAILED", "Camera frame could not be encoded for preview.")
        return encoded.tobytes(), {
            "backend": capture.getBackendName(),
            "width": width,
            "height": height,
            "preview_width": int(image.shape[1]),
            "preview_height": int(image.shape[0]),
            "captured_at_utc": utc_now().isoformat().replace("+00:00", "Z"),
        }
    finally:
        capture.release()


def probe_video(path: Path) -> dict:
    """Inspect a local file with the decoder and real frame reads."""
    if not path.is_file():
        raise SourceFault("MEDIA_MISSING", "Media file is not available for probing.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise SourceFault(
            "MEDIA_UNSUPPORTED",
            "The file could not be opened by the local decoder. Codec or container is unsupported.",
        )
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        backend = capture.getBackendName()
        ok, frame = capture.read()
        if not ok or frame is None:
            raise SourceFault("MEDIA_DECODE_FAILED", "The file opened but its first frame could not be decoded.")
        actual_h, actual_w = frame.shape[:2]
        pts_probe: list[float] = []
        for _ in range(4):
            pos = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            pts_probe.append(pos)
            capture.grab()
        codec_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = "".join(chr((codec_fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") or "unknown"
        notes: list[str] = []
        if not (0.5 <= fps <= 240):
            notes.append("Nominal FPS is absent or implausible; media timestamps will be labelled.")
        increasing = all(b > a for a, b in zip(pts_probe, pts_probe[1:]))
        if not increasing:
            notes.append("Decoder POS_MSEC is not monotonic on the probe frames; frame-index fallback may be used.")
        duration_ms = None
        if frame_count_raw > 1 and fps > 0:
            duration_ms = int(round(frame_count_raw / fps * 1000.0))
        first_frame_decoded = True
        return {
            "container": path.suffix.lower().lstrip(".") or "unknown",
            "codec": codec,
            "width": actual_w,
            "height": actual_h,
            "reported_width": width,
            "reported_height": height,
            "nominal_fps": fps if fps > 0 else None,
            "frame_count": int(frame_count_raw) if frame_count_raw > 0 else None,
            "duration_ms": duration_ms,
            "first_frame_decoded": first_frame_decoded,
            "probe_backend": backend,
            "pts_probe": pts_probe,
            "pts_looks_monotonic": increasing,
            "notes": notes,
        }
    finally:
        capture.release()


class FileVideoSource:
    """Sequential file decoder that reports the timestamp provenance it used."""

    def __init__(self, path: Path, *, source_id: str, session_id: str) -> None:
        self.path = path
        self.source_id = source_id
        self.session_id = session_id
        self.stats = SourceStats()
        self._capture: cv2.VideoCapture | None = None
        self._nominal_fps: float | None = None
        self._frame_count: int | None = None
        self._pts_mode: PtsProvenance = "media_pts"
        self._last_pts = -1
        self._sequence = 0
        self._first_monotonic_ns = time.monotonic_ns()

    def open(self) -> None:
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise SourceFault("MEDIA_OPEN_FAILED", "Video could not be opened by the local decoder.")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._nominal_fps = fps if fps > 0 else None
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._frame_count = count if count > 0 else None
        # Probe whether POS_MSEC is monotonic using a short grabfest.
        positions: list[float] = []
        for _ in range(8):
            ok = self._capture.grab()
            if not ok:
                break
            positions.append(float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0))
        increasing = len(positions) >= 2 and all(b > a for a, b in zip(positions, positions[1:]))
        self._pts_mode = "media_pts" if increasing else "frame_index_nominal_fps"
        if not increasing:
            self.stats.notes.append(
                "Media PTS unavailable; timestamps use frame index divided by nominal FPS and are labelled."
            )
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def frames(self, should_stop: Callable[[], bool] | None = None) -> Iterator[FrameEnvelope]:
        if self._capture is None:
            raise SourceFault("SOURCE_NOT_OPEN", "Source frames requested before open().")
        expected = self._frame_count
        while True:
            if should_stop is not None and should_stop():
                self.stats.end_reason = "cancelled"
                break
            read_started = time.perf_counter()
            ok, image = self._capture.read()
            decode_ms = (time.perf_counter() - read_started) * 1000.0
            if not ok or image is None:
                self.stats.end_reason = "eof"
                if expected and self.stats.frames_decoded < expected - max(2, int(expected * 0.02)):
                    self.stats.decode_incomplete = True
                    self.stats.notes.append(
                        f"Decoded {self.stats.frames_decoded} of {expected} declared frames before EOF."
                    )
                break
            self.stats.frames_decoded += 1
            pos = float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            provenance: PtsProvenance
            if self._pts_mode == "media_pts" and pos > self._last_pts:
                pts_ms = int(round(pos))
                provenance = "media_pts"
            elif self._nominal_fps:
                pts_ms = int(round((self.stats.frames_decoded - 1) / self._nominal_fps * 1000.0))
                provenance = "frame_index_nominal_fps"
            else:
                pts_ms = int((time.monotonic_ns() - self._first_monotonic_ns) / 1_000_000)
                provenance = "monotonic_capture"
            discontinuity = False
            if self._last_pts >= 0 and pts_ms <= self._last_pts:
                discontinuity = True
                self.stats.gaps += 1
                pts_ms = self._last_pts + 1
            self._last_pts = pts_ms
            h, w = image.shape[:2]
            yield FrameEnvelope(
                source_id=self.source_id,
                session_id=self.session_id,
                sequence=self._sequence,
                pts_ms=pts_ms,
                pts_provenance=provenance,
                received_monotonic_ns=time.monotonic_ns(),
                width=w,
                height=h,
                image_bgr=image,
                captured_at_utc=utc_now().isoformat().replace("+00:00", "Z"),
                discontinuity=discontinuity,
                decode_ms=decode_ms,
            )
            self._sequence += 1


class LiveFrameBuffer:
    """Bounded freshness-first queue for live capture.

    The camera producer overwrites the oldest frame when the queue is full;
    every overwrite is counted and surfaced. It never blocks the camera.
    """

    def __init__(self, max_frames: int = 2) -> None:
        self._frames: deque[FrameEnvelope] = deque(maxlen=max_frames)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._stop = threading.Event()
        self.dropped = 0
        self.captured = 0
        self.disconnected = False
        self.last_error: str | None = None

    def push(self, frame: FrameEnvelope) -> None:
        with self._not_empty:
            if len(self._frames) == self._frames.maxlen:
                self._frames.popleft()
                self.dropped += 1
            self._frames.append(frame)
            self.captured += 1
            self._not_empty.notify()

    def latest(self, timeout: float = 0.5) -> FrameEnvelope | None:
        with self._not_empty:
            if not self._frames:
                self._not_empty.wait(timeout)
            if not self._frames:
                return None
            frame = self._frames.pop()
            self._frames.clear()
            return frame

    def stop(self) -> None:
        self._stop.set()
        with self._not_empty:
            self._not_empty.notify_all()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()


class WebcamSource:
    """Live camera adapter with actual negotiated capability reporting."""

    def __init__(
        self,
        device_index: int,
        *,
        source_id: str,
        session_id: str,
        requested_width: int = 1280,
        requested_height: int = 720,
        requested_fps: float = 30.0,
        queue_frames: int = 2,
    ) -> None:
        self.device_index = device_index
        self.source_id = source_id
        self.session_id = session_id
        self.requested = (requested_width, requested_height, requested_fps)
        self.queue_frames = queue_frames
        self.stats = SourceStats()
        self.capability: dict = {}
        self._capture: cv2.VideoCapture | None = None
        self._buffer = LiveFrameBuffer(queue_frames)
        self._producer: threading.Thread | None = None
        self._sequence = 0
        self._first_ns = time.monotonic_ns()

    @staticmethod
    def probe(device_index: int) -> dict:
        if not Path(f"/dev/video{device_index}").exists():
            return {
                "device_index": device_index,
                "device_present": False,
                "opened": False,
                "reason": "Device node is not present.",
            }
        capture = cv2.VideoCapture(device_index)
        try:
            if not capture.isOpened():
                return {
                    "device_index": device_index,
                    "device_present": True,
                    "opened": False,
                    "reason": "Device exists but could not be opened (permission or busy).",
                }
            ok, frame = capture.read()
            capability = {
                "device_index": device_index,
                "device_present": True,
                "opened": True,
                "frame_read": bool(ok and frame is not None),
                "backend": capture.getBackendName(),
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "reported_fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
                "reason": None,
            }
            if ok and frame is not None:
                capability["decoded_width"] = int(frame.shape[1])
                capability["decoded_height"] = int(frame.shape[0])
            return capability
        finally:
            capture.release()

    def open(self) -> None:
        self._capture = cv2.VideoCapture(self.device_index)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise SourceFault(
                "CAMERA_UNAVAILABLE",
                "Camera could not be opened. It may be disconnected, busy or permission-restricted.",
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested[1])
        self._capture.set(cv2.CAP_PROP_FPS, self.requested[2])
        self.capability = {
            "backend": self._capture.getBackendName(),
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "reported_fps": float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0),
            "requested": {
                "width": self.requested[0],
                "height": self.requested[1],
                "fps": self.requested[2],
            },
            "note": "Negotiated values are reported by the local backend and are not a lighting or viewpoint check.",
        }
        self._producer = threading.Thread(target=self._capture_loop, name="shop-camera", daemon=True)
        self._producer.start()

    def _capture_loop(self) -> None:
        failures = 0
        while not self._buffer.stopped and self._capture is not None:
            ok, image = self._capture.read()
            if not ok or image is None:
                failures += 1
                if failures == 1:
                    self.stats.notes.append("Camera read failure; attempting bounded reconnect.")
                if failures > 5:
                    self._buffer.disconnected = True
                    self._buffer.last_error = "Camera stopped delivering frames."
                    break
                time.sleep(min(0.1 * failures, 0.5))
                continue
            failures = 0
            now_ns = time.monotonic_ns()
            pts_ms = int((now_ns - self._first_ns) / 1_000_000)
            h, w = image.shape[:2]
            self.stats.frames_decoded += 1
            self._buffer.push(
                FrameEnvelope(
                    source_id=self.source_id,
                    session_id=self.session_id,
                    sequence=self._sequence,
                    pts_ms=pts_ms,
                    pts_provenance="monotonic_capture",
                    received_monotonic_ns=now_ns,
                    width=w,
                    height=h,
                    image_bgr=image,
                    captured_at_utc=utc_now().isoformat().replace("+00:00", "Z"),
                )
            )
            self._sequence += 1
        self._buffer.stop()

    def next_frame(self, timeout: float = 0.5) -> FrameEnvelope | None:
        if self._buffer.disconnected:
            raise SourceFault("CAMERA_DISCONNECTED", self._buffer.last_error or "Camera disconnected.")
        frame = self._buffer.latest(timeout)
        if frame is not None:
            self.stats.frames_sampled += 1
        self.stats.dropped_stale = self._buffer.dropped
        return frame

    def close(self) -> None:
        self._buffer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._producer is not None:
            self._producer.join(timeout=2.0)
            self._producer = None

    @property
    def disconnected(self) -> bool:
        return self._buffer.disconnected

    @property
    def dropped_stale(self) -> int:
        return self._buffer.dropped
