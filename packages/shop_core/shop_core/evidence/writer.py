"""Atomic sampled-evidence clip generation.

Clips are encoded locally as VP8 WebM through the OpenCV wheel's bundled
FFmpeg. They contain only the sampled analysis cadence plus context, with
boxes and track IDs burned in for reviewer orientation. A failed encode is
reported as unavailable; no empty or fabricated file is published.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from shop_core.contracts import Detection, TrackObservation


class EvidenceError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class EvidenceFrame:
    pts_ms: int
    image_bgr: np.ndarray
    detections: list[Detection]
    tracks: list[TrackObservation]


@dataclass
class EvidenceClip:
    clip_id: str
    path: Path
    sha256: str
    bytes: int
    start_pts_ms: int
    end_pts_ms: int
    duration_ms: int
    width: int
    height: int
    fps: float
    frame_count: int
    mime_type: str
    encoder: str
    generate_ms: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_frame(image: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (max(2, int(round(w * scale))), max(2, int(round(h * scale)))))


def annotate_frame(frame: EvidenceFrame, size: tuple[int, int], label: str = "Sampled evidence") -> np.ndarray:
    image = _fit_frame(frame.image_bgr.copy(), max(size))
    height, width = image.shape[:2]
    if (width, height) != size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    sx = size[0] / frame.image_bgr.shape[1]
    sy = size[1] / frame.image_bgr.shape[0]
    for detection in frame.detections:
        x1, y1, x2, y2 = detection.box
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(image, p1, p2, (120, 200, 170), 2)
    for track in frame.tracks:
        if not track.observed or track.status.value not in {"confirmed", "tentative"}:
            continue
        x1, y1, x2, y2 = track.box
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        color = (210, 190, 60) if track.status.value == "confirmed" else (150, 150, 150)
        cv2.rectangle(image, p1, p2, color, 2)
        label_text = f"{track.track_id} {track.score:.2f}"
        cv2.putText(
            image,
            label_text,
            (max(2, p1[0]), max(14, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(image, (0, 0), (size[0], 24), (20, 33, 44), -1)
    cv2.putText(
        image,
        f"{label}  media {frame.pts_ms / 1000:.1f}s",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (225, 235, 238),
        1,
        cv2.LINE_AA,
    )
    return image


class EvidenceWriter:
    def __init__(self, root: Path, *, max_long_edge: int = 960, output_fps: float = 10.0) -> None:
        self.root = root
        self.max_long_edge = max_long_edge
        self.output_fps = output_fps

    def write_clip(
        self,
        *,
        event_id: str,
        frames: list[EvidenceFrame],
        start_pts_ms: int,
        end_pts_ms: int,
    ) -> EvidenceClip:
        if not frames:
            raise EvidenceError("EVIDENCE_NO_FRAMES", "No sampled frames are available for this evidence window.")
        if end_pts_ms <= start_pts_ms:
            raise EvidenceError("EVIDENCE_INVALID_WINDOW", "Evidence window has no positive media duration.")
        started = time.perf_counter()
        first = _fit_frame(frames[0].image_bgr, self.max_long_edge)
        height, width = first.shape[:2]
        width -= width % 2
        height -= height % 2
        size = (max(2, width), max(2, height))
        target = self.root / event_id
        target.mkdir(parents=True, exist_ok=True)
        clip_id = str(uuid.uuid4())
        final_path = target / f"{clip_id}.webm"
        partial = target / f".{clip_id}.partial.webm"
        writer = cv2.VideoWriter(
            str(partial), cv2.VideoWriter_fourcc(*"VP80"), self.output_fps, size
        )
        if not writer.isOpened():
            partial.unlink(missing_ok=True)
            raise EvidenceError("EVIDENCE_ENCODER_UNAVAILABLE", "Local VP8 encoder is unavailable.")
        written = 0
        try:
            previous_pts = frames[0].pts_ms
            for index, frame in enumerate(frames):
                if index + 1 < len(frames):
                    next_pts = frames[index + 1].pts_ms
                else:
                    next_pts = max(frame.pts_ms, end_pts_ms)
                hold = int(round(max(0.0, next_pts - previous_pts) / 1000.0 * self.output_fps))
                hold = max(1, min(hold, int(self.output_fps * 5)))
                annotated = annotate_frame(frame, size)
                for _ in range(hold):
                    writer.write(annotated)
                    written += 1
                previous_pts = next_pts
            writer.release()
            if written == 0:
                raise EvidenceError("EVIDENCE_EMPTY", "Evidence encoder wrote no frames.")
            verification = cv2.VideoCapture(str(partial))
            ok, _ = verification.read()
            verification.release()
            if not ok:
                raise EvidenceError("EVIDENCE_VERIFY_FAILED", "Encoded evidence could not be decoded back.")
            os.replace(partial, final_path)
        except Exception:
            writer.release()
            partial.unlink(missing_ok=True)
            raise

        duration_ms = int(round(written / self.output_fps * 1000.0))
        self._write_thumbnail(target, clip_id, frames)
        return EvidenceClip(
            clip_id=clip_id,
            path=final_path,
            sha256=_sha256(final_path),
            bytes=final_path.stat().st_size,
            start_pts_ms=start_pts_ms,
            end_pts_ms=end_pts_ms,
            duration_ms=duration_ms,
            width=size[0],
            height=size[1],
            fps=self.output_fps,
            frame_count=written,
            mime_type="video/webm",
            encoder="opencv_vp8",
            generate_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    def _write_thumbnail(self, target: Path, clip_id: str, frames: list[EvidenceFrame]) -> Path:
        frame = frames[len(frames) // 2]
        thumbnail = annotate_frame(frame, (480, 270), label="Sampled evidence")
        path = target / f"{clip_id}.jpg"
        ok = cv2.imwrite(str(path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            raise EvidenceError("EVIDENCE_THUMBNAIL_FAILED", "Thumbnail could not be written.")
        return path
