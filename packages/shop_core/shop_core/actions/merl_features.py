"""``merl_features_v1``: the per-step feature contract shared by MERL training and Shop runtime.

One *step* is 1/5 s (Shop's 5 Hz analysis cadence). At 30 fps a step is 6 frames, exactly as the
MERL research checkpoint was trained. For step ``i`` (frames ``s*i .. s*i+s-1``):

* ``roi``     672  SSDLite backbone map "0" (stride 16), ROI-aligned on the largest person box x1.4
* ``global``  480  SSDLite backbone map "1" (stride 32), global average
* ``geom``     11  person present, count >= 0.5, score, cx, cy, w, h (normalised), dcx, dcy, dw, dh
* ``motion``   38  frame-difference energy from every frame of the step: 4x4 grid mean (16), 4x4 grid
                   max (16), global (1), inside box (1), ring around box left/right/top/bottom (4)

The detector runs once per step on the first frame; one forward pass gives boxes and backbone maps.
Motion uses every decoded frame, so short reaches are not missed between samples.

Training (``ml/merl/features.py``) and the runtime worker both import this module, so there is a
single implementation of the contract. For sources that are not 30 fps the step length becomes
``round(fps / 5)`` frames; that changes the motion statistics and is reported as a domain warning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align

SCHEMA = "merl_features_v1"
TRAIN_FPS = 30.0
RATE_HZ = 5.0
STEP = 6  # frames per step at the training frame rate
SMALL = (92, 68)  # motion grid working resolution (w, h), 1/10 of 920x680
DIMS = {"roi": 672, "global": 480, "geom": 11, "motion": 38}
FEATURE_DIM = sum(DIMS.values())
GEOM_OFFSET = DIMS["roi"] + DIMS["global"]
# Overhead view: the COCO detector scores people low (0.15 -> ~55% of MERL steps vs 33% at 0.3).
DETECT_KEEP = 0.15
TRAIN_RESOLUTION = (920, 680)


class FeatureError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Extractor:
    """Wraps a *dedicated* SSDLite network: it lowers ``score_thresh``, so never share it."""

    def __init__(self, model: torch.nn.Module, person_class_index: int, model_ref: str) -> None:
        self.model = model
        # Only persons >= DETECT_KEEP are used. Raising the post-processing floor from 0.01 skips
        # per-class NMS over thousands of low-score anchors (measured: ~40% of runtime).
        self.model.score_thresh = DETECT_KEEP
        self.person = person_class_index
        self.model_ref = model_ref
        self.maps: dict[str, torch.Tensor] = {}
        self._hook = self.model.backbone.register_forward_hook(lambda _m, _i, out: self.maps.update(out))

    @classmethod
    def from_detector(cls, detector) -> "Extractor":  # noqa: ANN001 - PersonDetector, avoids import cycle
        return cls(detector._model, detector.person_class_index, f"{detector.model_id}:{detector.version}")  # noqa: SLF001

    def close(self) -> None:
        self._hook.remove()

    @torch.inference_mode()
    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ROI + global features, person boxes (original pixels) and scores from one pass.

        The frame is resized to the network's 320x320 input here, so TorchVision's own resize is a
        no-op and boxes come back in 320 space; they are scaled back to original pixels.
        """
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (320, 320), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        out = self.model([tensor])[0]
        keep = (out["labels"] == self.person) & (out["scores"] >= DETECT_KEEP)
        boxes320 = out["boxes"][keep]
        scores = out["scores"][keep].numpy()
        f0 = self.maps["0"]  # (1, 672, 20, 20)
        glob = self.maps["1"].mean(dim=(2, 3))[0].numpy()
        scale = np.array([w / 320, h / 320, w / 320, h / 320], dtype=np.float32)
        boxes = boxes320.numpy() * scale
        if len(boxes):
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            roi_box = expand(boxes[int(np.argmax(areas))], 1.4, w, h) / scale
            rois = torch.tensor([[0.0, *roi_box.tolist()]], dtype=torch.float32)
            roi = roi_align(f0, rois, output_size=1, spatial_scale=20 / 320, aligned=True)[0, :, 0, 0].numpy()
        else:
            roi = np.zeros(DIMS["roi"], dtype=np.float32)
        return roi, glob, boxes, scores


def expand(box: np.ndarray, factor: float, w: int, h: int) -> np.ndarray:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = (box[2] - box[0]) * factor, (box[3] - box[1]) * factor
    return np.array([max(0, cx - bw / 2), max(0, cy - bh / 2), min(w, cx + bw / 2), min(h, cy + bh / 2)], dtype=np.float32)


def motion_features(diffs: list[np.ndarray], box: np.ndarray | None, w: int, h: int) -> np.ndarray:
    """``diffs``: per-frame |gray_t - gray_{t-1}| at SMALL resolution, values 0..1."""
    if not diffs:
        return np.zeros(DIMS["motion"], dtype=np.float32)
    stack = np.stack(diffs)  # (n, 68, 92)
    gh, gw = stack.shape[1] // 4, stack.shape[2] // 4
    cells = stack[:, : gh * 4, : gw * 4].reshape(len(diffs), 4, gh, 4, gw).mean(axis=(2, 4)).reshape(len(diffs), 16)
    mean_img = stack.mean(axis=0)
    feats = [cells.mean(axis=0), cells.max(axis=0), [mean_img.mean()]]
    if box is None:
        feats.append(np.zeros(5, dtype=np.float32))
    else:
        sx, sy = SMALL[0] / w, SMALL[1] / h
        x0, y0, x1, y1 = (int(box[0] * sx), int(box[1] * sy), int(np.ceil(box[2] * sx)), int(np.ceil(box[3] * sy)))
        ex = expand(box, 1.6, w, h)
        X0, Y0, X1, Y1 = (int(ex[0] * sx), int(ex[1] * sy), int(np.ceil(ex[2] * sx)), int(np.ceil(ex[3] * sy)))

        def m(a: np.ndarray) -> float:
            return float(a.mean()) if a.size else 0.0

        feats.append([
            m(mean_img[y0:y1, x0:x1]),
            m(mean_img[Y0:Y1, X0:x0]),
            m(mean_img[Y0:Y1, x1:X1]),
            m(mean_img[Y0:y0, x0:x1]),
            m(mean_img[y1:Y1, x0:x1]),
        ])
    return np.concatenate([np.asarray(f, dtype=np.float32).ravel() for f in feats])


@dataclass
class VideoFeatures:
    features: np.ndarray  # (T, FEATURE_DIM) float16, the stored training dtype
    frame_index: np.ndarray  # (T,) first frame of each step
    pts_ms: np.ndarray  # (T,) media time of each step's first frame
    person_box: np.ndarray  # (T, 4) largest person box in original pixels, NaN when absent
    person_score: np.ndarray  # (T,)
    frame_count: int
    fps: float
    step_frames: int
    width: int
    height: int
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def step_ms(self) -> float:
        return 1000.0 / RATE_HZ

    @property
    def person_present(self) -> np.ndarray:
        return self.features[:, GEOM_OFFSET].astype(np.float32) > 0.5


def step_frames_for(fps: float) -> int:
    return max(1, int(round(fps / RATE_HZ)))


def extract_video(
    ex: Extractor,
    video: Path,
    *,
    require_training_fps: bool = False,
    frame_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    on_step: Callable[[int, int, int], bool] | None = None,
    max_frames: int | None = None,
) -> VideoFeatures:
    """Decode ``video`` once and return per-step ``merl_features_v1``.

    ``frame_transform`` runs on every decoded frame before detection and motion (privacy masks).
    ``on_step(steps_done, frame_index, pts_ms)`` is called after each step; returning ``True``
    stops extraction early (cancellation). At 30 fps the output equals the training extractor.
    """
    import time

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FeatureError("SOURCE_OPEN_FAILED", "The video could not be opened by the local decoder.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 1.0 or fps > 240.0:
        cap.release()
        raise FeatureError("SOURCE_FPS_INVALID", "The video does not declare a usable frame rate.")
    if require_training_fps and abs(fps - TRAIN_FPS) > 0.5:
        cap.release()
        raise FeatureError("SOURCE_FPS_UNSUPPORTED", f"Expected {TRAIN_FPS:.0f} fps, got {fps:.2f}.")
    step = STEP if abs(fps - TRAIN_FPS) <= 0.5 else step_frames_for(fps)
    rows: list[np.ndarray] = []
    frames: list[int] = []
    boxes_out: list[np.ndarray] = []
    scores_out: list[float] = []
    prev_small = None
    diffs: list[np.ndarray] = []
    pending = None
    prev_geom = None
    idx = 0
    size = (0, 0)
    detect_ms = 0.0
    decode_ms = 0.0
    motion_ms = 0.0
    stopped = False

    def flush() -> None:
        nonlocal prev_geom, motion_ms
        f_idx, roi, glob, best, count, score, w, h = pending
        if best is not None:
            cx, cy = (best[0] + best[2]) / 2 / w, (best[1] + best[3]) / 2 / h
            bw, bh = (best[2] - best[0]) / w, (best[3] - best[1]) / h
            cur = np.array([cx, cy, bw, bh], dtype=np.float32)
            delta = cur - prev_geom if prev_geom is not None else np.zeros(4, dtype=np.float32)
            geom = np.concatenate([[1.0, count, score], cur, delta])
            prev_geom = cur
        else:
            geom = np.zeros(DIMS["geom"], dtype=np.float32)
            prev_geom = None
        t0 = time.perf_counter()
        mot = motion_features(diffs, best, w, h)
        motion_ms += (time.perf_counter() - t0) * 1000.0
        rows.append(np.concatenate([roi, glob, geom.astype(np.float32), mot]).astype(np.float16))
        frames.append(f_idx)
        boxes_out.append(best if best is not None else np.full(4, np.nan, dtype=np.float32))
        scores_out.append(score)

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            decode_ms += (time.perf_counter() - t0) * 1000.0
            if not ok or frame is None:
                break
            if max_frames is not None and idx >= max_frames:
                break
            if frame_transform is not None:
                frame = frame_transform(frame)
            h, w = frame.shape[:2]
            size = (w, h)
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), SMALL, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            if idx % step == 0:
                if pending is not None:
                    flush()
                    if on_step is not None and on_step(len(rows), frames[-1], int(round(frames[-1] * 1000.0 / fps))):
                        stopped = True
                        pending = None
                        break
                diffs = []
                t0 = time.perf_counter()
                roi, glob, boxes, scores = ex.detect(frame)
                detect_ms += (time.perf_counter() - t0) * 1000.0
                if len(boxes):
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    bi = int(np.argmax(areas))
                    best = boxes[bi].astype(np.float32)
                    count = float((scores >= 0.5).sum())
                    score = float(scores[bi])
                else:
                    best, count, score = None, 0.0, 0.0
                pending = (idx, roi, glob, best, count, score, w, h)
            if prev_small is not None:
                diffs.append(np.abs(small - prev_small))
            prev_small = small
            idx += 1
        if pending is not None and not stopped:
            flush()
    finally:
        cap.release()
    if not rows:
        raise FeatureError("SOURCE_NO_FRAMES", "No frames could be decoded from the video.")
    frame_index = np.asarray(frames, dtype=np.int32)
    return VideoFeatures(
        features=np.stack(rows),
        frame_index=frame_index,
        pts_ms=np.round(frame_index * 1000.0 / fps).astype(np.int64),
        person_box=np.stack(boxes_out).astype(np.float32),
        person_score=np.asarray(scores_out, dtype=np.float32),
        frame_count=idx,
        fps=fps,
        step_frames=step,
        width=size[0],
        height=size[1],
        timings_ms={"decode": round(decode_ms, 1), "detect": round(detect_ms, 1), "motion": round(motion_ms, 1)},
    )
