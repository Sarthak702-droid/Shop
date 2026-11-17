"""Extract per-step features for every MERL Shopping video and store them on Google Drive.

One step = 6 frames at 30 fps (5 Hz, Shop's analysis cadence). For step ``i`` (frames ``6i .. 6i+5``):

* ``roi``     672  SSDLite backbone map "0" (stride 16), ROI-aligned on the largest person box x1.4
* ``global``  480  SSDLite backbone map "1" (stride 32), global average
* ``geom``     11  person present, count >= 0.5, score, cx, cy, w, h (normalised), dcx, dcy, dw, dh
* ``motion``   38  frame-difference energy from all 6 frames: 4x4 grid mean (16), 4x4 grid max (16),
                   global (1), inside box (1), ring around box left/right/top/bottom (4)

The detector runs once per step, on frame ``6i``. The same forward pass gives boxes and backbone maps.
Motion uses every decoded frame, so short reaches are not missed between samples.

Usage (repository root)::

    .venv/bin/python -I -m ml.merl.features                 # all videos, resumable
    .venv/bin/python -I -m ml.merl.features --limit 2       # smoke test

Videos stream from Drive one at a time into a temp dir that is deleted after each file.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.merl import drive  # noqa: E402
from shop_core.config import load_settings  # noqa: E402
from shop_core.detection.person import PersonDetector  # noqa: E402

SCHEMA = "merl_features_v1"
STEP = 6  # frames per step at 30 fps -> 5 Hz
SMALL = (92, 68)  # motion grid working resolution (w, h), 1/10 of 920x680
DIMS = {"roi": 672, "global": 480, "geom": 11, "motion": 38}
FEATURE_DIM = sum(DIMS.values())
DETECT_KEEP = 0.15  # overhead view: COCO detector scores people low (0.15 -> ~55% of steps vs 33% at 0.3); Shop events still use 0.5
DRIVE_VIDEOS = "raw/merl_shopping/videos"
DRIVE_OUT = f"derived/features/{SCHEMA}"


class Extractor:
    def __init__(self, threads: int) -> None:
        settings = load_settings(REPO / "configs" / "shop.yaml")
        detector = PersonDetector.load(settings)  # hash-verified, local, no download
        torch.set_num_threads(threads)
        self.model = detector._model  # noqa: SLF001 - reuse the verified network for one shared pass
        # Only persons >= DETECT_KEEP are used. Raising the post-processing floor from 0.01 skips
        # per-class NMS over thousands of low-score anchors (measured: ~40% of runtime).
        self.model.score_thresh = DETECT_KEEP
        self.person = detector.person_class_index
        self.model_ref = f"{detector.model_id}:{detector.version}"
        self.maps: dict[str, torch.Tensor] = {}
        self.model.backbone.register_forward_hook(lambda _m, _i, out: self.maps.update(out))

    @torch.inference_mode()
    def detect(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Boxes (original pixels) + scores for persons, and backbone features, from one pass.

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


def extract_video(ex: Extractor, video: Path) -> dict[str, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video.name}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(fps - 30.0) > 0.5:
        raise RuntimeError(f"{video.name}: expected 30 fps, got {fps}")
    rows, frames, boxes_out = [], [], []
    prev_small, diffs = None, []
    pending = None  # (frame_idx, roi, glob, best_box, count, score, w, h)
    prev_geom = None
    idx = 0

    def flush() -> None:
        nonlocal prev_geom
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
        mot = motion_features(diffs, best, w, h)
        rows.append(np.concatenate([roi, glob, geom.astype(np.float32), mot]).astype(np.float16))
        frames.append(f_idx)
        boxes_out.append(best if best is not None else np.full(4, np.nan, dtype=np.float32))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), SMALL, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if idx % STEP == 0:
            if pending is not None:
                flush()
            diffs = []
            roi, glob, boxes, scores = ex.detect(frame)
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
    if pending is not None:
        flush()
    cap.release()
    if not rows:
        raise RuntimeError(f"{video.name}: no frames decoded")
    return {
        "features": np.stack(rows),
        "frame_index": np.asarray(frames, dtype=np.int32),
        "person_box": np.stack(boxes_out).astype(np.float32),
        "frame_count": np.int32(idx),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--limit", type=int, default=0, help="process at most N pending videos")
    ap.add_argument("--only", nargs="*", default=None, help="explicit video names")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    videos = sorted(drive.list_files(DRIVE_VIDEOS), key=lambda n: tuple(int(p) for p in n.split("_")[:2]))
    done = {n.removesuffix(".npz") for n in drive.list_files(DRIVE_OUT) if n.endswith(".npz")}
    todo = [v for v in videos if (args.only is None or v in args.only) and v.removesuffix(".mp4") not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(videos)} videos on Drive, {len(done)} already extracted, {len(todo)} to do", flush=True)

    ex = Extractor(args.threads)
    schema = {
        "schema": SCHEMA, "step_frames": STEP, "fps": 30.0, "rate_hz": 5.0, "dims": DIMS, "feature_dim": FEATURE_DIM,
        "order": ["roi", "global", "geom", "motion"], "detector": ex.model_ref, "detect_keep": DETECT_KEEP,
        "label_frame": "frame_index + 3 (centre of the 6-frame step)",
        "note": "Research features for MERL Shopping; not shop_features_v1 (no pose).",
    }
    with tempfile.TemporaryDirectory(prefix="shop-merl-") as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "schema.json").write_text(json.dumps(schema, indent=2))
        drive.upload(tmpdir / "schema.json", f"{DRIVE_OUT}/schema.json")
        for n, name in enumerate(todo, 1):
            started = time.perf_counter()
            local = drive.download(f"{DRIVE_VIDEOS}/{name}", tmpdir / name)
            try:
                result = extract_video(ex, local)
            finally:
                local.unlink(missing_ok=True)
            out = tmpdir / f"{name.removesuffix('.mp4')}.npz"
            np.savez_compressed(out, **result, schema=np.array(SCHEMA))
            drive.upload(out, f"{DRIVE_OUT}/{out.name}")
            out.unlink()
            secs = time.perf_counter() - started
            print(f"[{n}/{len(todo)}] {name}: {len(result['features'])} steps, "
                  f"{result['frame_count']} frames, {secs:.1f}s", flush=True)


if __name__ == "__main__":
    main()
