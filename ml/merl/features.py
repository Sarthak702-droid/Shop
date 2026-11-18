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

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.merl import drive  # noqa: E402
from shop_core.config import load_settings  # noqa: E402
from shop_core.detection.person import PersonDetector  # noqa: E402

from shop_core.actions import merl_features as mf  # noqa: E402

SCHEMA = mf.SCHEMA
STEP = mf.STEP
DIMS = mf.DIMS
FEATURE_DIM = mf.FEATURE_DIM
DETECT_KEEP = mf.DETECT_KEEP
DRIVE_VIDEOS = "raw/merl_shopping/videos"
DRIVE_OUT = f"derived/features/{SCHEMA}"
expand = mf.expand
motion_features = mf.motion_features


class Extractor(mf.Extractor):
    """Training-side wrapper: loads its own hash-verified detector from ``configs/shop.yaml``."""

    def __init__(self, threads: int) -> None:
        settings = load_settings(REPO / "configs" / "shop.yaml")
        detector = PersonDetector.load(settings)  # hash-verified, local, no download
        torch.set_num_threads(threads)
        super().__init__(detector._model, detector.person_class_index, f"{detector.model_id}:{detector.version}")  # noqa: SLF001


def extract_video(ex: Extractor, video: Path) -> dict[str, np.ndarray]:
    """Training output format (unchanged): 30 fps only, float16 features."""
    try:
        out = mf.extract_video(ex, video, require_training_fps=True)
    except mf.FeatureError as error:
        raise RuntimeError(f"{video.name}: {error.message}") from error
    return {
        "features": out.features,
        "frame_index": out.frame_index,
        "person_box": out.person_box,
        "frame_count": np.int32(out.frame_count),
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
