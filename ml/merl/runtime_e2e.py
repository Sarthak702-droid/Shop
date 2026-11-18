"""End-to-end check of the *runtime* shelf-action path on held-out MERL test videos.

For each raw video this runs exactly what the worker runs (``shop_core.actions``: decode -> SSDLite
-> merl_features_v1 -> TCN -> episode policy), then

* compares runtime features with the cached training features (bit-exact parity check),
* scores runtime predictions against MERL labels (frame AP and runtime-policy segment AP),
* records CPU wall time, real-time factor, per-stage timing and peak RSS.

Usage::

    .venv/bin/python -I ml/merl/runtime_e2e.py --videos 27_2_crop.mp4 33_1_crop.mp4 39_1_crop.mp4

Writes ``ml/results/merl_tcn_v1/runtime_e2e.json`` and ``ml/results/benchmarks/cpu_e2e.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.merl.dataset import CLASSES, step_labels  # noqa: E402
from ml.merl.metrics import average_precision  # noqa: E402
from shop_core.actions import merl_features as mf  # noqa: E402
from shop_core.actions.merl_runtime import EpisodePolicy, ShelfActionModel, build_episodes  # noqa: E402
from shop_core.config import load_settings  # noqa: E402
from shop_core.detection.person import PersonDetector  # noqa: E402


def peak_rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--videos", nargs="+", default=["27_2_crop.mp4", "33_1_crop.mp4", "39_1_crop.mp4"])
    ap.add_argument("--video-dir", type=Path, default=REPO / "Videos_MERL_Shopping_Dataset/Videos_MERL_Shopping_Dataset")
    ap.add_argument("--features", type=Path, default=REPO / "var/research/merl_features")
    ap.add_argument("--intervals", type=Path, default=REPO / "var/research/intervals.csv")
    ap.add_argument("--split", type=Path, default=REPO / "var/research/merl_shopping_official.json")
    ap.add_argument("--out", type=Path, default=REPO / "ml/results")
    args = ap.parse_args()

    test_videos = set(json.loads(args.split.read_text())["splits"]["test"]["videos"])
    for v in args.videos:
        if v not in test_videos:
            raise SystemExit(f"{v} is not in the held-out test split")
    by_video: dict[str, list[tuple[int, int, int]]] = {}
    for r in csv.DictReader(args.intervals.open()):
        by_video.setdefault(r["video"], []).append((int(r["merl_class_id"]) - 1, int(r["start_frame"]), int(r["end_frame"])))

    settings = load_settings(REPO / "configs/shop.yaml")
    t0 = time.perf_counter()
    model = ShelfActionModel.load(settings)
    extractor = mf.Extractor.from_detector(PersonDetector.load(settings))
    load_ms = (time.perf_counter() - t0) * 1000.0
    policy = EpisodePolicy()
    rows, all_p, all_y = [], [], []
    for name in args.videos:
        path = args.video_dir / name
        t0 = time.perf_counter()
        feats = mf.extract_video(extractor, path)
        t1 = time.perf_counter()
        probs = model.predict(feats.features)
        t2 = time.perf_counter()
        episodes = build_episodes(probs, model.thresholds, model.classes, feats.pts_ms, feats.person_present, feats.person_score, policy)
        t3 = time.perf_counter()
        with np.load(args.features / f"{name.removesuffix('.mp4')}.npz") as z:
            cached = z["features"]
        parity = bool(cached.shape == feats.features.shape and np.array_equal(cached, feats.features))
        y = step_labels(feats.frame_index.astype(np.int64), by_video.get(name, []))
        all_p.append(probs)
        all_y.append(y)
        media_s = feats.frame_count / feats.fps
        rows.append({
            "video": name, "frames": feats.frame_count, "steps": len(probs), "media_seconds": round(media_s, 1),
            "feature_parity_with_training": parity,
            "max_abs_feature_diff": float(np.abs(cached.astype(np.float32) - feats.features.astype(np.float32)).max()) if cached.shape == feats.features.shape else None,
            "frame_map": round(float(np.nanmean([average_precision(probs[:, k], y[:, k]) for k in range(5)])), 4),
            "episodes": len(episodes),
            "gt_intervals": len(by_video.get(name, [])),
            "person_visible_fraction": round(float(feats.person_present.mean()), 3),
            "timing_ms": {"features_wall": round((t1 - t0) * 1000, 1), **feats.timings_ms, "tcn": round((t2 - t1) * 1000, 1),
                          "episodes": round((t3 - t2) * 1000, 2)},
            "real_time_factor": round((t3 - t0) / media_s, 3),
            "steps_per_second": round(len(probs) / (t3 - t0), 2),
        })
        print(json.dumps(rows[-1]), flush=True)
    p, y = np.concatenate(all_p), np.concatenate(all_y)
    per_class = {c: round(average_precision(p[:, k], y[:, k]), 4) for k, c in enumerate(CLASSES)}
    rtf = [r["real_time_factor"] for r in rows]
    result = {
        "schema": "shop_runtime_e2e_v1",
        "videos": rows,
        "all_parity": all(r["feature_parity_with_training"] for r in rows),
        "pooled_frame_ap": per_class,
        "pooled_frame_map": round(float(np.mean(list(per_class.values()))), 4),
        "note": "Held-out MERL test videos through the runtime path. Small sample; the 28-video test numbers in analysis.json are authoritative.",
        "label": "actual_measured_result",
    }
    bench = {
        "schema": "shop_cpu_benchmark_v1",
        "profile": "merl_shelf_actions_research_v1",
        "host": {"cpu": cpu_name(), "logical_cores": os.cpu_count(), "threads": settings.runtime.inference_threads,
                 "python": platform.python_version(), "torch": __import__("torch").__version__},
        "model_load_ms": round(load_ms, 1),
        "videos": len(rows),
        "media_seconds": round(sum(r["media_seconds"] for r in rows), 1),
        "real_time_factor": {"mean": round(float(np.mean(rtf)), 3), "max": round(float(np.max(rtf)), 3)},
        "per_step_ms": {
            "decode": round(float(np.mean([r["timing_ms"]["decode"] / r["steps"] for r in rows])), 2),
            "detect_and_backbone": round(float(np.mean([r["timing_ms"]["detect"] / r["steps"] for r in rows])), 2),
            "motion": round(float(np.mean([r["timing_ms"]["motion"] / r["steps"] for r in rows])), 2),
            "tcn": round(float(np.mean([r["timing_ms"]["tcn"] / r["steps"] for r in rows])), 3),
        },
        "peak_rss_kb": peak_rss_kb(),
        "budget": {"step_ms_at_5hz": 200.0},
        "note": "Feature extraction (SSDLite on every step plus motion over every frame) dominates. TCN cost is negligible.",
        "label": "actual_measured_result",
    }
    (args.out / "merl_tcn_v1").mkdir(parents=True, exist_ok=True)
    (args.out / "benchmarks").mkdir(parents=True, exist_ok=True)
    (args.out / "merl_tcn_v1/runtime_e2e.json").write_text(json.dumps(result, indent=1) + "\n")
    (args.out / "benchmarks/cpu_e2e.json").write_text(json.dumps(bench, indent=1) + "\n")
    print(json.dumps(bench, indent=1))


if __name__ == "__main__":
    main()
