"""Gate check: can Shop's raw-video pose path feed a model trained on RetailAction's supplied poses?

Raw video -> person track -> pose extraction -> matching features -> action model only works if
the extracted joints mean the same thing as the training joints. This script measures:

1. Joint-set compatibility: RetailAction's 24 named joints vs the COCO-17 order every TorchVision
   keypoint model emits (``shop_core.contracts.COCO_KEYPOINT_ORDER``).
2. Coordinate and confidence semantics of the supplied poses (streamed from the verified original
   ``train.tar``; nothing is extracted to disk).
3. CPU latency of TorchVision Keypoint R-CNN (ResNet-50 FPN) on this host. Weights are randomly
   initialised because no approved pose checkpoint is installed; the network cost is the same, but
   this is a *timing* measurement only, never a pose-quality measurement.

Usage::

    .venv/bin/python -I ml/retailaction/pose_compat.py --archive /path/to/train.tar --samples 150
    .venv/bin/python -I ml/retailaction/pose_compat.py   # joint-set + CPU timing only

Writes ``ml/results/retailaction/pose_compatibility.json``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.retailaction.data import JOINTS as RA_JOINTS  # noqa: E402
from shop_core.contracts import COCO_KEYPOINT_ORDER  # noqa: E402

DERIVABLE = {"neck": ("left_shoulder", "right_shoulder"), "middle_of_waist": ("left_hip", "right_hip")}
ACTION_CRITICAL = ("left_hand", "right_hand", "left_wrist", "right_wrist", "left_elbow", "right_elbow")


def joint_mapping() -> dict:
    coco = set(COCO_KEYPOINT_ORDER)
    direct = [j for j in RA_JOINTS if j in coco]
    derivable = [j for j in RA_JOINTS if j in DERIVABLE]
    missing = [j for j in RA_JOINTS if j not in coco and j not in DERIVABLE]
    return {
        "retailaction_joints": list(RA_JOINTS),
        "coco17_joints": list(COCO_KEYPOINT_ORDER),
        "direct": direct,
        "derivable_as_midpoint": {j: list(DERIVABLE[j]) for j in derivable},
        "missing_in_coco17": missing,
        "action_critical_missing": [j for j in ACTION_CRITICAL if j in missing],
        "coverage": round((len(direct) + len(derivable)) / len(RA_JOINTS), 3),
    }


def supplied_pose_stats(archive: Path, limit: int) -> dict:
    xs, ys, scores = [], [], []
    per_joint_valid: dict[str, list[int]] = {j: [0, 0] for j in RA_JOINTS}
    samples = poses = null_poses = 0
    frame_shapes = set()
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.name != "metadata.json" or ".." in path.parts or member.size > 10 * 1024 * 1024:
                continue
            content = json.load(tar.extractfile(member))["content"]
            for rank in ("rank0", "rank1"):
                cam = content["action_cam"][rank]
                for pose in cam.get("poses") or []:
                    if pose is None:
                        null_poses += 1
                        continue
                    poses += 1
                    coords = pose.get("pose") or {}
                    js = pose.get("joint_scores") or {}
                    for j in RA_JOINTS:
                        per_joint_valid[j][1] += 1
                        v = coords.get(j)
                        if v is None or len(v) != 2 or not all(np.isfinite(v)):
                            continue
                        per_joint_valid[j][0] += 1
                        xs.append(v[0])
                        ys.append(v[1])
                        if js.get(j) is not None:
                            scores.append(js[j])
            samples += 1
            if samples >= limit:
                break
    xs, ys, scores = (np.asarray(a, dtype=np.float64) for a in (xs, ys, scores))

    def pct(a: np.ndarray) -> dict:
        return {f"p{q}": round(float(np.percentile(a, q)), 4) for q in (0, 1, 50, 99, 100)} if a.size else {}

    on_grid = float(np.mean(np.isclose(xs * 64, np.round(xs * 64)) & np.isclose(ys * 64, np.round(ys * 64)))) if xs.size else None
    del frame_shapes
    return {
        "archive": archive.name,
        "samples_read": samples,
        "pose_observations": poses,
        "null_pose_observations": null_poses,
        "joint_observations": int(xs.size),
        "coordinate_0": pct(xs),
        "coordinate_1": pct(ys),
        "fraction_on_1_64_grid": round(on_grid, 4) if on_grid is not None else None,
        "joint_score": pct(scores),
        "joint_score_above_1_fraction": round(float((scores > 1).mean()), 4) if scores.size else None,
        "per_joint_present_fraction": {j: round(v[0] / v[1], 4) if v[1] else None for j, v in per_joint_valid.items()},
    }


def keypoint_rcnn_cpu_timing(frames: int, size: int, threads: int) -> dict:
    import torch
    from torchvision.models.detection import keypointrcnn_resnet50_fpn

    torch.manual_seed(0)
    torch.set_num_threads(threads)
    model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=2, num_keypoints=17).eval()
    model.rpn._post_nms_top_n["testing"] = 1000  # TorchVision default, stated explicitly
    x = [torch.rand(3, size, size)]
    with torch.inference_mode():
        model(x)  # warm-up
        times = []
        for _ in range(frames):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    rss = 0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                rss = int(line.split()[1])
    except OSError:
        pass
    return {
        "model": "torchvision keypointrcnn_resnet50_fpn (random init; timing only)",
        "input": f"{size}x{size}",
        "threads": threads,
        "frames": frames,
        "mean_ms": round(float(t.mean()), 1),
        "p50_ms": round(float(np.median(t)), 1),
        "p95_ms": round(float(np.percentile(t, 95)), 1),
        "max_fps": round(1000.0 / float(t.mean()), 2),
        "realtime_at_5hz": bool(t.mean() < 200.0),
        "peak_rss_kb": rss,
        "cpu": platform.processor() or platform.machine(),
        "label": "actual_measured_result",
    }


def verdict(mapping: dict, stats: dict | None, timing: dict | None) -> dict:
    blockers = []
    if mapping["action_critical_missing"]:
        blockers.append(f"COCO-17 pose models do not output {', '.join(mapping['action_critical_missing'])}; "
                        "RetailAction take/put/touch features rely on them.")
    if stats and stats.get("fraction_on_1_64_grid", 0) and stats["fraction_on_1_64_grid"] > 0.95:
        blockers.append("Supplied coordinates are quantised to a 1/64 grid with max 0.9375: heatmap-cell coordinates "
                        "inside the 600x600 shopper crop, not full-frame pixels. Extracted poses need the same crop "
                        "and normalisation before features can match.")
    if stats and (stats.get("joint_score_above_1_fraction") or 0) > 0:
        blockers.append("Supplied joint scores exceed 1.0 (raw heatmap peaks); Keypoint R-CNN scores are unbounded "
                        "logits. Confidence channels are not comparable without re-fitting.")
    if timing and not timing["realtime_at_5hz"]:
        blockers.append(f"Keypoint R-CNN costs {timing['mean_ms']} ms per {timing['input']} frame on this CPU, above the "
                        "200 ms budget for 5 Hz analysis of one person.")
    return {
        "compatible": not blockers,
        "runtime_enabled": False,
        "blockers": blockers,
        "required_before_enabling": [
            "Choose a pose model with hand/wrist joints (e.g. a whole-body or 26-joint model) or retrain RetailAction on the COCO-17 subset plus derived neck/waist.",
            "Re-express both supplied and extracted joints in person-box-normalised coordinates; use validity masks instead of raw confidences.",
            "Run the chosen extractor on RetailAction's own videos and measure agreement with the supplied poses (PCK) before trusting transfer.",
            "Retrain with the shared schema, then evaluate on held-out staged recordings from the target camera.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--archive", type=Path, default=None)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--size", type=int, default=600)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO / "ml/results/retailaction/pose_compatibility.json")
    args = ap.parse_args()
    mapping = joint_mapping()
    stats = supplied_pose_stats(args.archive, args.samples) if args.archive and args.archive.is_file() else None
    timing = keypoint_rcnn_cpu_timing(args.frames, args.size, args.threads) if args.frames > 0 else None
    report = {"schema": "shop_pose_compatibility_v1", "joint_mapping": mapping, "supplied_pose_semantics": stats,
              "cpu_pose_timing": timing, "verdict": verdict(mapping, stats, timing)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["verdict"], indent=1))
    if timing:
        print(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
