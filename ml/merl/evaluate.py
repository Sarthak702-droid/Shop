"""Evaluate a trained MERL checkpoint on the held-out test split (subjects 27-41) and save to Drive.

Usage (repository root)::

    .venv/bin/python ml/merl/evaluate.py --run tcn-20261118T...Z
    .venv/bin/python ml/merl/evaluate.py --run latest

Reports, per class: frame AP, chance AP (= prevalence), precision/recall/F1 at the validation
threshold, and segment AP / P / R / F1 at temporal IoU 0.5. Writes
``evaluation/merl_tcn_v1/<run>/test_metrics.json`` and ``report.md`` to Drive.
Thresholds come from validation; nothing is tuned on test.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.merl import drive  # noqa: E402
from ml.merl.dataset import CLASSES, load  # noqa: E402
from ml.merl.metrics import average_precision, f1_at, segment_detection  # noqa: E402
from ml.merl.train import DRIVE_CKPT, predict  # noqa: E402
from shop_core.temporal.tcn import Checkpoint  # noqa: E402

DRIVE_EVAL = "evaluation/merl_tcn_v1"


def latest_run() -> str:
    out = drive._run("lsf", "--dirs-only", drive.path(DRIVE_CKPT))  # noqa: SLF001
    runs = sorted(line.rstrip("/") for line in out.splitlines() if line.strip())
    if not runs:
        raise SystemExit("no checkpoints on Drive yet; run ml/merl/train.py")
    return runs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="latest")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    run = latest_run() if args.run == "latest" else args.run
    torch.set_num_threads(4)

    with tempfile.TemporaryDirectory(prefix="shop-merl-eval-") as tmp:
        ckpt = Checkpoint.load(drive.download(f"{DRIVE_CKPT}/{run}/model.pt", Path(tmp) / "model.pt"))
    model = ckpt.build()
    data, info = load(("test",))
    test = data["test"]
    if info["missing"]:
        raise SystemExit(f"features missing for {len(info['missing'])} test videos")
    probs = predict(model, test, ckpt.feature_mean.numpy(), ckpt.feature_std.numpy())
    p = np.concatenate(probs)
    y = np.concatenate([v.labels for v in test])

    per_class = {}
    for k, name in enumerate(CLASSES):
        thr = ckpt.thresholds[k]
        seg = segment_detection([(pr[:, k], v.labels[:, k]) for pr, v in zip(probs, test)], thr, iou=args.iou)
        per_class[name] = {
            "frame_ap": average_precision(p[:, k], y[:, k]),
            "chance_ap": float(y[:, k].mean()),
            "threshold_from_val": thr,
            **{f"frame_{m}": v for m, v in f1_at(p[:, k], y[:, k], thr).items()},
            "segment_ap": seg.ap, "segment_precision": seg.precision, "segment_recall": seg.recall,
            "segment_f1": seg.f1, "segments_predicted": seg.predicted, "segments_ground_truth": seg.ground_truth,
        }
    summary = {
        "run": run,
        "test_videos": len(test),
        "test_steps": int(len(p)),
        "frame_map": float(np.mean([c["frame_ap"] for c in per_class.values()])),
        "chance_map": float(np.mean([c["chance_ap"] for c in per_class.values()])),
        "segment_map_iou": args.iou,
        "segment_map": float(np.mean([c["segment_ap"] for c in per_class.values()])),
        "per_class": per_class,
        "checkpoint": ckpt.metadata,
        "notes": [
            "Labels at 5 Hz step centres from MERL frame intervals; segments are runs of positive steps.",
            "Scores are uncalibrated observable-action scores, not theft probabilities.",
            "MERL licence: research and educational use only; checkpoint is development_only.",
        ],
    }
    lines = [
        f"# MERL Shopping test results: `{run}`", "",
        f"{len(test)} test videos (subjects 27–41), {len(p):,} steps at 5 Hz. Thresholds were chosen on validation.", "",
        f"**Frame mAP {summary['frame_map']:.3f}** (chance {summary['chance_map']:.3f}) · "
        f"**Segment mAP@{args.iou} {summary['segment_map']:.3f}**", "",
        "| Class | Frame AP | Chance | Frame F1 | Segment AP | Segment P | Segment R | Segment F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, c in per_class.items():
        lines.append(f"| {name} | {c['frame_ap']:.3f} | {c['chance_ap']:.3f} | {c['frame_f1']:.3f} | {c['segment_ap']:.3f} | "
                     f"{c['segment_precision']:.3f} | {c['segment_recall']:.3f} | {c['segment_f1']:.3f} |")
    lines += ["", "Scores describe observable shelf interactions only. They are not theft, intent or guilt scores."]
    with tempfile.TemporaryDirectory(prefix="shop-merl-eval-") as tmp:
        t = Path(tmp)
        (t / "test_metrics.json").write_text(json.dumps(summary, indent=2, default=float))
        (t / "report.md").write_text("\n".join(lines) + "\n")
        drive.upload_dir(t, f"{DRIVE_EVAL}/{run}")
    print("\n".join(lines))
    print(f"\nSaved to {drive.path(f'{DRIVE_EVAL}/{run}')}")


if __name__ == "__main__":
    main()
