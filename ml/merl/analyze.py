"""Error analysis for the frozen MERL TCN: curves, per-class metrics, confusion, failures, runtime policy.

Reads local copies of the Drive artefacts (no Drive access needed once copied)::

    rclone copy shopdrive:ShopliftingAI-Data/derived/features/merl_features_v1 var/research/merl_features
    rclone copy shopdrive:ShopliftingAI-Data/annotations/merl_shopping_v1/intervals.csv var/research/
    rclone copy shopdrive:ShopliftingAI-Data/splits/merl_shopping_official.json var/research/
    scripts/models/fetch_merl_tcn.sh
    .venv/bin/python -I ml/merl/analyze.py

Writes ``ml/results/merl_tcn_v1/analysis.json`` and ``report.md`` (metrics only, no media).

Selection discipline: the checkpoint and per-class thresholds are frozen (chosen on validation in
``train.py``). The runtime episode policy (min segment length, gap bridging) is chosen here on
**validation only**; test numbers are then reported once for the chosen policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.merl.dataset import CLASSES, LABEL_OFFSET, Video, step_labels  # noqa: E402
from ml.merl.metrics import average_precision, f1_at, runs, segment_detection, tiou  # noqa: E402
from shop_core.actions.merl_features import GEOM_OFFSET  # noqa: E402
from shop_core.actions.merl_runtime import EpisodePolicy, class_segments  # noqa: E402
from shop_core.temporal.tcn import Checkpoint  # noqa: E402

STEP_S = 0.2


def load_local(features: Path, intervals: Path, split_file: Path) -> dict[str, list[Video]]:
    split = json.loads(split_file.read_text())
    by_video: dict[str, list[tuple[int, int, int]]] = {}
    for r in csv.DictReader(intervals.open()):
        by_video.setdefault(r["video"], []).append((int(r["merl_class_id"]) - 1, int(r["start_frame"]), int(r["end_frame"])))
    out: dict[str, list[Video]] = {}
    for name in ("train", "val", "test"):
        out[name] = []
        for video in split["splits"][name]["videos"]:
            with np.load(features / f"{video.removesuffix('.mp4')}.npz") as z:
                feats = z["features"].astype(np.float32)
                fidx = z["frame_index"].astype(np.int64)
            out[name].append(Video(video, feats, step_labels(fidx, by_video.get(video, [])), fidx))
    return out


@torch.inference_mode()
def predict(model, ckpt: Checkpoint, videos: list[Video]) -> list[np.ndarray]:
    mean, std = ckpt.feature_mean.numpy(), ckpt.feature_std.numpy()
    out = []
    for v in videos:
        x = torch.from_numpy(((v.features - mean) / std).T.astype(np.float32))[None]
        out.append(torch.sigmoid(model(x))[0].numpy().T)
    return out


def segment_eval(probs, videos, thresholds, policy: EpisodePolicy | None, iou=0.5) -> dict:
    """Segment AP/P/R/F1 per class. ``policy=None`` reproduces the original evaluation exactly."""
    result = {}
    for k, name in enumerate(CLASSES):
        if policy is None:
            seg = segment_detection([(p[:, k], v.labels[:, k]) for p, v in zip(probs, videos)], thresholds[k], iou=iou)
            result[name] = {"ap": seg.ap, "precision": seg.precision, "recall": seg.recall, "f1": seg.f1,
                            "predicted": seg.predicted, "ground_truth": seg.ground_truth}
            continue
        preds, gts = [], {}
        for vi, (p, v) in enumerate(zip(probs, videos)):
            gts[vi] = runs(v.labels[:, k])
            for s, e in class_segments(p[:, k], thresholds[k], policy):
                preds.append((float(p[s:e + 1, k].mean()), vi, (s, e)))
        preds.sort(key=lambda t: -t[0])
        used = {vi: [False] * len(g) for vi, g in gts.items()}
        hits = []
        for _sc, vi, seg in preds:
            best, bj = 0.0, -1
            for j, g in enumerate(gts[vi]):
                if not used[vi][j] and (o := tiou(seg, g)) > best:
                    best, bj = o, j
            if best >= iou:
                used[vi][bj] = True
            hits.append(best >= iou)
        total = sum(len(g) for g in gts.values())
        tp = int(sum(hits))
        prec = tp / len(preds) if preds else 0.0
        rec = tp / total if total else 0.0
        h = np.asarray(hits, dtype=float)
        ap = float(((np.cumsum(h) / np.arange(1, len(h) + 1)) * h).sum() / total) if len(h) and total else 0.0
        result[name] = {"ap": ap, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                        "predicted": len(preds), "ground_truth": total}
    result["_map"] = float(np.mean([result[c]["ap"] for c in CLASSES]))
    return result


def frame_eval(probs, videos, thresholds) -> dict:
    p = np.concatenate(probs)
    y = np.concatenate([v.labels for v in videos])
    out = {}
    for k, name in enumerate(CLASSES):
        out[name] = {"ap": average_precision(p[:, k], y[:, k]), "chance": float(y[:, k].mean()), "support_steps": int(y[:, k].sum()),
                     "threshold": float(thresholds[k]), **f1_at(p[:, k], y[:, k], thresholds[k])}
    out["_map"] = float(np.mean([out[c]["ap"] for c in CLASSES]))
    out["_chance_map"] = float(np.mean([out[c]["chance"] for c in CLASSES]))
    return out


def confusion(probs, videos, thresholds) -> dict:
    """Steps with at most one ground-truth label: GT class (or background) vs predicted class.

    Predicted = the class with the highest score/threshold ratio among classes above threshold,
    else background. Rows are normalised to the GT class's step count.
    """
    labels = [*CLASSES, "background"]
    m = np.zeros((6, 6), dtype=np.int64)
    multi = 0
    thr = np.asarray(thresholds)
    for p, v in zip(probs, videos):
        n = v.labels.sum(axis=1)
        multi += int((n > 1).sum())
        ratio = p / thr
        pred = np.where(ratio.max(axis=1) >= 1.0, ratio.argmax(axis=1), 5)
        gt = np.where(n == 1, v.labels.argmax(axis=1), 5)
        for g, q in zip(gt[n <= 1], pred[n <= 1]):
            m[g, q] += 1
    return {"labels": labels, "counts": m.tolist(),
            "row_normalised": np.round(m / np.maximum(1, m.sum(axis=1, keepdims=True)), 4).tolist(),
            "excluded_multi_label_steps": multi}


def threshold_sweep(probs, videos) -> dict:
    p = np.concatenate(probs)
    y = np.concatenate([v.labels for v in videos])
    grid = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]
    return {name: [{"threshold": t, **{k2: round(v2, 4) for k2, v2 in f1_at(p[:, k], y[:, k], t).items()}} for t in grid]
            for k, name in enumerate(CLASSES)}


def reliability(probs, videos, bins=10) -> dict:
    p = np.concatenate(probs)
    y = np.concatenate([v.labels for v in videos])
    edges = np.linspace(0, 1, bins + 1)
    out = {}
    for k, name in enumerate(CLASSES):
        rows = []
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (p[:, k] >= lo) & (p[:, k] < hi if hi < 1 else p[:, k] <= hi)
            if sel.sum() == 0:
                continue
            conf, acc = float(p[sel, k].mean()), float(y[sel, k].mean())
            ece += sel.mean() * abs(conf - acc)
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "mean_score": round(conf, 4), "positive_rate": round(acc, 4), "steps": int(sel.sum())})
        out[name] = {"bins": rows, "ece": round(float(ece), 4)}
    return out


def visibility_breakdown(probs, videos) -> dict:
    p = np.concatenate(probs)
    y = np.concatenate([v.labels for v in videos])
    present = np.concatenate([v.features[:, GEOM_OFFSET] > 0.5 for v in videos])
    out = {"person_present_fraction": round(float(present.mean()), 4)}
    for tag, sel in (("person_visible", present), ("person_not_detected", ~present)):
        out[tag] = {name: round(average_precision(p[sel, k], y[sel, k]), 4) if y[sel, k].sum() else None for k, name in enumerate(CLASSES)}
        out[tag]["_steps"] = int(sel.sum())
    return out


def failure_examples(probs, videos, thresholds, policy: EpisodePolicy, n=6) -> dict:
    """Highest-scoring false positives and longest missed ground-truth segments (test split)."""
    fps, misses = [], []
    for p, v in zip(probs, videos):
        for k, name in enumerate(CLASSES):
            gts = runs(v.labels[:, k])
            preds = class_segments(p[:, k], thresholds[k], policy)
            for s, e in preds:
                if max((tiou((s, e), g) for g in gts), default=0.0) < 0.1:
                    other = [CLASSES[j] for j in range(len(CLASSES)) if v.labels[s:e + 1, j].any()]
                    fps.append({"video": v.name, "class": name, "start_s": round(s * STEP_S, 1), "end_s": round((e + 1) * STEP_S, 1),
                                "mean_score": round(float(p[s:e + 1, k].mean()), 3), "overlapping_gt": other,
                                "person_visible": round(float((v.features[s:e + 1, GEOM_OFFSET] > 0.5).mean()), 2)})
            for s, e in gts:
                if max((tiou((s, e), q) for q in preds), default=0.0) < 0.1:
                    misses.append({"video": v.name, "class": name, "start_s": round(s * STEP_S, 1), "end_s": round((e + 1) * STEP_S, 1),
                                   "max_score": round(float(p[s:e + 1, k].max()), 3), "threshold": round(float(thresholds[k]), 3),
                                   "person_visible": round(float((v.features[s:e + 1, GEOM_OFFSET] > 0.5).mean()), 2)})
    fps.sort(key=lambda r: -r["mean_score"])
    misses.sort(key=lambda r: -(r["end_s"] - r["start_s"]))
    miss_vis = [m["person_visible"] for m in misses]
    return {"false_positives": fps[:n], "missed": misses[:n], "false_positive_count": len(fps), "missed_count": len(misses),
            "missed_mean_person_visible": round(float(np.mean(miss_vis)), 3) if miss_vis else None,
            "false_positive_with_other_gt_fraction": round(float(np.mean([bool(f["overlapping_gt"]) for f in fps])), 3) if fps else None}


def per_video(probs, videos) -> list[dict]:
    rows = []
    for p, v in zip(probs, videos):
        aps = [average_precision(p[:, k], v.labels[:, k]) for k in range(len(CLASSES))]
        valid = [a for a in aps if a == a]
        rows.append({"video": v.name, "map": round(float(np.mean(valid)), 4) if valid else None, "steps": len(p),
                     "person_visible": round(float((v.features[:, GEOM_OFFSET] > 0.5).mean()), 3)})
    rows.sort(key=lambda r: r["map"] if r["map"] is not None else 9)
    return rows


def class_stats(videos) -> dict:
    out = {}
    for k, name in enumerate(CLASSES):
        lens = [e - s + 1 for v in videos for s, e in runs(v.labels[:, k])]
        out[name] = {"segments": len(lens), "median_s": round(float(np.median(lens)) * STEP_S, 2) if lens else None,
                     "p10_s": round(float(np.percentile(lens, 10)) * STEP_S, 2) if lens else None, "steps": int(sum(lens))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--features", type=Path, default=REPO / "var/research/merl_features")
    ap.add_argument("--intervals", type=Path, default=REPO / "var/research/intervals.csv")
    ap.add_argument("--split", type=Path, default=REPO / "var/research/merl_shopping_official.json")
    ap.add_argument("--checkpoint", type=Path, default=REPO / "var/models/merl_tcn_v1/model.pt")
    ap.add_argument("--train-log", type=Path, default=REPO / "var/models/merl_tcn_v1/train_log.json")
    ap.add_argument("--out", type=Path, default=REPO / "ml/results/merl_tcn_v1")
    args = ap.parse_args()
    torch.set_num_threads(4)

    ckpt = Checkpoint.load(args.checkpoint)
    model = ckpt.build()
    thr = list(ckpt.thresholds)
    data = load_local(args.features, args.intervals, args.split)
    val, test = data["val"], data["test"]
    pv, pt = predict(model, ckpt, val), predict(model, ckpt, test)

    # Runtime policy grid, chosen on validation segment mAP only.
    grid = []
    for min_len in (1, 2, 3):
        for gap in (0, 1, 2):
            pol = EpisodePolicy(min_segment_steps=min_len, merge_gap_steps=gap)
            grid.append({"min_segment_steps": min_len, "merge_gap_steps": gap, "val_segment_map": round(segment_eval(pv, val, thr, pol)["_map"], 4)})
    best = max(grid, key=lambda g: g["val_segment_map"])
    chosen = EpisodePolicy(min_segment_steps=best["min_segment_steps"], merge_gap_steps=best["merge_gap_steps"])

    log = json.loads(args.train_log.read_text()) if args.train_log.is_file() else []
    analysis = {
        "run": ckpt.metadata.get("run"),
        "checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": {k: len(v) for k, v in data.items()},
        "steps": {k: int(sum(len(x.features) for x in v)) for k, v in data.items()},
        "training_curve": [{"epoch": r["epoch"], "train_loss": round(r["train_loss"], 4), "val_map": round(r["val_map"], 4),
                            "val_ap": dict(zip(CLASSES, [round(a, 4) for a in r["val_ap"]]))} for r in log],
        "best_epoch": ckpt.metadata.get("best_epoch"),
        "class_stats": {"train": class_stats(data["train"]), "test": class_stats(test)},
        "validation": {"frame": frame_eval(pv, val, thr), "segment_eval_policy": segment_eval(pv, val, thr, None)},
        "test": {"frame": frame_eval(pt, test, thr), "segment_eval_policy": segment_eval(pt, test, thr, None),
                 "segment_runtime_policy": segment_eval(pt, test, thr, chosen)},
        "runtime_policy_selection": {"grid": grid, "chosen": best, "selected_on": "validation"},
        "confusion_test": confusion(pt, test, thr),
        "threshold_sweep_validation": threshold_sweep(pv, val),
        "reliability_test": reliability(pt, test),
        "visibility_test": visibility_breakdown(pt, test),
        "per_video_test": per_video(pt, test),
        "failures_test": failure_examples(pt, test, thr, chosen),
        "notes": [
            "Frame metrics: per 5 Hz step. Segment metrics: temporal IoU >= 0.5.",
            "Thresholds and runtime policy were selected on validation; test is reported, not tuned.",
            "Scores are uncalibrated observable-action scores, never theft or intent probabilities.",
            "MERL licence: research and educational use only; the checkpoint is development_only.",
        ],
        "label": "actual_measured_result",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=1, default=float, allow_nan=True).replace("NaN", "null") + "\n")
    write_report(analysis, args.out / "report.md")
    print((args.out / "report.md").read_text())


def write_report(a: dict, path: Path) -> None:
    f, sv, st, sr = a["test"]["frame"], a["validation"]["segment_eval_policy"], a["test"]["segment_eval_policy"], a["test"]["segment_runtime_policy"]
    vf = a["validation"]["frame"]
    L = [f"# MERL TCN error analysis: `{a['run']}`", "",
         f"Checkpoint SHA-256 `{a['checkpoint_sha256'][:16]}…`, best epoch {a['best_epoch']}. Split (videos): "
         f"train {a['split']['train']}, val {a['split']['val']}, test {a['split']['test']}.", "",
         "## Per-class results", "",
         "| Class | Val frame AP | Test frame AP | Chance | Test P | Test R | Test F1 | Test seg AP (eval) | Test seg AP (runtime) |",
         "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in CLASSES:
        L.append(f"| {c} | {vf[c]['ap']:.3f} | {f[c]['ap']:.3f} | {f[c]['chance']:.3f} | {f[c]['precision']:.3f} | {f[c]['recall']:.3f} | "
                 f"{f[c]['f1']:.3f} | {st[c]['ap']:.3f} | {sr[c]['ap']:.3f} |")
    L += ["", f"Frame mAP: val {vf['_map']:.3f}, test {f['_map']:.3f} (chance {f['_chance_map']:.3f}). "
          f"Segment mAP@0.5: val {sv['_map']:.3f}, test {st['_map']:.3f}; runtime episode policy on test {sr['_map']:.3f}.", "",
          f"Runtime policy chosen on validation: {a['runtime_policy_selection']['chosen']}.", "",
          "## Confusion (test, single-label and background steps, row-normalised)", "",
          "| GT \\ Pred | " + " | ".join(a["confusion_test"]["labels"]) + " |", "| --- |" + " ---: |" * 6]
    for lab, row in zip(a["confusion_test"]["labels"], a["confusion_test"]["row_normalised"]):
        L.append(f"| {lab} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
    vis = a["visibility_test"]
    L += ["", "## Visibility", "", f"Person detected in {vis['person_present_fraction']:.0%} of test steps.", "",
          "| Class | AP, person visible | AP, person not detected |", "| --- | ---: | ---: |"]
    for c in CLASSES:
        pv, pn = vis["person_visible"][c], vis["person_not_detected"][c]
        L.append(f"| {c} | {pv if pv is not None else '—'} | {pn if pn is not None else '—'} |")
    fl = a["failures_test"]
    L += ["", "## Failure examples (test)", "",
          f"{fl['false_positive_count']} false-positive segments (tIoU < 0.1 with any same-class GT); "
          f"{fl['false_positive_with_other_gt_fraction']:.0%} of them overlap a *different* annotated action. "
          f"{fl['missed_count']} missed GT segments; mean person visibility inside misses {fl['missed_mean_person_visible']}.", "",
          "| Type | Video | Class | Time (s) | Score | Person visible | Note |", "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for r in fl["false_positives"]:
        L.append(f"| FP | {r['video']} | {r['class']} | {r['start_s']}–{r['end_s']} | {r['mean_score']} | {r['person_visible']} | GT here: {', '.join(r['overlapping_gt']) or 'none'} |")
    for r in fl["missed"]:
        L.append(f"| Miss | {r['video']} | {r['class']} | {r['start_s']}–{r['end_s']} | max {r['max_score']} (thr {r['threshold']}) | {r['person_visible']} | |")
    L += ["", "## Calibration (test, expected calibration error)", "",
          ", ".join(f"{c} {a['reliability_test'][c]['ece']:.3f}" for c in CLASSES) + ". Scores are not probabilities.", "",
          "## Weakest / strongest test videos (frame mAP)", ""]
    pvid = [r for r in a["per_video_test"] if r["map"] is not None]
    for r in pvid[:3] + pvid[-3:]:
        L.append(f"- {r['video']}: mAP {r['map']:.3f}, person visible {r['person_visible']:.0%}")
    L += ["", *[f"- {n}" for n in a["notes"]]]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
