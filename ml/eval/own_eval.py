"""Evaluate the runtime shelf-action path on your own staged, consented recordings.

Does public-dataset (MERL) performance transfer to *your* camera? This answers it on a small,
held-out set recorded with permission. Steps (details in ``docs/own-evaluation.md``)::

    # 1. VIA template with the five MERL action labels, for annotating each recording
    .venv/bin/python -I ml/eval/own_eval.py template --out var/own_eval/via_template.json
    # 2. Fill a manifest (see ml/eval/own_eval_manifest.example.json), then evaluate
    .venv/bin/python -I ml/eval/own_eval.py run --manifest var/own_eval/manifest.json

Rules enforced before any video is scored:

* every recording has a ``session`` and ``people`` list and a VIA project with temporal segments;
* ``split`` must be ``eval``; no eval session or person may also appear in ``training_sessions`` /
  ``training_people`` (grouped split; evaluation recordings are never used for training);
* scenario tags (``reach``, ``take_and_return``, ``inspect``, ``normal_pass``, ``occlusion``,
  ``low_light`` ...) are free text and are reported separately.

Writes ``ml/results/own_eval/summary.json`` (metrics only, no media), shown on the Research page.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.merl.annotation import MERL_CLASSES, _project, project_intervals  # noqa: E402
from ml.merl.metrics import average_precision, runs, tiou  # noqa: E402
from shop_core.actions import merl_features as mf  # noqa: E402
from shop_core.actions.merl_runtime import EpisodePolicy, ShelfActionModel, build_episodes, class_segments, input_warnings  # noqa: E402

OUT = REPO / "ml/results/own_eval/summary.json"


def validate(manifest: dict) -> list[dict]:
    recs = manifest.get("recordings") or []
    if not recs:
        raise SystemExit("manifest has no recordings")
    train_sessions = set(manifest.get("training_sessions") or [])
    train_people = set(manifest.get("training_people") or [])
    errors = []
    for r in recs:
        for key in ("video", "session", "people", "via_project", "split"):
            if not r.get(key):
                errors.append(f"{r.get('video', '?')}: missing {key}")
        if r.get("split") != "eval":
            errors.append(f"{r.get('video')}: split must be 'eval' (training recordings do not belong in this manifest)")
        if r.get("session") in train_sessions:
            errors.append(f"{r.get('video')}: session {r['session']} is also a training session")
        if set(r.get("people") or []) & train_people:
            errors.append(f"{r.get('video')}: person also appears in training data")
    if not manifest.get("consent_record"):
        errors.append("manifest.consent_record is required (path or reference to the written permission)")
    if errors:
        raise SystemExit("manifest rejected:\n  " + "\n  ".join(errors))
    return recs


def labels_for(project: dict, frame_index: np.ndarray, fps: float) -> np.ndarray:
    y = np.zeros((len(frame_index), len(MERL_CLASSES)), dtype=np.float32)
    centre_s = (frame_index + max(1, int(round(fps / 5))) / 2) / fps
    for row in project_intervals(project):
        if row["action"] in MERL_CLASSES:
            k = MERL_CLASSES.index(row["action"])
            y[(centre_s >= row["start_s"]) & (centre_s <= row["end_s"]), k] = 1
    return y


def seg_counts(p: np.ndarray, y: np.ndarray, thr: float, policy: EpisodePolicy, iou: float = 0.5) -> tuple[int, int, int]:
    preds = class_segments(p, thr, policy)
    gts = runs(y)
    used = [False] * len(gts)
    tp = 0
    for s in preds:
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if not used[j] and (o := tiou(s, g)) > best:
                best, bj = o, j
        if best >= iou:
            used[bj] = True
            tp += 1
    return tp, len(preds), len(gts)


def run(manifest_path: Path) -> None:
    from shop_core.config import load_settings
    from shop_core.detection.person import PersonDetector

    manifest = json.loads(manifest_path.read_text())
    recs = validate(manifest)
    base = manifest_path.parent
    settings = load_settings(REPO / "configs/shop.yaml")
    model = ShelfActionModel.load(settings)
    ex = mf.Extractor.from_detector(PersonDetector.load(settings))
    policy = EpisodePolicy()
    per_rec, pooled_p, pooled_y = [], [], []
    by_tag: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for r in recs:
        feats = mf.extract_video(ex, base / r["video"])
        probs = model.predict(feats.features)
        y = labels_for(json.loads((base / r["via_project"]).read_text()), feats.frame_index, feats.fps)
        eps = build_episodes(probs, model.thresholds, model.classes, feats.pts_ms, feats.person_present, feats.person_score, policy)
        pooled_p.append(probs)
        pooled_y.append(y)
        for tag in r.get("scenarios") or ["untagged"]:
            by_tag[tag].append((probs, y))
        no_action = y.sum() == 0
        per_rec.append({
            "video": r["video"], "session": r["session"], "scenarios": r.get("scenarios") or [],
            "steps": len(probs), "annotated_steps": int((y.sum(axis=1) > 0).sum()), "episodes": len(eps),
            "false_episodes_on_no_action_recording": len(eps) if no_action else None,
            "person_visible_fraction": round(float(feats.person_present.mean()), 3),
            "warnings": [w["code"] for w in input_warnings(feats, model.receptive_field_steps)],
        })
        print(json.dumps(per_rec[-1]), flush=True)

    def metrics(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        p = np.concatenate([a for a, _ in pairs])
        y = np.concatenate([b for _, b in pairs])
        out = {}
        for k, c in enumerate(MERL_CLASSES):
            tp = n_pred = n_gt = 0
            for a, b in pairs:
                t, q, g = seg_counts(a[:, k], b[:, k], float(model.thresholds[k]), policy)
                tp, n_pred, n_gt = tp + t, n_pred + q, n_gt + g
            out[c] = {"frame_ap": None if y[:, k].sum() == 0 else round(average_precision(p[:, k], y[:, k]), 4),
                      "support_steps": int(y[:, k].sum()), "segment_precision": round(tp / n_pred, 4) if n_pred else None,
                      "segment_recall": round(tp / n_gt, 4) if n_gt else None, "segments_predicted": n_pred, "segments_annotated": n_gt}
        return out

    summary = {
        "schema": "shop_own_eval_v1", "available": True,
        "updated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": f"{model.model_id}:{model.run}", "policy": policy.__dict__,
        "recordings": len(recs), "sessions": len({r["session"] for r in recs}),
        "people": len({p for r in recs for p in r["people"]}),
        "overall": metrics(list(zip(pooled_p, pooled_y))),
        "by_scenario": {tag: metrics(pairs) for tag, pairs in by_tag.items()},
        "per_recording": per_rec,
        "consent_record": manifest.get("consent_record"),
        "note": "Small staged set; report counts with every number. Never merged into training data.",
        "label": "actual_measured_result",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "template":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(_project("Shop own evaluation (MERL labels)", None, MERL_CLASSES, objects=False), indent=1))
        print(f"wrote {args.out}; open it in VIA, add your video, mark each action as a temporal segment")
    else:
        run(args.manifest)


if __name__ == "__main__":
    main()
