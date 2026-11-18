"""Cross-domain test: Shop's MERL shelf-action model on RetailAction clips from other cameras.

RetailAction (Standard AI) clips come from ceiling cameras in real stores, two views per clip
(rank0 / rank1), as 600x600 shopper-centred crops with 32 non-uniformly sampled frames over ~8-11 s.
The MERL TCN never saw this data, so this measures transfer, not fit.

Stages::

    # 1. Stream the verified original train.tar; keep a stratified sample (no arbitrary extraction)
    .venv/bin/python -I ml/retailaction/merl_transfer.py extract --archive /path/train.tar --per-label 30
    # 2. Run the runtime model under two input conditions and score it
    .venv/bin/python -I ml/retailaction/merl_transfer.py evaluate

Conditions:

* ``as_uploaded``: the mp4 exactly as a user would upload it to Shop (declared 6 fps, 32 frames).
* ``time_corrected``: frames held for their real timestamps and re-timed to 30 fps, so 5 Hz steps
  match real time (the motion signal is then concentrated at frame changes).

Label mapping (clip level): ``take`` / ``put`` / ``touch`` -> "shelf interaction present";
an explicit empty action list -> "no labelled interaction". RetailAction intervals are fractions of
the 32-frame index (multiples of 1/32). MERL classes that indicate an interaction: reach_to_shelf,
retract_from_shelf, hand_in_shelf. inspect_* are reported but not counted as interaction evidence.

Writes ``ml/results/retailaction/merl_transfer.json`` and ``merl_transfer.md`` (metrics only).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.merl.metrics import average_precision  # noqa: E402
from shop_core.actions import merl_features as mf  # noqa: E402
from shop_core.actions.merl_runtime import EpisodePolicy, ShelfActionModel, build_episodes  # noqa: E402

PROBE = REPO / "var/research/retailaction_probe"
OUT = REPO / "ml/results/retailaction"
MEMBER = re.compile(r"^train/(\d{6})/(rank[01]_video\.mp4|metadata\.json)$")
INTERACTION = ("reach_to_shelf", "retract_from_shelf", "hand_in_shelf")
GROUPS = ("take", "put", "touch", "none")


def _ts(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def extract(archive: Path, per_label: int, seed: int) -> None:
    """One sequential pass. Media bytes of a sample are buffered until its metadata decides the group."""
    rng = random.Random(seed)
    keep_prob = {"take": 0.02, "put": 1.0, "touch": 1.0, "none": 0.25}  # take is ~250x more common than touch
    buffers: dict[str, dict[str, bytes]] = defaultdict(dict)
    chosen: dict[str, list[dict]] = {g: [] for g in GROUPS}
    clips = PROBE / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    scanned = 0
    decided: set[str] = set()
    pending: dict[str, dict] = {}  # sid -> accepted sample record awaiting its media (member order varies)

    def finalize(sid: str) -> None:
        rec, media = pending.get(sid), buffers.get(sid, {})
        if rec is None or set(media) != {"rank0_video.mp4", "rank1_video.mp4"}:
            return
        if len(chosen[rec["group"]]) < per_label:
            for rank in ("rank0", "rank1"):
                (clips / f"{sid}_{rank}.mp4").write_bytes(media[f"{rank}_video.mp4"])
            chosen[rec["group"]].append(rec)
        pending.pop(sid)
        buffers.pop(sid, None)

    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            m = MEMBER.match(member.name)
            if not m or not member.isfile() or member.size > 20 * 1024 * 1024:
                continue
            sid, name = m.groups()
            data = tar.extractfile(member).read()
            if name != "metadata.json":
                if sid in pending or sid not in decided:
                    buffers[sid][name] = data
                finalize(sid)
                continue
            scanned += 1
            decided.add(sid)
            content = json.loads(data)["content"]
            actions = (content.get("labels") or {}).get("action")
            if actions is None:
                buffers.pop(sid, None)
                continue  # missing container: quarantined by the RetailAction loader too
            labels = {a["label"] for a in actions}
            group = "none" if not labels else (next(iter(labels)) if len(labels) == 1 else None)
            cams = content["action_cam"]
            ok_ts = all(cams[r].get("frame_timestamps") and len(cams[r]["frame_timestamps"]) == 32 for r in ("rank0", "rank1"))
            if group is None or not ok_ts or len(chosen[group]) >= per_label or rng.random() > keep_prob[group]:
                buffers.pop(sid, None)
                continue
            pending[sid] = {
                "sample_id": sid, "group": group,
                "intervals": [[a["start"], a["end"], a["label"]] for a in actions],
                "timestamps": {r: [_ts(t) - _ts(cams[r]["frame_timestamps"][0]) for t in cams[r]["frame_timestamps"]] for r in ("rank0", "rank1")},
                "duration_s": _ts(content["segment_info"]["sampled_at_end"]) - _ts(content["segment_info"]["sampled_at_start"]),
            }
            finalize(sid)
            if all(len(v) >= per_label for v in chosen.values()):
                break
    manifest = {"source": f"{archive.name} (publisher SHA-256 verified)", "split": "train", "scanned_samples": scanned,
                "seed": seed, "per_label": per_label, "samples": [s for g in GROUPS for s in chosen[g]]}
    (PROBE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print({g: len(v) for g, v in chosen.items()}, "scanned", scanned)


def retime(src: Path, timestamps: list[float], dst: Path) -> None:
    """Hold each decoded frame for its real duration at 30 fps (MJPG: held frames decode identically)."""
    cap = cv2.VideoCapture(str(src))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
    ts = list(timestamps[: len(frames)])
    ends = ts[1:] + [ts[-1] + (ts[-1] - ts[-2] if len(ts) > 1 else 0.3)]
    for f, t0, t1 in zip(frames, ts, ends):
        for _ in range(max(1, int(round((t1 - t0) * 30)))):
            out.write(f)
    out.release()


def frame_labels(intervals: list, n_frames: int) -> np.ndarray:
    y = np.zeros(n_frames, dtype=bool)
    for s, e, _ in intervals:
        y[int(round(s * 32)): max(int(round(s * 32)) + 1, int(round(e * 32)))] = True
    return y[:n_frames]


def evaluate() -> None:
    from shop_core.config import load_settings
    from shop_core.detection.person import PersonDetector

    manifest = json.loads((PROBE / "manifest.json").read_text())
    settings = load_settings(REPO / "configs/shop.yaml")
    model = ShelfActionModel.load(settings)
    ex = mf.Extractor.from_detector(PersonDetector.load(settings))
    policy = EpisodePolicy()
    cls = list(model.classes)
    inter_idx = [cls.index(c) for c in INTERACTION]
    thr = model.thresholds
    mean, std = model.mean, model.std
    rows = []
    with tempfile.TemporaryDirectory(prefix="shop-ra-") as tmp:
        for s in manifest["samples"]:
            for rank in ("rank0", "rank1"):
                src = PROBE / "clips" / f"{s['sample_id']}_{rank}.mp4"
                retimed = Path(tmp) / f"{s['sample_id']}_{rank}.avi"
                retime(src, s["timestamps"][rank], retimed)
                for cond, path in (("as_uploaded", src), ("time_corrected", retimed)):
                    feats = mf.extract_video(ex, path)
                    probs = model.predict(feats.features)
                    eps = build_episodes(probs, thr, model.classes, feats.pts_ms, feats.person_present, feats.person_score, policy)
                    # map steps -> source frame index (as_uploaded: 1 frame/step; time_corrected: by real time)
                    if cond == "as_uploaded":
                        step_frame = np.minimum(np.arange(len(probs)), 31)
                    else:
                        t = feats.pts_ms / 1000.0
                        step_frame = np.searchsorted(np.asarray(s["timestamps"][rank]), t, side="right") - 1
                    y = frame_labels(s["intervals"], 32)[np.clip(step_frame, 0, 31)]
                    ratio = probs[:, inter_idx] / thr[inter_idx]
                    inter = ratio.max(axis=1)
                    above = [e for e in eps if any(a.label in INTERACTION for a in e.actions)]
                    peak = int(inter.argmax())
                    z = np.abs((feats.features.astype(np.float32) - mean) / std)
                    rows.append({
                        "sample_id": s["sample_id"], "group": s["group"], "view": rank, "condition": cond,
                        "steps": len(probs), "fps": round(feats.fps, 2), "duration_s": round(s["duration_s"], 2),
                        "interaction_score": round(float(inter.max()), 4),
                        "episode_with_interaction": bool(above),
                        "episodes": len(eps),
                        "max_by_class": {c: round(float(probs[:, k].max()), 4) for k, c in enumerate(cls)},
                        "above_thr_steps": {c: int((probs[:, k] >= thr[k]).sum()) for k, c in enumerate(cls)},
                        "peak_in_label": bool(y[peak]) if y.any() else None,
                        "any_above_inside_label": bool(((ratio.max(axis=1) >= 1) & y).any()) if y.any() else None,
                        "person_visible": round(float(feats.person_present.mean()), 3),
                        "feature_shift_z": {
                            "roi": round(float(z[:, :672].mean()), 3), "global": round(float(z[:, 672:1152].mean()), 3),
                            "geom": round(float(z[:, 1152:1163].mean()), 3), "motion": round(float(z[:, 1163:].mean()), 3),
                        },
                        "motion_energy": round(float(feats.features[:, 1163 + 32].astype(np.float32).mean()), 5),
                    })
            print(s["sample_id"], s["group"], [(r["condition"][:4], r["view"], r["interaction_score"]) for r in rows[-4:]], flush=True)
    summarize(rows, manifest)


def merl_reference() -> dict:
    """Same diagnostics on MERL test features, so shifts and scores have a baseline."""
    ref = {}
    split = json.loads((REPO / "var/research/merl_shopping_official.json").read_text())["splits"]["test"]["videos"]
    zs, mot, vis = [], [], []
    from shop_core.temporal.tcn import Checkpoint
    ck = Checkpoint.load(REPO / "var/models/merl_tcn_v1/model.pt")
    mean, std = ck.feature_mean.numpy(), ck.feature_std.numpy()
    for v in split[:10]:
        with np.load(REPO / f"var/research/merl_features/{v.removesuffix('.mp4')}.npz") as z:
            f = z["features"].astype(np.float32)
        a = np.abs((f - mean) / std)
        zs.append([a[:, :672].mean(), a[:, 672:1152].mean(), a[:, 1152:1163].mean(), a[:, 1163:].mean()])
        mot.append(f[:, 1163 + 32].mean())
        vis.append((f[:, 1152] > 0.5).mean())
    zs = np.asarray(zs).mean(axis=0)
    ref["feature_shift_z"] = dict(zip(("roi", "global", "geom", "motion"), [round(float(x), 3) for x in zs]))
    ref["motion_energy"] = round(float(np.mean(mot)), 5)
    ref["person_visible"] = round(float(np.mean(vis)), 3)
    return ref


def summarize(rows: list[dict], manifest: dict) -> None:
    out = {"schema": "shop_merl_transfer_v1", "source": manifest["source"], "samples": len(manifest["samples"]),
           "groups": {g: sum(1 for s in manifest["samples"] if s["group"] == g) for g in GROUPS},
           "merl_reference": merl_reference(), "conditions": {}}
    for cond in ("as_uploaded", "time_corrected"):
        R = [r for r in rows if r["condition"] == cond]
        pos = [r for r in R if r["group"] != "none"]
        neg = [r for r in R if r["group"] == "none"]
        scores = np.array([r["interaction_score"] for r in R])
        labels = np.array([r["group"] != "none" for r in R])
        c = {
            "clips": len(R),
            "clip_ap_interaction": round(average_precision(scores, labels), 4),
            "chance_ap": round(float(labels.mean()), 4),
            "detection_rate": {g: round(float(np.mean([r["episode_with_interaction"] for r in R if r["group"] == g])), 4) for g in GROUPS},
            "mean_interaction_score": {g: round(float(np.mean([r["interaction_score"] for r in R if r["group"] == g])), 4) for g in GROUPS},
            "by_view": {v: {"clip_ap": round(average_precision(np.array([r["interaction_score"] for r in R if r["view"] == v]),
                                                            np.array([r["group"] != "none" for r in R if r["view"] == v])), 4),
                            "detection_rate_positive": round(float(np.mean([r["episode_with_interaction"] for r in pos if r["view"] == v])), 4),
                            "false_episode_rate_negative": round(float(np.mean([r["episode_with_interaction"] for r in neg if r["view"] == v])), 4)}
                        for v in ("rank0", "rank1")},
            "view_agreement": round(float(np.mean([a["episode_with_interaction"] == b["episode_with_interaction"]
                                                   for a, b in zip(R[0::2], R[1::2])])), 4),
            "peak_inside_label": round(float(np.mean([r["peak_in_label"] for r in pos if r["peak_in_label"] is not None])), 4),
            "label_fraction_of_clip": round(float(np.mean([sum(e - s for s, e, _ in next(m for m in manifest["samples"] if m["sample_id"] == r["sample_id"])["intervals"]) for r in pos])), 4),
            "class_fire_rate": {k: round(float(np.mean([r["above_thr_steps"][k] > 0 for r in R])), 4) for k in R[0]["above_thr_steps"]},
            "person_visible": round(float(np.mean([r["person_visible"] for r in R])), 4),
            "feature_shift_z": {k: round(float(np.mean([r["feature_shift_z"][k] for r in R])), 3) for k in ("roi", "global", "geom", "motion")},
            "motion_energy": round(float(np.mean([r["motion_energy"] for r in R])), 5),
            "steps": round(float(np.mean([r["steps"] for r in R])), 1),
        }
        out["conditions"][cond] = c
    out["rows"] = rows
    out["label"] = "actual_measured_result"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "merl_transfer.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--archive", type=Path, required=True)
    e.add_argument("--per-label", type=int, default=30)
    e.add_argument("--seed", type=int, default=7)
    sub.add_parser("evaluate")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.archive, args.per_label, args.seed)
    else:
        evaluate()


if __name__ == "__main__":
    main()
