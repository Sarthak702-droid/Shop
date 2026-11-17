"""Train the causal action TCN on MERL Shopping (train split), select on validation, save to Drive.

Usage (repository root)::

    .venv/bin/python ml/merl/train.py                     # TCN, default settings
    .venv/bin/python ml/merl/train.py --model linear      # per-step baseline without temporal context

Outputs on Drive under ``checkpoints/merl_tcn_v1/<run>/``: ``model.pt`` (weights, normalisation,
per-class thresholds chosen on validation), ``config.json`` and ``train_log.json``. The test split
is not touched here; run ``ml/merl/evaluate.py`` once afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "packages" / "shop_core"))

from ml.merl import drive  # noqa: E402
from ml.merl.dataset import CLASSES, Video, load  # noqa: E402
from ml.merl.metrics import average_precision, best_f1_threshold  # noqa: E402
from shop_core.temporal.tcn import ActionTCN, Checkpoint, TCNConfig  # noqa: E402

DRIVE_CKPT = "checkpoints/merl_tcn_v1"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(videos: list[Video], size: int, mean: np.ndarray, std: np.ndarray, shuffle: bool):
    order = list(range(len(videos)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), size):
        chunk = [videos[j] for j in order[i : i + size]]
        t_max = max(len(v.features) for v in chunk)
        c, k = chunk[0].features.shape[1], chunk[0].labels.shape[1]
        x = np.zeros((len(chunk), c, t_max), dtype=np.float32)
        y = np.zeros((len(chunk), k, t_max), dtype=np.float32)
        m = np.zeros((len(chunk), 1, t_max), dtype=np.float32)
        for b, v in enumerate(chunk):
            t = len(v.features)
            x[b, :, :t] = ((v.features - mean) / std).T
            y[b, :, :t] = v.labels.T
            m[b, :, :t] = 1.0
        yield chunk, torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


@torch.inference_mode()
def predict(model: ActionTCN, videos: list[Video], mean: np.ndarray, std: np.ndarray) -> list[np.ndarray]:
    model.eval()
    out = []
    for chunk, x, _y, _m in batches(videos, 8, mean, std, shuffle=False):
        prob = torch.sigmoid(model(x)).numpy()
        out.extend(prob[b, :, : len(v.features)].T for b, v in enumerate(chunk))
    return out


def per_class_ap(probs: list[np.ndarray], videos: list[Video]) -> list[float]:
    p = np.concatenate(probs)
    y = np.concatenate([v.labels for v in videos])
    return [average_precision(p[:, k], y[:, k]) for k in range(len(CLASSES))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=("tcn", "linear"), default="tcn")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--run", default=None, help="run name (default: <model>-<utc timestamp>)")
    args = ap.parse_args()

    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    run = args.run or f"{args.model}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    data, info = load(("train", "val"))
    if info["missing"]:
        raise SystemExit(f"features missing on Drive for {len(info['missing'])} videos, e.g. {info['missing'][:3]}; "
                         "finish ml/merl/features.py first")
    train, val = data["train"], data["val"]
    all_train = np.concatenate([v.features for v in train])
    mean = all_train.mean(axis=0)
    std = np.maximum(all_train.std(axis=0), 1e-3)
    y_train = np.concatenate([v.labels for v in train])
    prevalence = y_train.mean(axis=0)
    pos_weight = torch.tensor(np.clip((1 - prevalence) / np.maximum(prevalence, 1e-6), 1.0, 10.0), dtype=torch.float32)
    print(f"train {len(train)} videos / {len(all_train)} steps, val {len(val)} videos; "
          f"prevalence {dict(zip(CLASSES, np.round(prevalence, 3).tolist()))}", flush=True)

    config = TCNConfig(in_channels=all_train.shape[1], classes=CLASSES, hidden=args.hidden,
                       dropout=args.dropout, temporal=args.model == "tcn")
    model = ActionTCN(config)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight.view(1, -1, 1))

    best_map, best_state, best_epoch, log = -1.0, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, steps = 0.0, 0.0
        for _chunk, x, y, m in batches(train, args.batch, mean, std, shuffle=True):
            loss = (bce(model(x), y) * m).sum() / (m.sum() * len(CLASSES))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * float(m.sum())
            steps += float(m.sum())
        sched.step()
        val_ap = per_class_ap(predict(model, val, mean, std), val)
        val_map = float(np.nanmean(val_ap))
        log.append({"epoch": epoch, "train_loss": total / steps, "val_map": val_map, "val_ap": val_ap})
        if val_map > best_map:
            best_map, best_epoch = val_map, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} loss {total / steps:.4f} val mAP {val_map:.4f} (best {best_map:.4f} @ {best_epoch})", flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    val_probs = predict(model, val, mean, std)
    p = np.concatenate(val_probs)
    y = np.concatenate([v.labels for v in val])
    thresholds, val_f1 = zip(*(best_f1_threshold(p[:, k], y[:, k]) for k in range(len(CLASSES))))
    val_ap = per_class_ap(val_probs, val)

    split_bytes = json.dumps({n: [v.name for v in data[n]] for n in data}, sort_keys=True).encode()
    meta = {
        "run": run,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "merl_shopping (research and educational use only)",
        "commercial_status": "development_only",
        "approved_for_commercial_release": False,
        "feature_schema": info["schema"],
        "split_digest": hashlib.sha256(split_bytes).hexdigest(),
        "best_epoch": best_epoch,
        "train_seconds": round(time.perf_counter() - started, 1),
        "val_map": float(np.nanmean(val_ap)),
        "val_ap": dict(zip(CLASSES, val_ap)),
        "val_f1_at_threshold": dict(zip(CLASSES, val_f1)),
        "receptive_field_steps": config.receptive_field,
        "args": vars(args),
        "semantics": "Uncalibrated per-step logits for observable shelf interactions. Not a theft score.",
    }
    ckpt = Checkpoint(config, best_state, torch.from_numpy(mean), torch.from_numpy(std), list(thresholds), meta)
    with tempfile.TemporaryDirectory(prefix="shop-merl-ckpt-") as tmp:
        t = Path(tmp)
        ckpt.save(t / "model.pt")
        (t / "config.json").write_text(json.dumps(meta, indent=2, default=float))
        (t / "train_log.json").write_text(json.dumps(log, indent=1, default=float))
        drive.upload_dir(t, f"{DRIVE_CKPT}/{run}")
    print(json.dumps({"run": run, "best_epoch": best_epoch, "val_map": round(meta["val_map"], 4),
                      "val_ap": {k: round(v, 4) for k, v in meta["val_ap"].items()},
                      "drive": drive.path(f"{DRIVE_CKPT}/{run}")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
