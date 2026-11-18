"""T4-only offline clip baseline. Does not change MERL or enable Shop runtime inference."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "packages/shop_core")]
from ml.retailaction.data import (  # noqa: E402
    CLASSES, FEATURES, SCHEMA, SCHEMA_HASH, EXPECTED, QualityGateError, check_archive, check_drive_gate, load_archive, sha256,
)
from shop_core.temporal.tcn import ActionTCN, TCNConfig  # noqa: E402
from ml.retailaction.state import atomic_json, atomic_bytes, run_lock, resume_epoch, check_resume_budget  # noqa: E402


class ClipModel(nn.Module):
    def __init__(self, hidden=32, dropout=0.1):
        super().__init__()
        self.network = ActionTCN(TCNConfig(FEATURES, CLASSES, hidden=hidden,
                                           dilations=(1, 2, 4), dropout=dropout))

    def forward(self, x, valid):
        batch, views, channels, steps = x.shape
        logits = self.network(x.reshape(batch*views, channels, steps))
        mask = valid.reshape(batch*views, 1, steps)
        pooled = logits.masked_fill(~mask, -10000).max(dim=-1).values.reshape(batch, views, 3)
        present = valid.any(dim=-1)
        if not present.any(dim=1).all():
            raise ValueError("An example has no observed pose in either view")
        return (pooled * present.unsqueeze(-1)).sum(dim=1) / present.sum(dim=1, keepdim=True)


def average_precision(scores, labels):
    """Grouped tied scores; returns null if there are no positives."""
    labels = np.asarray(labels, dtype=bool)
    if not labels.any():
        return None
    order = np.argsort(-scores, kind="stable")
    scores, labels = np.asarray(scores)[order], labels[order]
    end = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    tp = np.cumsum(labels)[end]
    recall = tp / labels.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * tp/(end+1)))


def metrics(prob, truth):
    ap = [average_precision(prob[:, k], truth[:, k]) for k in range(3)]
    if not np.isfinite(prob).all():
        raise ValueError("Nonfinite validation predictions")
    supported = [v for v in ap if v is not None]
    return {"clip_map_supported_classes": float(np.mean(supported)) if supported else None,
            "supported_classes": len(supported), "clip_ap": dict(zip(CLASSES, ap)),
            "positive_clips": dict(zip(CLASSES, truth.sum(axis=0).astype(int).tolist())),
            "constant_score_ap": {name:float(truth[:,k].mean()) if ap[k] is not None else None
                                  for k,name in enumerate(CLASSES)},
            "negative_clips": dict(zip(CLASSES,(len(truth)-truth.sum(axis=0)).astype(int).tolist())),
            "clips": len(truth), "unit": "paired-view clip; not per-frame or event localization"}


def batches(data, batch, order):
    x, m, y = data
    for i in range(0, len(order), batch):
        idx = order[i:i+batch]
        yield tuple(torch.from_numpy(a[idx]) for a in (x, m, y))


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


@torch.inference_mode()
def predict(model, data, batch, device):
    model.eval()
    outputs = []
    for x, valid, _ in batches(data, batch, np.arange(len(data[0]))):
        outputs.append(model(x.to(device), valid.to(device)).sigmoid().cpu().numpy())
    return np.concatenate(outputs)


def code_digest():
    return hashlib.sha256(b"".join(p.read_bytes() for p in [
        Path(__file__), REPO/"ml/retailaction/data.py", REPO/"ml/retailaction/state.py",
        REPO/"packages/shop_core/shop_core/temporal/tcn.py"])).hexdigest()


def selected_bundle(last):
    return {"format":last["format"], "model":last["best_model"], "epoch":last["best_epoch"],
            "best_epoch":last["best_epoch"], "best":last["best"], "config":last["config"],
            "feature_schema":last["feature_schema"], "runtime_enabled":False,
            "commercial_release_approved":False, "semantics":last["semantics"]}


def recover_artifacts(saved, out):
    payload = saved["best_artifact"]
    digest = saved["best_artifact_sha256"]
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Committed selected artifact is corrupt")
    path = out/"best.pt"
    if not path.exists() or sha256(path) != digest:
        atomic_bytes(payload, path)
    atomic_json(saved["history"], out/"history.json")
    return digest, payload


def publish_epoch(bundle, out, improved):
    atomic_save(bundle, out/"last.pt")
    if improved:
        atomic_bytes(bundle["best_artifact"], out/"best.pt")
    atomic_json(bundle["history"], out/"history.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--drive-status", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-report", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (args.epochs > 0 and args.patience > 0 and args.batch > 0 and args.hidden > 0
            and np.isfinite(args.lr) and args.lr > 0):
        parser.error("Training parameters must be positive")
    if not 0 <= args.seed <= 2**32-1:
        parser.error("seed must be between 0 and 2**32-1")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise SystemExit("Select a Google Colab T4 GPU runtime; CPU/other GPU fallback is disabled")
    device = torch.device("cuda")
    with run_lock(args.out):
        try:
            train_run(args, device)
        except QualityGateError as error:
            failure=args.out/'attempts'/('rejected-'+uuid.uuid4().hex+'.json')
            atomic_json({'error':str(error),'data_quality':error.report},failure)
            print('Rejected split report:',failure,flush=True)
            raise


def train_run(args, device):
    check_drive_gate(args.drive_status)
    for name in ("train.tar", "validation.tar"):
        check_archive(args.archives/name)
    if not args.pilot:
        if not args.pilot_report:
            raise ValueError("Full training requires --pilot-report from a passing T4 pilot")
        pilot = json.loads(args.pilot_report.read_text())
        if not (pilot.get("passed") is True and pilot.get("schema_hash") == SCHEMA_HASH
                and pilot.get("train_archive_sha256") == EXPECTED["train.tar"][1]
                and "T4" in pilot.get("gpu", "") and pilot.get("code_sha256") == code_digest()
                and pilot.get("hidden") == args.hidden):
            raise ValueError("Pilot evidence is absent or incompatible")
    train = load_archive(args.archives/"train.tar", "train", 32 if args.pilot else None)
    val = load_archive(args.archives/"validation.tar", "validation", 32 if args.pilot else None)
    if set(train[3]) & set(val[3]):
        raise ValueError("Sample IDs overlap across train and validation")
    for name, loaded, expected in (("train",train,17222),("validation",val,1277)):
        if loaded[4]["source_samples"] != expected:
            raise ValueError(f"Unexpected {name} source sample count")
    train_data, val_data = train[:3], val[:3]
    args.out.mkdir(parents=True, exist_ok=True)
    data_report = {"schema": SCHEMA, "schema_hash": SCHEMA_HASH,
                   "train": train[4], "validation": val[4], "train_ids": train[3],
                   "validation_ids": val[3], "test_loaded": False}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ClipModel(args.hidden, 0 if args.pilot else 0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    # No learned normalization fitted on validation; coordinates/masks use the frozen schema.
    positives = train[2].sum(axis=0)
    if not args.pilot and ((positives == 0).any() or (positives == len(train[2])).any()):
        raise ValueError("Every full-training class needs positive and negative clips")
    weights = np.clip((len(train[2])-positives)/np.maximum(positives, 1), 1, 20)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,device=device))
    scaler = torch.amp.GradScaler("cuda")
    history, start, best, best_epoch, best_model = [], 1, -1.0, 0, None
    best_artifact_sha256, best_artifact = None, None
    config = {"schema_hash": SCHEMA_HASH, "hidden": args.hidden, "pilot": args.pilot,
              "batch": args.batch, "lr": args.lr, "seed": args.seed, "patience": args.patience,
              "source_train_sha256": EXPECTED['train.tar'][1], "class_order": list(CLASSES),
              "code_sha256": code_digest()}
    last = args.out/"last.pt"
    if args.resume:
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("Resume config does not match the saved run")
        check_resume_budget(30 if args.pilot else args.epochs, saved["epoch_budget"])
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        start = resume_epoch(saved["epoch"], saved["best_epoch"], args.patience, args.pilot)
        best, best_epoch, best_model = saved["best"], saved["best_epoch"], saved["best_model"]
        best_artifact_sha256, best_artifact = recover_artifacts(saved, args.out)
    elif last.exists() or (args.out/"best.pt").exists():
        raise ValueError("Output run already exists; use --resume or choose a new run directory")
    atomic_json(data_report, args.out/"data_report.json")
    print('GPU',torch.cuda.get_device_name(0),'train_clips',len(train[0]),'val_clips',len(val[0]),flush=True)
    (args.out/"completed.json").unlink(missing_ok=True)
    total_epochs = 30 if args.pilot else args.epochs
    for epoch in range(start or total_epochs+1, total_epochs+1):
        torch.manual_seed(args.seed+epoch)
        order = np.random.default_rng(args.seed+epoch).permutation(len(train[0]))
        model.train(); total_loss=0.0; count=0
        for x, valid, truth in batches(train_data, args.batch, order):
            x, valid, truth = x.to(device), valid.to(device), truth.to(device)
            for overflow_attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = criterion(model(x,valid),truth)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss; no checkpoint promotion")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                finite = bool(torch.isfinite(norm))
                if finite:
                    scaler.step(optimizer); scaler.update()
                    break
                # Skip the optimizer even if finite components overflowed only the global norm.
                scaler.update(new_scale=max(scaler.get_scale()/2, 1e-8))
                print(f"AMP overflow: skipped update {overflow_attempt+1}/8", flush=True)
            else:
                raise ValueError("Persistent nonfinite gradients after eight skipped updates")
            total_loss += loss.item()*len(x); count += len(x)
        result = metrics(predict(model,val_data,args.batch,device),val[2])
        score = result['clip_map_supported_classes']
        if score is None:
            raise ValueError("Validation has no supported action class")
        row={'epoch':epoch,'train_loss':total_loss/count,'validation':result}
        history.append(row)
        improved = score > best
        if improved:
            best,best_epoch = score,epoch
            best_model = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        bundle={'format':'shop_retailaction_clip_tcn_v1','model':model.state_dict(),
                'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                'config':config,'feature_schema':SCHEMA,'epoch':epoch,'epoch_budget':total_epochs,'best':best,'best_model':best_model,
                'best_epoch':best_epoch,'history':history,'commercial_release_approved':False,
                'runtime_enabled':False,'semantics':'Uncalibrated clip action presence; not theft; not temporal localization'}
        if improved:
            buffer = io.BytesIO()
            torch.save(selected_bundle(bundle), buffer)
            best_artifact = buffer.getvalue()
            best_artifact_sha256 = hashlib.sha256(best_artifact).hexdigest()
        bundle['best_artifact_sha256'] = best_artifact_sha256
        bundle['best_artifact'] = best_artifact
        # Commit a recoverable optimization/selection transaction before publishing best.pt.
        publish_epoch(bundle,args.out,improved)
        print(json.dumps(row),flush=True)
        if not args.pilot and epoch-best_epoch >= args.patience: break
    if args.pilot:
        all_losses=[r['train_loss'] for r in history]
        passed=len(all_losses)>=10 and np.mean(all_losses[-5:]) < np.mean(all_losses[:5])
        report={'passed':bool(passed),'gpu':torch.cuda.get_device_name(0),'schema_hash':SCHEMA_HASH,
                'train_archive_sha256':EXPECTED['train.tar'][1], 'code_sha256':code_digest(), 'hidden':args.hidden,
                'first_five_loss_mean':float(np.mean(all_losses[:5])),
                'last_five_loss_mean':float(np.mean(all_losses[-5:])),
                'note':'Tiny-subset learning/finite-gradient pilot only; not an accuracy or deployment result'}
        atomic_json(report,args.out/'pilot_report.json')
        if not passed: raise RuntimeError('Learning pilot did not pass; investigate before full training')
    atomic_json({'state':'complete','best_epoch':best_epoch,
                 'checkpoint_sha256':sha256(args.out/'best.pt'), 'config':config,
                 'stop_reason':'epoch_limit_or_validation_early_stop'}, args.out/'completed.json')
    print('Saved checkpoints. Final test intentionally not evaluated. Runtime remains disabled.',flush=True)


if __name__ == '__main__':
    main()
