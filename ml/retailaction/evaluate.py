"""Frozen, one-model final clip evaluation. Never called by model selection/training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'packages/shop_core')]
from ml.retailaction.data import EXPECTED, SCHEMA_HASH, check_archive, check_drive_gate, load_archive, sha256  # noqa: E402
from ml.retailaction.state import atomic_json, run_lock  # noqa: E402
from ml.retailaction.train import ClipModel, metrics, predict, code_digest  # noqa: E402


def freeze(registry, checkpoint_hash):
    identity={'checkpoint_sha256':checkpoint_hash,'schema_hash':SCHEMA_HASH,
              'test_archive_sha256':EXPECTED['test.tar'][1]}
    with run_lock(registry.with_name(registry.name+'.reservation-lock')):
        if registry.exists():
            previous=json.loads(registry.read_text())
            if any(previous.get(k)!=v for k,v in identity.items()):
                raise ValueError('Final test already reserved for another frozen model; do not tune against it')
            return previous
        record={**identity,'frozen_utc':datetime.now(timezone.utc).isoformat(),'state':'reserved',
                'selection':'validation only','score_semantics':'uncalibrated clip action presence'}
        atomic_json(record,registry)
        return record


def canonical_registry(dataset_root):
    return dataset_root/'evaluation/retailaction_clip_tcn_v1'/('final_test_'+EXPECTED['test.tar'][1]+'.json')


def persistent_output(dataset_root, requested):
    base=(dataset_root/'evaluation/retailaction_clip_tcn_v1').resolve()
    requested=requested.resolve()
    if not requested.is_relative_to(base):
        raise ValueError('Final outputs must be under the persistent dataset directory '+str(base))
    return requested


def existing_result(record, requested, dataset_root):
    stored=(dataset_root/record['report_relative']).resolve()
    predictions=(dataset_root/record['predictions_relative']).resolve()
    persistent_output(dataset_root,stored.parent)
    persistent_output(dataset_root,predictions.parent)
    if requested.resolve() != stored.parent:
        raise ValueError('Final test is already recorded at '+str(stored.parent))
    if not stored.is_file() or not predictions.is_file():
        raise ValueError('Recorded final-test outputs are missing at '+str(stored.parent))
    if sha256(stored)!=record['report_sha256'] or sha256(predictions)!=record['predictions_sha256']:
        raise ValueError('Recorded final-test outputs are corrupt at '+str(stored.parent))
    return stored


def check_completion(completion, checkpoint_hash, config):
    if not (completion.get('state')=='complete' and completion.get('checkpoint_sha256')==checkpoint_hash
            and completion.get('config')==config):
        raise ValueError('Full validation training has not completed for this selected checkpoint')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,required=True)
    args=parser.parse_args()
    args.dataset_root=args.dataset_root.resolve()
    args.out=persistent_output(args.dataset_root,args.out)
    args.registry=canonical_registry(args.dataset_root)
    check_drive_gate(args.dataset_root/'raw/retailaction/TRANSFER_STATUS.json')
    if args.archive.name!='test.tar': raise ValueError('Expected the untouched official test archive')
    with run_lock(args.checkpoint.parent):
        evaluate_run(args)


def evaluate_run(args):
    # Hash and deserialize exactly the same snapshot, even if another process replaces best.pt.
    snapshot=args.checkpoint.read_bytes()
    checkpoint_hash=hashlib.sha256(snapshot).hexdigest()
    saved=torch.load(io.BytesIO(snapshot),map_location='cpu',weights_only=True)
    if not (saved.get('format')=='shop_retailaction_clip_tcn_v1'
            and saved['config']['schema_hash']==SCHEMA_HASH and not saved['config']['pilot']):
        raise ValueError('Incompatible or pilot checkpoint')
    if saved["config"].get("code_sha256") != code_digest():
        raise ValueError("Inference code differs from the validation-selected implementation")
    completion=json.loads((args.checkpoint.parent/'completed.json').read_text())
    check_completion(completion,checkpoint_hash,saved['config'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ClipModel(saved['config']['hidden'],dropout=0.1).to(device)
    model.load_state_dict(saved['model'])  # Validate weights before reserving/reading test labels.
    check_archive(args.archive)
    with run_lock(args.registry):
        record=freeze(args.registry,checkpoint_hash)
        if record['state']=='complete':
            print('Final evaluation already completed:',existing_result(record,args.out,args.dataset_root),flush=True)
            return
        loaded=load_archive(args.archive,'test',allow_final_test=True)
        if loaded[4]['source_samples']!=2501: raise ValueError('Unexpected official test sample count')
        development=json.loads((args.checkpoint.parent/'data_report.json').read_text())
        development_ids=set(development['train_ids'])|set(development['validation_ids'])
        if development_ids & set(loaded[3]): raise ValueError('Test sample IDs overlap development data')
        scores=predict(model,loaded[:3],128,device)
        report={**metrics(scores,loaded[2]),'checkpoint_sha256':checkpoint_hash,
                'schema_hash':SCHEMA_HASH,'data_quality':loaded[4],
                'runtime_enabled':False,'commercial_release_approved':False,
                'limits':'Clip presence only; not temporal/spatial localization, real-store event utility, calibration, or theft'}
        args.out.mkdir(parents=True,exist_ok=True)
        report_path=args.out/'final_clip_metrics.json'
        atomic_json(report,report_path)
        # The per-registry lock protects the result and completion publication in this runtime.
        np.savez_compressed(args.out/'final_clip_predictions.npz',scores=scores,
                            targets=loaded[2],sample_ids=np.asarray(loaded[3]))
        record.update(state='complete',report_relative=str(report_path.relative_to(args.dataset_root)),
                      report_sha256=sha256(report_path),
                      predictions_relative=str((args.out/'final_clip_predictions.npz').relative_to(args.dataset_root)),
                      predictions_sha256=sha256(args.out/'final_clip_predictions.npz'),
                      completed_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(record,args.registry)
        print(json.dumps({k:v for k,v in report.items() if k!='data_quality'},indent=2),flush=True)


if __name__=='__main__': main()
