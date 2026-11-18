import json
import numpy as np
import pytest
import torch

from ml.retailaction.data import CLASSES, EXPECTED, FEATURES, MAX_STEPS, check_drive_gate, make_sample
from ml.retailaction.train import ClipModel, average_precision


def content():
    camera = {'poses': [
        {'pose': {'left_wrist': [0.2, 0.3]}, 'joint_scores': {'left_wrist': 0.8}},
        None,
        {'pose': {'left_wrist': [0.4, 0.5]}, 'joint_scores': {'left_wrist': 0.9}},
    ], 'frame_timestamps': ['1970-01-01T00:00:00Z', '1970-01-01T00:00:02Z',
                            '1970-01-01T00:00:10Z']}
    return {'labels': {'action': [{'label': 'take', 'start': 0.2, 'end': 0.5}]},
            'action_cam': {'rank0': camera, 'rank1': {**camera, 'frame_timestamps': None}}}


def test_missing_pose_and_unknown_time_have_masks_not_synthetic_samples():
    x, mask, y = make_sample(content())
    assert x.shape == (2, FEATURES, MAX_STEPS)
    assert mask[0, :3].tolist() == [True, False, True]
    assert not x[0, :96, 1].any()
    assert x[0, 96, 2] == pytest.approx(8/60)
    assert x[0, 97, 2] == 1
    assert not x[1, 96:].any()  # Entire view has unknown physical timestamps.
    assert y.tolist() == [1, 0, 0]


def test_unknown_annotations_do_not_become_negative_examples():
    item = content(); item['labels']['action'] = None
    with pytest.raises(ValueError, match='annotations'):
        make_sample(item)


def test_repeated_timestamp_rejected_and_no_merl_rename():
    item = content()
    item['action_cam']['rank0']['frame_timestamps'][1] = '1970-01-01T00:00:00Z'
    with pytest.raises(ValueError, match='timestamps'):
        make_sample(item)
    item = content(); item['labels']['action'][0]['label'] = 'reach_to_shelf'
    with pytest.raises(ValueError, match='Unknown action'):
        make_sample(item)


def test_clip_model_keeps_views_in_one_example_and_ignores_invalid_frames():
    torch.manual_seed(0)
    model = ClipModel(hidden=8, dropout=0).eval()
    x = torch.randn(2, 2, FEATURES, 32)
    mask = torch.zeros(2, 2, 32, dtype=torch.bool); mask[:, :, :3] = True
    with torch.inference_mode():
        original = model(x, mask)
        x[:, :, :, 3:] += 1000  # Causal network cannot let right padding affect real steps.
        assert torch.allclose(original, model(x, mask))
        assert original.shape == (2, len(CLASSES))
    model.train(); model(x, mask).square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ap_is_tie_order_invariant_and_undefined_without_positives():
    assert average_precision(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    assert average_precision(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert average_precision(np.array([0.8, 0.1]), np.array([0, 0])) is None


def test_drive_gate_requires_all_original_hashes(tmp_path):
    path=tmp_path/'status.json'
    data={'state':'complete','files':[{'name':n,'verified':True,'actual_bytes':s,
                                      'actual_sha256':h} for n,(s,h) in EXPECTED.items()]}
    path.write_text(json.dumps(data)); check_drive_gate(path)
    data['files'][0]['actual_sha256']='wrong'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='verification'):
        check_drive_gate(path)


def test_off_image_pose_is_masked_instead_of_clipped_into_validity():
    item = content()
    item['action_cam']['rank0']['poses'][0]['pose']['left_wrist'] = [-0.1, 1.2]
    x, mask, _ = make_sample(item)
    assert not mask[0, 0]
    assert not x[0, :96, 0].any()


def test_actual_archive_longer_sequence_is_preserved_and_missing_view_allowed():
    item = content()
    pose = item['action_cam']['rank0']['poses'][0]
    item['action_cam']['rank0'] = {'poses': [pose] * 63, 'frame_timestamps': None}
    item['action_cam']['rank1'] = {'poses': None, 'frame_timestamps': None}
    x, valid, _ = make_sample(item)
    assert valid[0].sum() == 63
    assert not valid[1].any()
    assert not x[:, 96:].any()


def test_final_test_cannot_be_reused_for_model_selection(tmp_path):
    from ml.retailaction.evaluate import freeze
    registry = tmp_path/'final.json'
    first = freeze(registry, 'one-frozen-checkpoint')
    assert freeze(registry, 'one-frozen-checkpoint') == first
    with pytest.raises(ValueError, match='another frozen model'):
        freeze(registry, 'another-checkpoint')


def test_resume_does_not_extend_an_already_early_stopped_run():
    from ml.retailaction.state import resume_epoch
    assert resume_epoch(15, 5, 10, False) is None
    assert resume_epoch(14, 5, 10, False) == 15
    assert resume_epoch(15, 5, 10, True) == 16


def test_best_bundle_uses_committed_winning_weights():
    from ml.retailaction.train import selected_bundle
    saved = {'format':'shop_retailaction_clip_tcn_v1', 'model':{'w':torch.tensor(99)},
             'best_model':{'w':torch.tensor(5)}, 'epoch':15, 'best_epoch':5,
             'best':0.7, 'config':{}, 'feature_schema':{}, 'semantics':'clip'}
    bundle = selected_bundle(saved)
    assert bundle['model']['w'].item() == 5
    assert bundle['epoch'] == 5
    assert 'optimizer' not in bundle


def test_incomplete_validation_run_cannot_reserve_final_test():
    from ml.retailaction.evaluate import check_completion
    with pytest.raises(ValueError, match='not completed'):
        check_completion({'state':'running'}, 'abc', {})
    check_completion({'state':'complete','checkpoint_sha256':'abc','config':{}}, 'abc', {})


def test_run_lock_releases_and_rejects_another_writer(tmp_path):
    from ml.retailaction.state import run_lock
    with run_lock(tmp_path/'run'):
        with pytest.raises(RuntimeError, match='already writing'):
            with run_lock(tmp_path/'run'):
                pass
    with run_lock(tmp_path/'run'):
        pass


def test_numeric_strings_are_not_accepted_as_joint_coordinates():
    item = content()
    item['action_cam']['rank0']['poses'][0]['pose']['left_wrist'] = '01'
    with pytest.raises(ValueError, match='numeric pair'):
        make_sample(item)


@pytest.mark.parametrize('bad', [{'labels':None}, {'labels':{'action':[None]}}])
def test_bad_metadata_is_quarantined_and_pilot_retention_stays_bounded(tmp_path,bad):
    import io, tarfile
    from ml.retailaction.data import load_archive
    path=tmp_path/'train.tar'
    with tarfile.open(path,'w') as archive:
        for n in range(101):
            item=bad if n==0 else content()
            for name, payload in [('rank0_video.mp4',b'fixture'),('rank1_video.mp4',b'fixture'),
                                  ('metadata.json',json.dumps({'content':item}).encode())]:
                member=tarfile.TarInfo(f'train/{n:06d}/{name}'); member.size=len(payload)
                archive.addfile(member,io.BytesIO(payload))
    x, mask, y, ids, info=load_archive(path,'train',limit=2)
    assert len(x)==2 and ids==['000001','000002']
    assert len(info['quarantined_metadata'])==1
    assert info['source_positive_clips']['take']==100
    assert info['usable_positive_clips']['take']==100


def test_empty_view_with_timestamps_and_invalid_camera_are_not_silently_accepted():
    item = content()
    item['action_cam']['rank0'] = {'poses': [], 'frame_timestamps':['1970-01-01T00:00:00Z']}
    with pytest.raises(ValueError, match='count mismatch'):
        make_sample(item)
    item['action_cam']['rank0'] = None
    with pytest.raises(ValueError, match='Camera metadata'):
        make_sample(item)


def test_known_rare_label_is_counted_when_another_label_is_malformed(tmp_path):
    import io, tarfile
    from ml.retailaction.data import load_archive
    path=tmp_path/'train.tar'
    with tarfile.open(path,'w') as archive:
        for n in range(201):
            item = content()
            if n < 11: item['labels']['action'][0]['label'] = 'touch'
            if n < 2: item['labels']['action'].append({'label':[],'start':0.1,'end':0.2})
            for name,payload in [('rank0_video.mp4',b'f'),('rank1_video.mp4',b'f'),
                                  ('metadata.json',json.dumps({'content':item}).encode())]:
                member=tarfile.TarInfo(f'train/{n:06d}/{name}'); member.size=len(payload)
                archive.addfile(member,io.BytesIO(payload))
    with pytest.raises(ValueError, match='over 10%') as error:
        load_archive(path,'train',limit=2)
    assert len(error.value.report['quarantined_metadata'])==2
    assert error.value.report['source_positive_clips']['touch']==11


def test_interrupted_selected_publication_recovers_exact_bytes(tmp_path,monkeypatch):
    import hashlib
    from ml.retailaction import train
    payload=b'exact selected checkpoint bytes'
    bundle={'best_artifact':payload,'best_artifact_sha256':hashlib.sha256(payload).hexdigest(),
            'history':[{'epoch':1,'train_loss':0.4}]}
    original=train.atomic_bytes
    def interrupted(value,path):
        if path.name=='best.pt': raise OSError('simulated interrupted publication')
        return original(value,path)
    monkeypatch.setattr(train,'atomic_bytes',interrupted)
    with pytest.raises(OSError): train.publish_epoch(bundle,tmp_path,True)
    saved=torch.load(tmp_path/'last.pt',weights_only=True)
    assert not (tmp_path/'best.pt').exists()
    monkeypatch.setattr(train,'atomic_bytes',original)
    train.recover_artifacts(saved,tmp_path)
    assert (tmp_path/'best.pt').read_bytes()==payload
    assert json.loads((tmp_path/'history.json').read_text())==bundle['history']


def test_resume_cannot_shrink_the_saved_budget():
    from ml.retailaction.state import check_resume_budget
    with pytest.raises(ValueError, match='saved epoch budget'):
        check_resume_budget(6,60)
    check_resume_budget(60,60)
    check_resume_budget(90,60)


def test_reservation_suffix_is_not_a_self_lock(tmp_path):
    from ml.retailaction.evaluate import freeze
    from ml.retailaction.state import run_lock
    path=tmp_path/'final.reservation'
    with run_lock(path):
        assert freeze(path,'same-checkpoint')['state']=='reserved'


def test_false_and_list_joint_maps_are_not_accepted():
    for field,value in [('pose',[]),('joint_scores',False)]:
        item=content()
        item['action_cam']['rank0']['poses'][0][field]=value
        with pytest.raises(ValueError,match='joint maps'):
            make_sample(item)


def test_off_image_source_value_is_rejected_before_float32_rounding():
    item=content()
    item['action_cam']['rank0']['poses'][0]['pose']['left_wrist']=[1.00000001,0.5]
    x,valid,_=make_sample(item)
    assert not valid[0,0]
    assert not x[0,:96,0].any()


def test_boolean_action_endpoints_are_not_labels():
    item=content()
    item['labels']['action'][0].update(start=False,end=True)
    with pytest.raises(ValueError,match='Boolean'):
        make_sample(item)


def test_final_registry_is_canonical_for_the_dataset_root(tmp_path):
    from ml.retailaction.evaluate import canonical_registry
    a=canonical_registry(tmp_path)
    assert a==canonical_registry(tmp_path)
    assert EXPECTED['test.tar'][1] in a.name
    assert a.parent==tmp_path/'evaluation/retailaction_clip_tcn_v1'


def test_completed_evaluation_cannot_succeed_with_missing_or_different_outputs(tmp_path):
    from ml.retailaction.evaluate import existing_result
    from ml.retailaction.data import sha256
    out=tmp_path/'evaluation/retailaction_clip_tcn_v1/run'; out.mkdir(parents=True)
    report=out/'final_clip_metrics.json'; report.write_text('{}')
    pred=out/'final_clip_predictions.npz'; pred.write_bytes(b'fixture')
    record={'report_relative':str(report.relative_to(tmp_path)),
            'predictions_relative':str(pred.relative_to(tmp_path)),'report_sha256':sha256(report),
            'predictions_sha256':sha256(pred)}
    assert existing_result(record,out,tmp_path)==report
    with pytest.raises(ValueError,match='already recorded'):
        existing_result(record,out.parent/'different',tmp_path)
    pred.unlink()
    with pytest.raises(ValueError,match='missing'):
        existing_result(record,out,tmp_path)


@pytest.mark.parametrize('seed',['-1',str(2**32)])
def test_invalid_seed_is_rejected_before_gpu_or_archive_access(monkeypatch,tmp_path,seed):
    from ml.retailaction import train
    monkeypatch.setattr('sys.argv',['train','--archives',str(tmp_path),'--drive-status',str(tmp_path/'absent'),
                                     '--out',str(tmp_path/'run'),'--seed',seed])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code==2
    assert not (tmp_path/'run').exists()


def test_final_outputs_must_live_under_the_persistent_dataset_root(tmp_path):
    from ml.retailaction.evaluate import persistent_output
    with pytest.raises(ValueError,match='persistent dataset'):
        persistent_output(tmp_path,tmp_path/'ephemeral')
    target=tmp_path/'evaluation/retailaction_clip_tcn_v1/run'
    assert persistent_output(tmp_path,target)==target.resolve()


def test_absent_class_baseline_ap_is_also_undefined():
    from ml.retailaction.train import metrics
    result=metrics(np.array([[0.2,0.3,0.4],[0.7,0.4,0.2]]),np.array([[0,0,0],[1,0,0]]))
    assert result['clip_ap']['touch'] is None
    assert result['constant_score_ap']['touch'] is None
