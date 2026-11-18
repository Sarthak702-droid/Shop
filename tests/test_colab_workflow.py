"""Exercise notebook launch-state regressions without Google access or GPU training."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

NOTEBOOK=Path(__file__).resolve().parents[1]/'notebooks/Shop_RetailAction_T4.ipynb'


def cells():
    notebook=json.loads(NOTEBOOK.read_text())
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']


def context(tmp_path):
    root=tmp_path/'drive'; root.mkdir()
    local=tmp_path/'local'; local.mkdir()
    raw=root/'raw'; raw.mkdir()
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    expected={}
    for name in ('train.tar','validation.tar','test.tar'):
        path=raw/name; path.write_bytes(name.encode())
        expected[name]=(path.stat().st_size,digest(path))
    def check_archive(path):
        if (path.stat().st_size,digest(path)) != expected[path.name]: raise ValueError('bad cache')
    env={'DRIVE_ROOT':root,'LOCAL':local,'RAW':raw,'expected':expected,'sha256':digest,
         'check_archive':check_archive,'shutil':shutil,'Path':Path,'sys':sys,
         'CODE':tmp_path/'code','RUN_ID':'A','MODE':'train',
         'READINESS_STATE':{'root':str(root),'receipt':str(root/'status.json'),'token':'one'},
         'STAGING_STATE':None,'BUNDLE_STATE':{'root':str(root),'sha256':'verified'},
         'BUNDLE_SHA256':'verified','json':json}
    from ml.retailaction.state import run_lock
    env['run_lock']=run_lock
    wanted={'verified_sources','cached','stage','stage_final_inputs','run_paths','common_arguments'}
    functions=[node for source in cells() for node in ast.parse(source).body
               if isinstance(node,ast.FunctionDef) and node.name in wanted]
    exec(compile(ast.Module(body=functions,type_ignores=[]),'colab-functions','exec'),env)
    return env


def test_failed_test_copy_preserves_verified_development_inputs_for_retry(tmp_path,monkeypatch):
    env=context(tmp_path)
    env['stage'](('train.tar','validation.tar'))
    original=shutil.copyfile
    def fail_test(source,target):
        if Path(source).name=='test.tar': raise OSError('temporary copy interruption')
        return original(source,target)
    monkeypatch.setattr(shutil,'copyfile',fail_test)
    with pytest.raises(OSError): env['stage'](('test.tar',))
    assert set(env['STAGING_STATE']['archives'])=={'train.tar','validation.tar'}
    env['run_paths']()  # Allows the final cell to reach the test-copy retry.
    monkeypatch.setattr(shutil,'copyfile',original)
    env['stage'](('test.tar',))
    assert 'test.tar' in env['STAGING_STATE']['archives']


def test_failed_bundle_checksum_invalidates_old_launch_state(tmp_path):
    env=context(tmp_path)
    env['stage'](('train.tar','validation.tar'))
    folder=env['DRIVE_ROOT']/'notebooks'; folder.mkdir()
    (folder/'shop_retailaction_t4_source.zip').write_bytes(b'not the verified source bundle')
    env['hashlib']=hashlib
    source=next(c for c in cells() if c.startswith('BUNDLE_STATE = None'))
    with pytest.raises(RuntimeError,match='checksum mismatch'):
        exec(compile(source,'colab-bundle-cell','exec'),env)
    assert env['BUNDLE_STATE'] is None
    with pytest.raises(RuntimeError,match='source bundle verification'):
        env['common_arguments']()


def test_failed_readiness_clears_prior_staging_before_external_operations(tmp_path):
    env=context(tmp_path); env['stage'](('train.tar','validation.tar'))
    source=next(c for c in cells() if c.startswith('READINESS_STATE = None'))
    reset=ast.parse(source).body[:2]
    exec(compile(ast.Module(body=reset,type_ignores=[]),'readiness-reset','exec'),env)
    assert env['READINESS_STATE'] is None and env['STAGING_STATE'] is None
    with pytest.raises(RuntimeError): env['common_arguments']()


def test_current_root_and_run_are_used_for_every_launch(tmp_path):
    env=context(tmp_path); env['stage'](('train.tar','validation.tar'))
    other=tmp_path/'other'
    env['DRIVE_ROOT']=other; env['RUN_ID']='B'
    env['READINESS_STATE']={'root':str(other),'receipt':str(other/'status.json'),'token':'two'}
    env['STAGING_STATE']={'root':str(other),'token':'two','archives':['train.tar','validation.tar']}
    env['BUNDLE_STATE']={'root':str(other),'sha256':'verified'}
    assert env['common_arguments']()[-1]==str(other/'status.json')
    assert env['run_paths']()[2]==other/'checkpoints/retailaction_clip_tcn_v1/B/full'


def test_final_only_staging_releases_development_caches(tmp_path):
    env=context(tmp_path)
    env['stage'](('train.tar','validation.tar'))
    full=tmp_path/'completed-run'; full.mkdir()
    checkpoint=full/'best.pt'; checkpoint.write_bytes(b'selected validation checkpoint')
    (full/'completed.json').write_text(json.dumps({'state':'complete',
        'checkpoint_sha256':env['sha256'](checkpoint)}))
    env['MODE']='evaluate'
    env['stage_final_inputs'](full)
    assert env['STAGING_STATE']['archives']==['test.tar']
    assert not (env['LOCAL']/'train.tar').exists()
    assert not (env['LOCAL']/'validation.tar').exists()
    assert (env['RAW']/'train.tar').exists()
    env['run_paths']()
    with pytest.raises(RuntimeError): env['common_arguments']()
