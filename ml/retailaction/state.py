"""Crash-recoverable local-runtime locks and same-filesystem artifact publication.

Use one Colab runtime per Drive run directory. These advisory locks coordinate processes in
one runtime; a mounted Drive directory is not a distributed locking service across machines.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid


@contextmanager
def run_lock(identity: Path):
    folder=Path(tempfile.gettempdir())/'shop-retailaction-locks'
    folder.mkdir(exist_ok=True)
    name=hashlib.sha256(str(identity.resolve()).encode()).hexdigest()+'.lock'
    with (folder/name).open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is already writing this run/evaluation') from exc
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def atomic_bytes(value: bytes, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('wb') as out:
            out.write(value)
            out.flush(); os.fsync(out.fileno())
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(value, path: Path):
    atomic_bytes(json.dumps(value,indent=2,allow_nan=False).encode(),path)


def resume_epoch(epoch, best_epoch, patience, pilot):
    if not pilot and epoch-best_epoch >= patience:
        return None
    return epoch+1


def check_resume_budget(requested, saved):
    if requested < saved:
        raise ValueError("Cannot shrink the saved epoch budget on resume")
