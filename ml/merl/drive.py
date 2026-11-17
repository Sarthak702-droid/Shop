"""Small rclone wrapper so the MERL pipeline reads and writes Google Drive, not the local disk.

Remote and root come from the same environment variables as ``scripts/data/common.sh``.
Local files are only short-lived temporaries that callers delete immediately.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REMOTE = os.environ.get("SHOP_DRIVE_REMOTE", "shopdrive")
ROOT = os.environ.get("SHOP_DRIVE_ROOT", "ShopliftingAI-Data")
RCLONE = os.environ.get("RCLONE") or shutil.which("rclone") or str(Path.home() / ".local/bin/rclone")

_NOISE = "rclone's shared Google Drive client_id"


def path(rel: str) -> str:
    return f"{REMOTE}:{ROOT}/{rel.lstrip('/')}"


def _run(*args: str) -> str:
    proc = subprocess.run([RCLONE, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        err = "\n".join(line for line in proc.stderr.splitlines() if _NOISE not in line)
        raise RuntimeError(f"rclone {' '.join(args[:2])} failed: {err.strip()}")
    return proc.stdout


def list_files(rel: str) -> list[str]:
    try:
        return [line for line in _run("lsf", "--files-only", path(rel)).splitlines() if line]
    except RuntimeError as error:
        if "directory not found" in str(error):
            return []
        raise


def download(rel: str, local: Path) -> Path:
    local.parent.mkdir(parents=True, exist_ok=True)
    _run("copyto", path(rel), str(local))
    return local


def upload(local: Path, rel: str) -> None:
    _run("copyto", str(local), path(rel))


def upload_dir(local: Path, rel: str) -> None:
    _run("copy", str(local), path(rel))
