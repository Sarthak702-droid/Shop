"""Snapshot the RetailAction experiment's gate status for the Shop research dashboard.

Reads Drive ``raw/retailaction/TRANSFER_STATUS.json`` (via rclone) and the local preprocessing pilot,
then writes ``ml/results/retailaction/status.json``. Re-run after each experiment step::

    .venv/bin/python -I ml/retailaction/status_snapshot.py
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.merl import drive  # noqa: E402

OUT = REPO / "ml/results/retailaction/status.json"
STEPS = [
    ("archive_upload", "Training archives on Drive with publisher SHA-256"),
    ("data_report", "Train/validation data quality report"),
    ("learning_pilot", "Colab T4 learning pilot (32 clips)"),
    ("full_training", "Full training, validation-only selection"),
    ("validation_analysis", "Per-class AP, precision/recall, error analysis"),
    ("final_test", "Frozen model evaluated once on test"),
    ("pose_compatibility", "Raw-video pose path matches training features"),
]


def main() -> None:
    transfer = {}
    try:
        transfer = json.loads(drive._run("cat", drive.path("raw/retailaction/TRANSFER_STATUS.json")))  # noqa: SLF001
    except Exception as error:  # noqa: BLE001
        transfer = {"error": str(error)[:200]}
    files = {f["name"]: f for f in transfer.get("files", [])}
    verified = [n for n, f in files.items() if f.get("verified")]
    pilot_path = REPO / "var/retailaction/preprocessing_pilot.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.is_file() else {}
    compat_path = REPO / "ml/results/retailaction/pose_compatibility.json"
    compat = json.loads(compat_path.read_text()) if compat_path.is_file() else {}
    runs = []
    try:
        runs = [r.rstrip("/") for r in drive._run("lsf", "--dirs-only", drive.path("checkpoints/retailaction")).splitlines() if r.strip()]  # noqa: SLF001
    except Exception:  # noqa: BLE001
        runs = []
    upload_done = len(verified) == 3
    state = {
        "archive_upload": "complete" if upload_done else "in_progress",
        "data_report": "partial" if pilot else "pending",
        "learning_pilot": "pending",
        "full_training": "complete" if runs else "pending",
        "validation_analysis": "pending",
        "final_test": "pending",
        "pose_compatibility": "failed" if compat and not compat.get("verdict", {}).get("compatible") else ("pending" if not compat else "passed"),
    }
    snapshot = {
        "schema": "shop_retailaction_status_v1",
        "updated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "classes": ["take", "put", "touch"],
        "steps": [{"key": k, "label": label, "state": state[k]} for k, label in STEPS],
        "archives": [{"name": n, "bytes": f.get("expected_bytes"), "verified": bool(f.get("verified"))} for n, f in files.items()],
        "transfer_state": transfer.get("state"),
        "data": {
            "train_source_clips": pilot.get("source_samples"),
            "source_positive_clips": pilot.get("source_positive_clips"),
            "usable_positive_clips": pilot.get("usable_positive_clips"),
            "quarantined_metadata": len(pilot.get("quarantined_metadata") or []),
            "no_valid_pose_excluded": pilot.get("no_valid_pose_excluded"),
        },
        "class_imbalance_note": "touch has only ~60 usable positive training clips; its AP will be weak and have wide uncertainty.",
        "pose_compatibility": compat.get("verdict"),
        "runs_on_drive": runs,
        "runtime_enabled": False,
        "next_action": ("Wait for train.tar upload to finish, then run notebooks/Shop_RetailAction_T4.ipynb on a Colab T4 (pilot first)."
                        if not upload_done else "Run the Colab T4 learning pilot with notebooks/Shop_RetailAction_T4.ipynb."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=1) + "\n")
    print(json.dumps(snapshot, indent=1))


if __name__ == "__main__":
    main()
