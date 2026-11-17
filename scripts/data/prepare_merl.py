"""Convert MERL Shopping labels into Shop's dataset artefacts.

Reads the 106 ``xx_yy_label.mat`` files (cell array ``tlabs``; ``tlabs{k}`` is a K x 2 array of
start/end frames for action k) and writes:

* ``annotations/merl_shopping_v1/intervals.csv``: one row per labelled action interval
* ``splits/merl_shopping_official.json``: the publisher's subject-grouped split
* ``manifests/merl_shopping.json``: provenance, rights and class mapping

Usage (from the repository root)::

    .venv/bin/python -I scripts/data/prepare_merl.py <labels_dir> <out_dir> [--md5 md5sums.txt]

``out_dir`` mirrors the Drive root; upload it with ``rclone copy``. Frame numbers are kept exactly as
published. The ReadMe does not state 0- or 1-based indexing, so no conversion is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.io as sio

FPS = 30.0  # verified on 1_1_crop.mp4 (30 fps, 3920 frames); max label frame across files is 5951

# Publisher class order (ReadMe "Labels" section), index 0 == tlabs{1}.
MERL_CLASSES = (
    "reach_to_shelf",
    "retract_from_shelf",
    "hand_in_shelf",
    "inspect_product",
    "inspect_shelf",
)

# How each MERL class relates to Shop's ACTION_LABELS (packages/shop_core/shop_core/contracts.py).
# Only reach_to_shelf is the same definition. Nothing in MERL supervises hand_near_bag or
# item_visibility_lost, and no MERL class means "product picked up".
SHOP_MAPPING = {
    "reach_to_shelf": {"shop_label": "reach_to_shelf", "relation": "same"},
    "retract_from_shelf": {"shop_label": "return_to_shelf", "relation": "related_not_equal",
                           "note": "MERL retract = hand leaves shelf; Shop return_to_shelf = item put back"},
    "hand_in_shelf": {"shop_label": None, "relation": "auxiliary"},
    "inspect_product": {"shop_label": "product_pickup_observed", "relation": "weak_proxy",
                        "note": "Product held in hand implies an earlier pickup; boundaries differ"},
    "inspect_shelf": {"shop_label": None, "relation": "auxiliary_negative_context"},
}

# Publisher split (ReadMe): subjects 1-20 train, 21-26 validation, 27-41 test.
def split_of(subject: int) -> str:
    if 1 <= subject <= 20:
        return "train"
    if 21 <= subject <= 26:
        return "val"
    if 27 <= subject <= 41:
        return "test"
    raise ValueError(f"unexpected subject id {subject}")


NAME = re.compile(r"^(\d+)_(\d+)_label\.mat$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--md5", type=Path, help="rclone md5sum output for the videos and labels")
    args = ap.parse_args()

    rows: list[dict] = []
    videos: dict[str, dict] = {}
    for path in sorted(args.labels_dir.glob("*_label.mat"), key=lambda p: tuple(map(int, NAME.match(p.name).groups()))):
        subject, session = map(int, NAME.match(path.name).groups())
        video = f"{subject}_{session}_crop.mp4"
        tlabs = sio.loadmat(path)["tlabs"].ravel()
        if len(tlabs) != len(MERL_CLASSES):
            raise SystemExit(f"{path.name}: expected {len(MERL_CLASSES)} classes, found {len(tlabs)}")
        counts = {}
        for class_idx, intervals in enumerate(tlabs):
            arr = np.asarray(intervals, dtype=np.int64).reshape(-1, 2)
            if (arr[:, 1] < arr[:, 0]).any():
                raise SystemExit(f"{path.name}: end before start in class {class_idx + 1}")
            counts[MERL_CLASSES[class_idx]] = int(len(arr))
            for start, end in arr.tolist():
                rows.append({
                    "video": video,
                    "subject_id": subject,
                    "session_id": session,
                    "split": split_of(subject),
                    "merl_class_id": class_idx + 1,
                    "merl_class": MERL_CLASSES[class_idx],
                    "shop_label": SHOP_MAPPING[MERL_CLASSES[class_idx]]["shop_label"] or "",
                    "start_frame": start,
                    "end_frame": end,
                    "start_ms": round(start / FPS * 1000),
                    "end_ms": round(end / FPS * 1000),
                })
        videos[video] = {"subject_id": subject, "session_id": session, "split": split_of(subject),
                         "label_file": path.name, "intervals": counts}

    if len(videos) != 106:
        raise SystemExit(f"expected 106 label files, found {len(videos)}")

    ann = args.out_dir / "annotations" / "merl_shopping_v1"
    ann.mkdir(parents=True, exist_ok=True)
    with (ann / "intervals.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split_doc = {"dataset": "merl_shopping", "grouping": "subject_id (a subject never crosses splits)",
                 "source": "publisher ReadMe", "splits": {}}
    for name in ("train", "val", "test"):
        members = sorted((v for v, meta in videos.items() if meta["split"] == name),
                         key=lambda v: tuple(map(int, v.split("_")[:2])))
        split_doc["splits"][name] = {"videos": members, "count": len(members)}
    (args.out_dir / "splits").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "splits" / "merl_shopping_official.json").write_text(json.dumps(split_doc, indent=2) + "\n")

    md5 = {}
    if args.md5 and args.md5.exists():
        for line in args.md5.read_text().splitlines():
            digest, _, name = line.partition("  ")
            md5[name.strip()] = digest
    per_class = {c: sum(1 for r in rows if r["merl_class"] == c) for c in MERL_CLASSES}
    manifest = {
        "dataset": "merl_shopping",
        "version": "publisher release (106 videos, revised boundaries)",
        "acquired_by": "user browser download, organised by scripts/data on Google Drive",
        "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_page": "https://www.merl.com/demos/merl-shopping-dataset",
        "distribution": "https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/",
        "paper": "Singh et al., CVPR 2016, MERL TR2016-080",
        "rights": {
            "grant": "Permission to use these data and their documentation without fee for research and "
                     "educational purposes (publisher ReadMe, Legalese)",
            "commercial_use": "not granted",
            "approved_for_commercial_release": False,
        },
        "media": {"videos": 106, "subjects": 41, "fps": FPS, "resolution": "920x680",
                  "camera": "fixed overhead, mock grocery store", "audio": "not used"},
        "classes": list(MERL_CLASSES),
        "shop_mapping": SHOP_MAPPING,
        "interval_counts": per_class,
        "intervals_total": len(rows),
        "frame_indexing": "as published; base (0/1) not stated by publisher",
        "videos": {v: {**meta, "md5": md5.get(f"videos/{v}"),
                       "label_md5": md5.get(f"labels/{meta['label_file']}")} for v, meta in videos.items()},
    }
    (args.out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifests" / "merl_shopping.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"videos={len(videos)} intervals={len(rows)} per_class={per_class}")
    print({name: split_doc["splits"][name]["count"] for name in split_doc["splits"]})


if __name__ == "__main__":
    main()
