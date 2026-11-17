"""Load MERL features + labels from Google Drive into memory (temp files are deleted on exit)."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml.merl import drive

CLASSES = ("reach_to_shelf", "retract_from_shelf", "hand_in_shelf", "inspect_product", "inspect_shelf")
FEATURES_DIR = "derived/features/merl_features_v1"
INTERVALS = "annotations/merl_shopping_v1/intervals.csv"
SPLIT = "splits/merl_shopping_official.json"
LABEL_OFFSET = 3  # label each 6-frame step by its centre frame


@dataclass
class Video:
    name: str
    features: np.ndarray  # (T, C) float32
    labels: np.ndarray    # (T, K) float32 in {0, 1}
    frame_index: np.ndarray


def step_labels(frame_index: np.ndarray, intervals: list[tuple[int, int, int]]) -> np.ndarray:
    """intervals: (class_idx, start_frame, end_frame) inclusive."""
    centre = frame_index + LABEL_OFFSET
    y = np.zeros((len(frame_index), len(CLASSES)), dtype=np.float32)
    for k, s, e in intervals:
        y[(centre >= s) & (centre <= e), k] = 1.0
    return y


def load(splits: tuple[str, ...] = ("train", "val", "test")) -> tuple[dict[str, list[Video]], dict]:
    with tempfile.TemporaryDirectory(prefix="shop-merl-train-") as tmp:
        t = Path(tmp)
        split = json.loads(drive.download(SPLIT, t / "split.json").read_text())
        rows = list(csv.DictReader(drive.download(INTERVALS, t / "intervals.csv").open()))
        schema = json.loads(drive.download(f"{FEATURES_DIR}/schema.json", t / "schema.json").read_text())
        by_video: dict[str, list[tuple[int, int, int]]] = {}
        for r in rows:
            by_video.setdefault(r["video"], []).append((int(r["merl_class_id"]) - 1, int(r["start_frame"]), int(r["end_frame"])))
        available = set(drive.list_files(FEATURES_DIR))
        # One bulk copy is much faster than 106 separate rclone calls.
        drive._run("copy", drive.path(FEATURES_DIR), str(t / "feat"), "--include", "*.npz")  # noqa: SLF001
        out: dict[str, list[Video]] = {}
        missing: list[str] = []
        for name in splits:
            out[name] = []
            for video in split["splits"][name]["videos"]:
                stem = video.removesuffix(".mp4")
                if f"{stem}.npz" not in available:
                    missing.append(video)
                    continue
                with np.load(t / "feat" / f"{stem}.npz") as z:
                    feats = z["features"].astype(np.float32)
                    fidx = z["frame_index"].astype(np.int64)
                out[name].append(Video(video, feats, step_labels(fidx, by_video.get(video, [])), fidx))
    return out, {"schema": schema, "missing": missing}
