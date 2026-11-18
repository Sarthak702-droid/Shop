"""Validated sparse-pose features and clip-presence targets from original RetailAction TARs.

No extraction of arbitrary archive paths. Both views stay in one training example. Original
ordered observations are retained; missing timestamps are masked, never invented at five Hz.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

CLASSES = ("take", "put", "touch")
JOINTS = (
    "top_of_head", "nose", "neck", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hand", "right_hand", "middle_of_waist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot",
)
FEATURE_SCHEMA = "retailaction_sparse_pose_clip_v1"
FEATURES = 100  # 24 * (supplied coordinate 0, coordinate 1, score, valid) + dt/mask/time/mask
MAX_STEPS = 128  # Bounded padding; actual archive contains pose sequences longer than the card's 32.
# No truncation or interpolation is performed. Exceeding this audited cap fails closed.
EXPECTED = {
    "train.tar": (26082080256, "272dcd90f2a12020dfe40542b54196dc36309a0e42298ff50f3c384956369ad0"),
    "validation.tar": (1937612800, "1287dc0b491f3eab5807d216bb96db2436da845ed0ea6e4ec69daf6182873836"),
    "test.tar": (3838385152, "26a376627f22df0702d1bb536dd073843d3720a6b4a1788d4d226513a3ac8a25"),
}
SCHEMA = {"version": FEATURE_SCHEMA, "joints": JOINTS, "class_order": CLASSES,
          "features": FEATURES, "max_steps": MAX_STEPS, "time_scale_seconds": 60.0,
          "coordinate_order": "as supplied; no unverified x/y reinterpretation",
          "target": "multi-label clip presence; not per-frame or spatial localization"}
SCHEMA_HASH = hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_archive(path: Path) -> None:
    size, digest = EXPECTED[path.name]
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Archive size/hash mismatch: {path.name}")


def check_drive_gate(path: Path) -> None:
    status = json.loads(path.read_text())
    if status.get("state") != "complete":
        raise ValueError("Drive upload/verification is not complete; training remains gated")
    records = {row["name"]: row for row in status.get("files", [])}
    for name, (size, digest) in EXPECTED.items():
        row = records.get(name, {})
        if not (row.get("verified") is True and row.get("actual_bytes") == size
                and row.get("actual_sha256") == digest):
            raise ValueError(f"Drive archive verification missing: {name}")


def _time(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Timestamp must be ISO8601 or null")
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # Epoch-offset anonymized timestamps; interpret naive values as UTC deterministically.
    if date.tzinfo is None:
        from datetime import timezone
        date = date.replace(tzinfo=timezone.utc)
    return date.timestamp()


def make_sample(content: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = content["labels"]["action"]
    if not isinstance(actions, list):
        raise ValueError("Missing/unknown action annotations must not become negative labels")
    y = np.zeros(3, dtype=np.float32)
    for action in actions:
        if not isinstance(action, dict) or action.get("label") not in CLASSES:
            raise ValueError("Unknown action label")
        if isinstance(action["start"], bool) or isinstance(action["end"], bool):
            raise ValueError("Boolean action endpoints are invalid")
        start, end = float(action["start"]), float(action["end"])
        if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end <= 1):
            raise ValueError("Invalid annotated interval")
        y[CLASSES.index(action["label"])] = 1
    x = np.zeros((2, FEATURES, MAX_STEPS), dtype=np.float32)
    valid = np.zeros((2, MAX_STEPS), dtype=bool)
    for view, rank in enumerate(("rank0", "rank1")):
        camera = content["action_cam"][rank]
        if not isinstance(camera, dict):
            raise ValueError("Camera metadata must be an object")
        poses, stamps = camera.get("poses"), camera.get("frame_timestamps")
        if poses is None:
            poses = [None] * len(stamps) if isinstance(stamps, list) else []
        if isinstance(poses, list) and not poses:
            if stamps is not None and (not isinstance(stamps, list) or stamps):
                raise ValueError("Pose/timestamp count mismatch")
            continue
        if not isinstance(poses, list) or not 1 <= len(poses) <= MAX_STEPS:
            raise ValueError(f"Unsupported pose observations: {rank}, {type(poses).__name__}, count={len(poses) if isinstance(poses, list) else None}")
        if stamps is None:
            stamps = [None] * len(poses)
        if not isinstance(stamps, list) or len(stamps) != len(poses):
            raise ValueError("Pose/timestamp count mismatch")
        times = [_time(t) for t in stamps]
        known = [t for t in times if t is not None]
        if not all(b > a for a, b in zip(known, known[1:])):
            raise ValueError("Non-increasing observed timestamps")
        first = times[0]
        for step, pose in enumerate(poses):
            if pose is not None and not isinstance(pose, dict):
                raise ValueError("Invalid pose observation")
            coordinates = (pose or {}).get("pose")
            scores = (pose or {}).get("joint_scores")
            if coordinates is None: coordinates = {}
            if scores is None: scores = {}
            if not isinstance(coordinates, dict) or not isinstance(scores, dict):
                raise ValueError("Invalid joint maps")
            for joint, name in enumerate(JOINTS):
                point, score = coordinates.get(name), scores.get(name)
                if point is None or score is None:
                    continue
                if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) for v in [*point, score]):
                    raise ValueError("Coordinates must be a numeric pair with numeric confidence")
                values = np.asarray([*point, score], dtype=np.float64)
                if values.shape != (3,):
                    raise ValueError("Invalid coordinate/score shape")
                # Off-image or nonfinite estimates are unobserved, never clipped into validity.
                if not np.isfinite(values).all() or not ((values >= 0).all() and (values <= 1).all()):
                    continue
                x[view, 4*joint:4*joint+4, step] = [*values, 1]
                valid[view, step] = True
            if step and times[step] is not None and times[step-1] is not None:
                x[view, 96:98, step] = [min(times[step]-times[step-1], 60)/60, 1]
            if first is not None and times[step] is not None:
                x[view, 98:100, step] = [min(times[step]-first, 60)/60, 1]
    return x, valid, y


class QualityGateError(ValueError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def load_archive(path: Path, split: str, limit: int | None = None, *, allow_final_test=False):
    """Only train/validation accepted here. Test data requires a separate frozen evaluation."""
    if split not in ("train", "validation", "test") or (split == "test" and not allow_final_test) or path.name != split + ".tar":
        raise ValueError("This loader intentionally excludes the final test split")
    examples, ids, excluded, quarantined = [], [], [], []
    pairs: dict[str, set[str]] = {}
    source_positive = np.zeros(3, dtype=np.int64)
    kept_positive = np.zeros(3, dtype=np.int64)
    selected: dict[str, tuple] = {}
    usable_labels: dict[str, np.ndarray] = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            p = PurePosixPath(member.name)
            if p.is_absolute() or ".." in p.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe TAR member")
            if not member.isfile() or len(p.parts) != 3 or p.parts[0] != split:
                continue
            sid, name = p.parts[1:]
            files = pairs.setdefault(sid, set())
            if name in files:
                raise ValueError("Duplicate member within sample")
            files.add(name)
            if name == "metadata.json":
                if member.size > 10 * 1024 * 1024:
                    raise ValueError("Oversized metadata")
                try:
                    content = json.load(archive.extractfile(member))["content"]
                    if not isinstance(content, dict):
                        raise ValueError("Invalid content object")
                    annotations = content.get("labels")
                    if not isinstance(annotations, dict):
                        raise ValueError("Missing/invalid label object")
                    actions = annotations.get("action")
                    if isinstance(actions, list):
                        present = {a.get("label") for a in actions if isinstance(a, dict)
                                   and isinstance(a.get("label"), str) and a["label"] in CLASSES}
                        source_positive += np.array([name in present for name in CLASSES])
                    value = make_sample(content)
                    if not value[1].any():
                        excluded.append(sid)
                        continue
                    usable_labels[sid] = value[2]
                    selected[sid] = value
                    # Audit every record, but retain only the lexicographically first pilot IDs.
                    if limit is not None and len(selected) > limit:
                        del selected[max(selected)]
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    quarantined.append({"sample_id": sid, "reason": str(exc)})
    required = {"rank0_video.mp4", "rank1_video.mp4", "metadata.json"}
    for sid in sorted(pairs):
        if not required.issubset(pairs[sid]):
            raise ValueError(f"Missing paired media/metadata: {sid}")
        if sid in usable_labels:
            kept_positive += usable_labels[sid].astype(np.int64)
        if sid in selected:
            examples.append(selected[sid]); ids.append(sid)
    report = {"split":split, "archive":path.name, "source_samples":len(pairs), "no_valid_pose_excluded":excluded,
              "quarantined_metadata":quarantined, "quarantine_limit_fraction":0.01,
              "source_positive_clips":dict(zip(CLASSES,source_positive.tolist())),
              "usable_positive_clips":dict(zip(CLASSES,kept_positive.tolist())),
              "maximum_class_removal_fraction":0.10, "used_samples":len(examples),
              "schema":SCHEMA,"schema_hash":SCHEMA_HASH}
    if not examples:
        raise QualityGateError("No usable examples", report)
    if (len(quarantined)+len(excluded))/len(pairs) > 0.01:
        raise QualityGateError(f"Quality gate: over 1% unusable clips ({len(quarantined)} invalid, {len(excluded)} no pose)", report)
    removed_fraction = (source_positive-kept_positive)/np.maximum(source_positive, 1)
    if (removed_fraction > 0.10).any():
        raise QualityGateError("Quality gate: over 10% of a class removed; review rare-class exclusions", report)
    xs, masks, ys = (np.stack(v) for v in zip(*examples))
    return xs, masks, ys, ids, report
