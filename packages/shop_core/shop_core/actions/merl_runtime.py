"""Research-mode shelf-action inference with the MERL causal TCN.

Pipeline for one uploaded file::

    video -> merl_features_v1 (per 5 Hz step) -> ActionTCN -> per-class scores
          -> per-class segments (validation thresholds) -> interaction episodes -> review events

Scores are *uncalibrated observable-action scores* for five MERL shelf interactions. They are never
theft, intent or guilt scores, and the event ``review_score`` stays ``null``. The model was trained
on one overhead camera in one mock store; every result carries that domain warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shop_core.actions import merl_features as mf
from shop_core.config import Settings
from shop_core.registry import ModelUnavailable, inspect_model, load_manifest
from shop_core.temporal.tcn import ActionTCN, Checkpoint

PROFILE = "merl_shelf_actions_research_v1"
EVENT_TYPE = "shelf_interaction_observed"
POLICY_VERSION = "merl_action_episodes_v1"
SCORE_SEMANTICS = "uncalibrated_action_score"
ACTION_TEXT = {
    "reach_to_shelf": "Reach to shelf",
    "retract_from_shelf": "Retract from shelf",
    "hand_in_shelf": "Hand in shelf",
    "inspect_product": "Inspect product",
    "inspect_shelf": "Inspect shelf",
}


@dataclass(frozen=True)
class EpisodePolicy:
    """Runtime post-processing. Evaluated on MERL validation/test by ``ml/merl/runtime_eval.py``."""

    min_segment_steps: int = 2  # drop single-step (0.2 s) blips
    merge_gap_steps: int = 1  # bridge one-step dropouts inside a class run
    episode_gap_ms: int = 1000  # actions closer than this form one interaction episode
    max_episode_ms: int = 20000  # split very long runs so evidence stays reviewable
    context_ms: int = 3000  # evidence context before and after an episode
    low_visibility_fraction: float = 0.4


@dataclass
class ActionSegment:
    label: str
    start_step: int
    end_step: int  # inclusive
    start_ms: int
    end_ms: int
    peak_score: float
    mean_score: float
    threshold: float

    def to_public(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "display": ACTION_TEXT.get(self.label, self.label),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "steps": self.end_step - self.start_step + 1,
            "peak_score": round(self.peak_score, 4),
            "mean_score": round(self.mean_score, 4),
            "threshold": round(self.threshold, 4),
            "score_semantics": SCORE_SEMANTICS,
        }


@dataclass
class Episode:
    start_step: int
    end_step: int
    start_ms: int
    end_ms: int
    actions: list[ActionSegment]
    person_visible_fraction: float
    mean_person_score: float
    warnings: list[str] = field(default_factory=list)

    @property
    def primary(self) -> ActionSegment:
        # Strongest evidence: mean score weighted by duration, ties broken by peak.
        return max(self.actions, key=lambda a: (a.mean_score * (a.end_step - a.start_step + 1), a.peak_score))


class ShelfActionModel:
    """Hash-verified MERL TCN checkpoint plus its validation thresholds."""

    def __init__(self, checkpoint: Checkpoint, model: ActionTCN, *, model_id: str, run: str, sha256: str, manifest: dict) -> None:
        self.checkpoint = checkpoint
        self.model = model
        self.model_id = model_id
        self.run = run
        self.sha256 = sha256
        self.manifest = manifest
        self.classes = tuple(checkpoint.config.classes)
        self.thresholds = np.asarray(checkpoint.thresholds, dtype=np.float32)
        self.mean = checkpoint.feature_mean.numpy().astype(np.float32)
        self.std = checkpoint.feature_std.numpy().astype(np.float32)

    @property
    def version_label(self) -> str:
        return f"{self.model_id}:{self.run}"

    @property
    def receptive_field_steps(self) -> int:
        return self.checkpoint.config.receptive_field

    @classmethod
    def load(cls, settings: Settings) -> "ShelfActionModel":
        status = inspect_model(settings, "temporal")
        if not status.available:
            raise ModelUnavailable(status.details.get("code", "MODEL_UNAVAILABLE"), status.reason, {"model_id": status.model_id})
        manifest = load_manifest(settings.model_manifest_path(settings.models.temporal))
        if manifest.get("feature_version") != mf.SCHEMA:
            raise ModelUnavailable("MODEL_SCHEMA_MISMATCH", "Temporal manifest does not use merl_features_v1.")
        path = settings.resolve_path(manifest["weights"]["path"])
        try:
            checkpoint = Checkpoint.load(path)
        except Exception as error:  # noqa: BLE001 - corrupt checkpoints fail closed
            raise ModelUnavailable("MODEL_LOAD_FAILED", "The shelf-action checkpoint could not be loaded.") from error
        cfg = checkpoint.config
        if cfg.in_channels != mf.FEATURE_DIM or list(cfg.classes) != list(manifest.get("class_order") or []):
            raise ModelUnavailable("MODEL_SCHEMA_MISMATCH", "Checkpoint channels or classes differ from the manifest.")
        if len(checkpoint.thresholds) != len(cfg.classes):
            raise ModelUnavailable("MODEL_SCHEMA_MISMATCH", "Checkpoint has no per-class validation thresholds.")
        torch.set_num_threads(settings.runtime.inference_threads)
        model = checkpoint.build()
        return cls(
            checkpoint,
            model,
            model_id=str(manifest.get("bundle_id", "merl_action_tcn")),
            run=str(manifest.get("run", "unknown")),
            sha256=str(status.sha256),
            manifest=manifest,
        )

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """(T, C) features -> (T, K) sigmoid scores. Causal: step t only sees steps <= t."""
        if features.ndim != 2 or features.shape[1] != mf.FEATURE_DIM:
            raise ValueError("feature matrix does not match merl_features_v1")
        x = (features.astype(np.float32) - self.mean) / self.std
        logits = self.model(torch.from_numpy(np.ascontiguousarray(x.T))[None])
        return torch.sigmoid(logits)[0].numpy().T.astype(np.float32)


def _runs(binary: np.ndarray) -> list[tuple[int, int]]:
    b = np.asarray(binary).astype(np.int8).ravel()
    if b.size == 0:
        return []
    edges = np.diff(np.r_[0, b, 0])
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist()))


def class_segments(scores: np.ndarray, threshold: float, policy: EpisodePolicy) -> list[tuple[int, int]]:
    """Thresholded runs with short gaps bridged and blips removed (inclusive step indices)."""
    runs = _runs(scores >= threshold)
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= policy.merge_gap_steps:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s + 1 >= policy.min_segment_steps]


def build_episodes(
    probs: np.ndarray,
    thresholds: np.ndarray,
    classes: tuple[str, ...],
    pts_ms: np.ndarray,
    person_present: np.ndarray,
    person_score: np.ndarray,
    policy: EpisodePolicy,
    step_ms: float = 200.0,
) -> list[Episode]:
    segments: list[ActionSegment] = []
    total = len(probs)
    for k, label in enumerate(classes):
        for s, e in class_segments(probs[:, k], float(thresholds[k]), policy):
            window = probs[s : e + 1, k]
            end_ms = int(pts_ms[e + 1]) if e + 1 < total else int(pts_ms[e] + step_ms)
            segments.append(ActionSegment(label, s, e, int(pts_ms[s]), end_ms, float(window.max()), float(window.mean()), float(thresholds[k])))
    segments.sort(key=lambda a: (a.start_step, a.end_step))
    episodes: list[Episode] = []
    gap_steps = max(0, int(round(policy.episode_gap_ms / step_ms)))
    max_steps = max(1, int(round(policy.max_episode_ms / step_ms)))
    group: list[ActionSegment] = []
    g_end = -1
    for seg in segments:
        if group and (seg.start_step - g_end - 1 > gap_steps or seg.end_step - group[0].start_step + 1 > max_steps):
            episodes.append(_episode(group, pts_ms, person_present, person_score, policy, step_ms))
            group, g_end = [], -1
        group.append(seg)
        g_end = max(g_end, seg.end_step)
    if group:
        episodes.append(_episode(group, pts_ms, person_present, person_score, policy, step_ms))
    return episodes


def _episode(group: list[ActionSegment], pts_ms: np.ndarray, present: np.ndarray, score: np.ndarray, policy: EpisodePolicy, step_ms: float) -> Episode:
    s = min(a.start_step for a in group)
    e = max(a.end_step for a in group)
    end_ms = max(a.end_ms for a in group)
    visible = float(present[s : e + 1].mean()) if e >= s else 0.0
    scores = score[s : e + 1][present[s : e + 1]]
    warnings = []
    if visible < policy.low_visibility_fraction:
        warnings.append("insufficient_person_visibility")
    if s * step_ms < 12000:
        warnings.append("limited_temporal_context")
    return Episode(s, e, int(pts_ms[s]), end_ms, group, visible, float(scores.mean()) if len(scores) else 0.0, warnings)


def input_warnings(feats: mf.VideoFeatures, receptive_field_steps: int) -> list[dict[str, str]]:
    """Job-level warnings about how far this input is from the training domain."""
    out = [{
        "code": "research_model_domain",
        "severity": "info",
        "message": "Trained on one overhead camera in one mock store (MERL Shopping). Accuracy on other angles, stores and lighting is unmeasured.",
    }]
    if abs(feats.fps - mf.TRAIN_FPS) > 0.5:
        out.append({
            "code": "frame_rate_differs",
            "severity": "warning",
            "message": f"Video is {feats.fps:.2f} fps; the model was trained at 30 fps. Motion features use {feats.step_frames} frames per step instead of 6.",
        })
    tw, th = mf.TRAIN_RESOLUTION
    if feats.width and feats.height and abs(feats.width / feats.height - tw / th) > 0.25:
        out.append({
            "code": "aspect_ratio_differs",
            "severity": "warning",
            "message": f"Frame is {feats.width}x{feats.height}; training video was {tw}x{th}. Spatial features may not transfer.",
        })
    visible = float(feats.person_present.mean()) if len(feats.features) else 0.0
    if visible < 0.3:
        out.append({
            "code": "insufficient_visibility",
            "severity": "warning",
            "message": f"A person was detected in only {visible:.0%} of steps. Action scores are unreliable when nobody is visible.",
        })
    if len(feats.features) < receptive_field_steps:
        out.append({
            "code": "short_video",
            "severity": "warning",
            "message": f"Video is shorter than the model's {receptive_field_steps / mf.RATE_HZ:.0f} s temporal context.",
        })
    return out


def timeline_payload(model: ShelfActionModel, feats: mf.VideoFeatures, probs: np.ndarray, episodes: list[Episode], policy: EpisodePolicy) -> dict[str, Any]:
    """Compact per-step score timeline for the UI (scores rounded to 3 decimals)."""
    return {
        "schema": "shop_action_timeline_v1",
        "model": {"model_id": model.model_id, "run": model.run, "sha256": model.sha256, "feature_schema": mf.SCHEMA},
        "classes": list(model.classes),
        "display": {c: ACTION_TEXT.get(c, c) for c in model.classes},
        "thresholds": [round(float(t), 4) for t in model.thresholds],
        "step_ms": feats.step_ms,
        "pts_ms": feats.pts_ms.astype(int).tolist(),
        "scores": np.round(probs, 3).T.tolist(),  # K lists of T
        "person_present": feats.person_present.astype(int).tolist(),
        "episodes": [
            {"start_ms": ep.start_ms, "end_ms": ep.end_ms, "primary": ep.primary.label, "actions": [a.to_public() for a in ep.actions]}
            for ep in episodes
        ],
        "policy": {**policy.__dict__, "version": POLICY_VERSION},
        "score_semantics": SCORE_SEMANTICS,
    }


def summarize(model: ShelfActionModel, feats: mf.VideoFeatures, probs: np.ndarray, episodes: list[Episode]) -> dict[str, Any]:
    duration_s = max(1e-6, len(probs) / mf.RATE_HZ)
    per_class = {}
    for k, label in enumerate(model.classes):
        segs = [a for ep in episodes for a in ep.actions if a.label == label]
        active = probs[:, k] >= model.thresholds[k]
        per_class[label] = {
            "segments": len(segs),
            "active_seconds": round(float(active.sum()) / mf.RATE_HZ, 1),
            "active_fraction": round(float(active.mean()), 4),
            "mean_score": round(float(probs[:, k].mean()), 4),
            "max_score": round(float(probs[:, k].max()), 4),
            "per_minute": round(len(segs) / (duration_s / 60.0), 2),
        }
    return {
        "steps": int(len(probs)),
        "analysed_seconds": round(duration_s, 1),
        "episodes": len(episodes),
        "person_visible_fraction": round(float(feats.person_present.mean()), 4),
        "per_class": per_class,
        "episodes_with_low_visibility": sum(1 for ep in episodes if "insufficient_person_visibility" in ep.warnings),
    }


def checkpoint_path(settings: Settings) -> Path:
    manifest = load_manifest(settings.model_manifest_path(settings.models.temporal))
    return settings.resolve_path(manifest["weights"]["path"])
