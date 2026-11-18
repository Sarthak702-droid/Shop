"""Research insights: offline model evidence joined with what this host actually measured.

Everything returned is either a committed, reproducible result file under ``ml/results/`` (with the
script that produced it) or an aggregate of the local database. Nothing here is illustrative.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from shop_api.deps import get_context, get_session
from shop_api.service import ACTION_PROFILE, AppContext
from shop_core.config import PROJECT_ROOT
from shop_core.persistence.models import EventRow, JobRow, UploadRow
from shop_core.persistence.repository import iso

router = APIRouter(tags=["research"])
RESULTS = PROJECT_ROOT / "ml" / "results"
CLASSES = ("reach_to_shelf", "retract_from_shelf", "hand_in_shelf", "inspect_product", "inspect_shelf")


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def wilson(successes: int, n: int, z: float = 1.96) -> dict[str, Any]:
    """Point estimate plus Wilson 95% interval; honest for the small samples a reviewer produces."""
    if n == 0:
        return {"estimate": None, "low": None, "high": None, "n": 0}
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {"estimate": round(p, 4), "low": round(max(0.0, centre - half), 4), "high": round(min(1.0, centre + half), 4), "n": n}


def merl_section() -> dict[str, Any]:
    analysis = _read(RESULTS / "merl_tcn_v1" / "analysis.json")
    manifest = _read(PROJECT_ROOT / "ml" / "manifests" / "merl_action_tcn_dev.json") or {}
    if analysis is None:
        return {"available": False, "reason": "Run ml/merl/analyze.py to produce ml/results/merl_tcn_v1/analysis.json."}
    keep = ("run", "checkpoint_sha256", "split", "steps", "training_curve", "best_epoch", "class_stats", "validation", "test",
            "runtime_policy_selection", "confusion_test", "visibility_test", "failures_test", "notes")
    out = {k: analysis.get(k) for k in keep}
    out["reliability_test"] = {c: {"ece": v.get("ece"), "bins": v.get("bins")} for c, v in (analysis.get("reliability_test") or {}).items()}
    out["threshold_sweep_validation"] = analysis.get("threshold_sweep_validation")
    pv = [r for r in analysis.get("per_video_test") or [] if r.get("map") is not None]
    out["per_video_test"] = pv
    out["baseline_no_context"] = (manifest.get("evaluation") or {}).get("baseline_no_context")
    out["model"] = {"bundle_id": manifest.get("bundle_id"), "run": manifest.get("run"), "architecture": manifest.get("architecture"),
                    "constructor": manifest.get("constructor"), "receptive_field_samples": manifest.get("receptive_field_samples"),
                    "feature_version": manifest.get("feature_version"), "commercial_approval": manifest.get("commercial_approval"),
                    "license": manifest.get("license")}
    out["available"] = True
    return out


def transfer_section() -> dict[str, Any] | None:
    """MERL model on RetailAction ceiling cameras (``ml/retailaction/merl_transfer.py``), rows trimmed."""
    data = _read(RESULTS / "retailaction" / "merl_transfer.json")
    if data is None:
        return None
    rows = data.get("rows") or []
    worst_misses = sorted((r for r in rows if r["condition"] == "as_uploaded" and r["group"] != "none" and not r["episode_with_interaction"]),
                          key=lambda r: r["interaction_score"])[:6]
    worst_false = sorted((r for r in rows if r["condition"] == "as_uploaded" and r["group"] == "none" and r["episode_with_interaction"]),
                         key=lambda r: -r["interaction_score"])[:6]
    keep = ("sample_id", "group", "view", "interaction_score", "person_visible", "steps", "duration_s", "max_by_class")
    return {k: v for k, v in data.items() if k != "rows"} | {
        "examples": {"missed": [{k: r[k] for k in keep} for r in worst_misses], "false_alarms": [{k: r[k] for k in keep} for r in worst_false]},
        "findings": data.get("findings"),
    }


def app_section(session: Session) -> dict[str, Any]:
    jobs = session.execute(
        select(JobRow).where(JobRow.analysis_profile == ACTION_PROFILE).order_by(JobRow.created_at.desc())
    ).scalars().all()
    uploads = {u.id: u for u in session.execute(select(UploadRow)).scalars().all()}
    job_rows = []
    states = Counter()
    warning_counts: Counter[str] = Counter()
    analysed_s = 0.0
    for job in jobs:
        states[job.state] += 1
        act = (job.coverage or {}).get("action_analysis") or {}
        summary = act.get("summary") or {}
        timing = act.get("timing_ms") or {}
        metrics = job.metrics or {}
        for w in act.get("warnings") or []:
            warning_counts[w.get("code", "unknown")] += 1
        analysed_s += float(summary.get("analysed_seconds") or 0.0)
        upload = uploads.get(job.upload_id or "")
        job_rows.append({
            "job_id": job.id,
            "status": job.state,
            "filename": upload.original_filename if upload else None,
            "created_at": iso(job.created_at),
            "analysed_seconds": summary.get("analysed_seconds"),
            "episodes": summary.get("episodes"),
            "person_visible_fraction": summary.get("person_visible_fraction"),
            "per_class": summary.get("per_class"),
            "warnings": [w.get("code") for w in act.get("warnings") or []],
            "real_time_factor": metrics.get("real_time_factor"),
            "wall_time_ms": metrics.get("wall_time_ms"),
            "peak_rss_kb": metrics.get("peak_rss_kb"),
            "steps_per_second": act.get("steps_per_second"),
            "timing_ms": timing,
            "input": act.get("input"),
            "error": {"code": job.error_code, "message": job.error_message} if job.error_code else None,
        })

    events = session.execute(
        select(EventRow).where(EventRow.event_type == "shelf_interaction_observed", EventRow.deleted_at.is_(None))
        .options(selectinload(EventRow.reviews))
    ).scalars().all()
    predicted = Counter()
    tp = Counter()
    fp = Counter()
    missed = Counter()
    verdicts = Counter()
    decisions = Counter()
    vis_bins = Counter()
    event_warnings: Counter[str] = Counter()
    reviewed_with_feedback = 0
    duration_ms = []
    for ev in events:
        labels = {a.get("label") for a in ev.actions or []}
        for lab in labels:
            predicted[lab] += 1
        for w in (ev.quality or {}).get("warnings") or []:
            event_warnings[w] += 1
        v = (ev.quality or {}).get("person_visible_fraction")
        if isinstance(v, (int, float)):
            vis_bins[min(4, int(v * 5))] += 1
        duration_ms.append(ev.end_pts_ms - ev.start_pts_ms)
        if ev.review_state != "unreviewed":
            decisions[ev.review_state] += 1
        latest = max(ev.reviews, key=lambda r: r.created_at, default=None) if ev.reviews else None
        fb = latest.action_feedback if latest is not None else None
        if not fb:
            continue
        reviewed_with_feedback += 1
        verdicts[fb.get("verdict", "unclear")] += 1
        if fb.get("verdict") == "unclear":
            continue
        observed = set(fb.get("observed_actions") or [])
        for lab in labels:
            (tp if lab in observed else fp)[lab] += 1
        for lab in observed - labels:
            missed[lab] += 1
    per_class = {}
    for lab in CLASSES:
        judged = tp[lab] + fp[lab]
        per_class[lab] = {
            "predicted_events": predicted[lab],
            "confirmed": tp[lab],
            "rejected": fp[lab],
            "reviewer_added": missed[lab],
            "precision": wilson(tp[lab], judged),
        }
    total_judged = sum(tp.values()) + sum(fp.values())
    return {
        "jobs": job_rows,
        "job_states": dict(states),
        "analysed_minutes": round(analysed_s / 60.0, 2),
        "job_warnings": dict(warning_counts),
        "events": {
            "total": len(events),
            "reviewed": sum(decisions.values()),
            "with_action_feedback": reviewed_with_feedback,
            "decisions": dict(decisions),
            "verdicts": dict(verdicts),
            "warnings": dict(event_warnings),
            "person_visibility_histogram": [{"bin": f"{i * 20}-{i * 20 + 20}%", "events": vis_bins[i]} for i in range(5)],
            "median_duration_ms": sorted(duration_ms)[len(duration_ms) // 2] if duration_ms else None,
        },
        "reviewer_precision": {"per_class": per_class, "overall": wilson(sum(tp.values()), total_judged)},
        "label": "actual_measured_result",
    }


@router.get("/research/insights")
def research_insights(session: Session = Depends(get_session), context: AppContext = Depends(get_context)) -> dict:
    temporal = context.models()["temporal"]
    own_eval = _read(RESULTS / "own_eval" / "summary.json")
    benchmark = _read(RESULTS / "benchmarks" / "cpu_e2e.json")
    runtime_eval = _read(RESULTS / "merl_tcn_v1" / "runtime_e2e.json")
    return {
        "merl": merl_section(),
        "retailaction": _read(RESULTS / "retailaction" / "status.json") or {"available": False},
        "pose_compatibility": _read(RESULTS / "retailaction" / "pose_compatibility.json"),
        "own_evaluation": own_eval or {
            "available": False,
            "reason": "No staged own recordings have been annotated yet. Follow docs/own-evaluation.md.",
        },
        "cpu_benchmark": benchmark,
        "runtime_e2e": runtime_eval,
        "merl_transfer": transfer_section(),
        "app": app_section(session),
        "action_model": {"available": temporal.available, "model_id": temporal.model_id, "version": temporal.version,
                         "reason": temporal.reason, "commercial_status": temporal.commercial_status},
        "label": "actual_measured_result",
    }
