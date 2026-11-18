"""File analysis for the ``merl_shelf_actions_research_v1`` profile.

    upload -> merl_features_v1 (one decode pass) -> MERL TCN -> episodes -> events + evidence clips

Runs in the worker, never in an HTTP handler. Every failure becomes an explicit job error code;
nothing is reported as "no actions" when analysis did not actually complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from shop_core.actions import merl_features as mf
from shop_core.actions.merl_runtime import (
    EVENT_TYPE,
    POLICY_VERSION,
    PROFILE,
    SCORE_SEMANTICS,
    EpisodePolicy,
    ShelfActionModel,
    build_episodes,
    input_warnings,
    summarize,
    timeline_payload,
)
from shop_core.config import Settings
from shop_core.contracts import AnalysisMetrics, Detection, EventRecord, EvidenceStatus
from shop_core.evidence.writer import EvidenceError, EvidenceWriter
from shop_core.pipeline.runner import (
    PipelineSink,
    RunResult,
    SampleRecord,
    _finalize_metrics,
    now_iso,
    read_window_frames,
)
from shop_core.zones import apply_privacy_mask


def model_bundle_label(model: ShelfActionModel, extractor: mf.Extractor) -> str:
    return f"{model.model_id}:{model.run}|{extractor.model_ref}|policy:{POLICY_VERSION}"[:100]


def timeline_path(settings: Settings, job_id: str) -> Path:
    return settings.storage.root / "timelines" / f"{job_id}.json"


def _write_json_atomic(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def _samples(feats: mf.VideoFeatures, extractor: mf.Extractor) -> list[SampleRecord]:
    model_id, _, version = extractor.model_ref.partition(":")
    out = []
    for i, pts in enumerate(feats.pts_ms.tolist()):
        box = feats.person_box[i]
        dets = []
        if box[0] == box[0] and box[2] > box[0] and box[3] > box[1]:  # not NaN, positive area
            dets.append(Detection(tuple(float(v) for v in box), float(min(1.0, max(0.0, feats.person_score[i]))), 1, "person", model_id, version or "unknown"))
        out.append(SampleRecord(pts_ms=int(pts), detections=dets, tracks=[]))
    return out


def run_action_analysis(
    *,
    settings: Settings,
    extractor: mf.Extractor,
    model: ShelfActionModel,
    path: Path,
    duration_ms: int | None,
    job_id: str,
    source_id: str,
    session_id: str,
    sink: PipelineSink,
    evidence_writer: EvidenceWriter | None = None,
    generate_evidence: bool = True,
    policy: EpisodePolicy | None = None,
) -> RunResult:
    policy = policy or EpisodePolicy()
    started_wall = time.perf_counter()
    metrics = AnalysisMetrics(started_at_utc=now_iso(), media_duration_ms=duration_ms, inference_threads=settings.runtime.inference_threads)
    writer = evidence_writer or EvidenceWriter(settings.evidence_dir)
    bundle = model_bundle_label(model, extractor)
    privacy = settings.zones.privacy_mask
    last_progress = [0.0]
    last_heartbeat = [0.0]
    expected = int(round(duration_ms / 1000.0 * mf.RATE_HZ)) if duration_ms else None

    def on_step(done: int, _frame: int, pts_ms: int) -> bool:
        now = time.perf_counter()
        if now - last_progress[0] >= 1.0:
            sink.report_progress({
                "stage": "extracting_features",
                "media_pts_ms": pts_ms,
                "duration_ms": duration_ms,
                "analysed_frames": done,
                "dropped_frames": 0,
                "expected_samples": expected,
                "analysis_coverage": round(done / expected, 4) if expected else None,
                "events_created": 0,
                "analysis_profile": PROFILE,
            })
            last_progress[0] = now
        if now - last_heartbeat[0] >= 5.0:
            sink.heartbeat()
            last_heartbeat[0] = now
        return sink.should_cancel()

    def fail(code: str, message: str, notes: list[str] | None = None) -> RunResult:
        _finalize_metrics(metrics, started_wall=started_wall, media_span_ms=None)
        return RunResult("failed", code, message, metrics, {"analysis_profile": PROFILE}, 0, 0, notes or [])

    transform = (lambda img: apply_privacy_mask(img, privacy)) if privacy else None
    try:
        feats = mf.extract_video(extractor, path, frame_transform=transform, on_step=on_step)
    except mf.FeatureError as error:
        return fail(error.code, error.message)
    except Exception as error:  # noqa: BLE001 - recorded, never hidden
        return fail("FEATURE_EXTRACTION_FAILED", "Shelf-action feature extraction failed.", [f"failure_type={type(error).__name__}"])

    steps = len(feats.features)
    metrics.analyzed_frames = steps
    metrics.expected_samples = expected
    metrics.person_detections = int(feats.person_present.sum())
    metrics.detect_ms = [feats.timings_ms["detect"] / steps] if steps else []
    if sink.should_cancel():
        _finalize_metrics(metrics, started_wall=started_wall, media_span_ms=int(feats.pts_ms[-1]) if steps else None)
        return RunResult("cancelled", None, None, metrics, {"analysis_profile": PROFILE, "analyzed_samples": steps}, 0, 0,
                         ["Cancelled during feature extraction; no actions were published."])

    sink.report_progress({"stage": "temporal_inference", "media_pts_ms": int(feats.pts_ms[-1]), "duration_ms": duration_ms,
                          "analysed_frames": steps, "dropped_frames": 0, "events_created": 0, "analysis_profile": PROFILE})
    t0 = time.perf_counter()
    try:
        probs = model.predict(feats.features)
    except Exception as error:  # noqa: BLE001
        return fail("TEMPORAL_MODEL_FAILED", "The shelf-action model failed during inference.", [f"failure_type={type(error).__name__}"])
    tcn_ms = (time.perf_counter() - t0) * 1000.0

    episodes = build_episodes(probs, model.thresholds, model.classes, feats.pts_ms, feats.person_present, feats.person_score, policy)
    cap = settings.analysis.max_events_per_job
    notes: list[str] = []
    if len(episodes) > cap:
        notes.append(f"{len(episodes) - cap} episodes beyond max_events_per_job={cap} were not published.")
    warnings = input_warnings(feats, model.receptive_field_steps)
    summary = summarize(model, feats, probs, episodes)
    timeline = timeline_payload(model, feats, probs, episodes, policy)
    timeline_sha = _write_json_atomic(timeline_path(settings, job_id), timeline)

    records: list[EventRecord] = []
    for ep in episodes[:cap]:
        primary = ep.primary
        quality = {
            "policy": POLICY_VERSION,
            "observed_samples": ep.end_step - ep.start_step + 1,
            "observed_duration_ms": ep.end_ms - ep.start_ms,
            "observed_fraction": round(ep.person_visible_fraction, 4),
            "person_visible_fraction": round(ep.person_visible_fraction, 4),
            "mean_person_score": round(ep.mean_person_score, 4),
            "warnings": list(ep.warnings) + [w["code"] for w in warnings if w["severity"] == "warning"],
            "primary_action": primary.label,
            "primary_action_score": round(primary.mean_score, 4),
            "action_score_semantics": SCORE_SEMANTICS,
            "research_mode": True,
            "no_activity_score_available": True,
        }
        record = EventRecord(
            event_id=str(uuid.uuid4()),
            source_id=source_id,
            session_id=session_id,
            job_id=job_id,
            event_type=EVENT_TYPE,
            reasons=[f"action:{a.label}" for a in sorted(ep.actions, key=lambda a: a.start_step)][:16],
            start_pts_ms=ep.start_ms,
            end_pts_ms=max(ep.end_ms, ep.start_ms + 1),
            track_ids=[],
            review_score=None,
            score_semantics=None,
            calibration_version=None,
            quality=quality,
            required_evidence=["clip_with_context"],
            available_evidence=["action_timeline", "person_box"],
            model_bundle=bundle,
            policy_version=POLICY_VERSION,
            evidence=EvidenceStatus(status="pending"),
            created_at_utc=now_iso(),
            actions=[a.to_public() for a in sorted(ep.actions, key=lambda a: a.start_step)],
        )
        records.append(record)
        sink.event_started(record)

    evidence_failures: dict[str, str] = {}
    if not generate_evidence:
        for record in records:
            reason = "EVIDENCE_RETENTION_DISABLED: this job requested no retained evidence media."
            evidence_failures[record.event_id] = reason
            sink.event_evidence_ready(record.event_id, None, reason)
    else:
        samples = _samples(feats, extractor)
        media_end = max(duration_ms or 0, int(feats.pts_ms[-1]) + 200)
        for n, record in enumerate(records, 1):
            if sink.should_cancel():
                reason = "EVIDENCE_CANCELLED: job was cancelled before this clip was generated."
                evidence_failures[record.event_id] = reason
                sink.event_evidence_ready(record.event_id, None, reason)
                continue
            start = max(0, record.start_pts_ms - policy.context_ms)
            end = min(media_end, record.end_pts_ms + policy.context_ms)
            t0 = time.perf_counter()
            try:
                frames = read_window_frames(path, start_ms=start, end_ms=end, sampling_fps=mf.RATE_HZ * 2,
                                            observations=samples, privacy_polygon=privacy)
                clip = writer.write_clip(event_id=record.event_id, frames=frames, start_pts_ms=frames[0].pts_ms,
                                         end_pts_ms=max(frames[0].pts_ms + 1, frames[-1].pts_ms + 100))
                metrics.evidence_ms.append(clip.generate_ms)
                sink.event_evidence_ready(record.event_id, clip, None)
            except EvidenceError as error:
                metrics.evidence_ms.append((time.perf_counter() - t0) * 1000.0)
                evidence_failures[record.event_id] = f"{error.code}: {error.message}"
                sink.event_evidence_ready(record.event_id, None, evidence_failures[record.event_id])
            except Exception:  # noqa: BLE001
                metrics.evidence_ms.append((time.perf_counter() - t0) * 1000.0)
                evidence_failures[record.event_id] = "EVIDENCE_GENERATION_FAILED: clip could not be generated."
                sink.event_evidence_ready(record.event_id, None, evidence_failures[record.event_id])
            sink.report_progress({"stage": "writing_evidence", "media_pts_ms": int(feats.pts_ms[-1]), "duration_ms": duration_ms,
                                  "analysed_frames": steps, "dropped_frames": 0, "events_created": len(records),
                                  "evidence_written": n, "analysis_profile": PROFILE})

    span = int(feats.pts_ms[-1] - feats.pts_ms[0]) + 200 if steps else None
    _finalize_metrics(metrics, started_wall=started_wall, media_span_ms=span)
    metrics.analysis_coverage = round(min(1.0, steps / expected), 4) if expected else None
    wall_s = max(1e-6, metrics.wall_time_ms / 1000.0)
    action_analysis = {
        "profile": PROFILE,
        "model": {"model_id": model.model_id, "run": model.run, "sha256": model.sha256, "feature_schema": mf.SCHEMA,
                  "detector": extractor.model_ref, "policy": POLICY_VERSION, "receptive_field_steps": model.receptive_field_steps},
        "input": {"fps": round(feats.fps, 3), "frames": feats.frame_count, "width": feats.width, "height": feats.height,
                  "step_frames": feats.step_frames},
        "warnings": warnings,
        "summary": summary,
        "timeline": {"path": f"timelines/{job_id}.json", "sha256": timeline_sha},
        "timing_ms": {
            "decode": feats.timings_ms["decode"], "detect_and_backbone": feats.timings_ms["detect"],
            "motion": feats.timings_ms["motion"], "tcn": round(tcn_ms, 1),
            "evidence": round(sum(metrics.evidence_ms), 1), "wall": metrics.wall_time_ms,
            "per_step_feature_ms": round((feats.timings_ms["detect"] + feats.timings_ms["motion"]) / max(1, steps), 2),
        },
        "steps_per_second": round(steps / wall_s, 2),
        "label": "actual_measured_result",
    }
    coverage = {
        "analysis_coverage": metrics.analysis_coverage,
        "expected_samples": expected,
        "analyzed_samples": steps,
        "missing_samples": max(0, (expected or steps) - steps),
        "decode_gaps": 0,
        "decode_incomplete": bool(expected and steps < expected * 0.98),
        "measured_media_span_ms": span,
        "declared_duration_ms": duration_ms,
        "evidence_generation": {"requested": len(records), "failures": len(evidence_failures), "failure_reasons": evidence_failures},
        "action_analysis": action_analysis,
        "label": "actual_measured_result",
    }
    if coverage["decode_incomplete"]:
        notes.append("Decoder reached EOF before the declared duration; analysis coverage is partial.")
    status = "cancelled" if sink.should_cancel() else "completed"
    sink.report_progress({"stage": "finished", "media_pts_ms": int(feats.pts_ms[-1]), "duration_ms": duration_ms,
                          "analysed_frames": steps, "dropped_frames": 0, "events_created": len(records),
                          "analysis_coverage": metrics.analysis_coverage, "analysis_profile": PROFILE})
    return RunResult(status, None, None, metrics, coverage, len(records), len(evidence_failures), notes)
