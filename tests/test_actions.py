"""Research shelf-action path: episode policy, runtime pipeline, safe failures, API contract."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from shop_core.actions import merl_features as mf
from shop_core.actions.merl_runtime import (
    EVENT_TYPE,
    EpisodePolicy,
    ShelfActionModel,
    build_episodes,
    class_segments,
    input_warnings,
)
from shop_core.config import PROJECT_ROOT
from shop_core.persistence.models import EventRow
from shop_core.pipeline.runner import InMemorySink
from tests.helpers import seed_event

CLASSES = ("reach_to_shelf", "retract_from_shelf", "hand_in_shelf", "inspect_product", "inspect_shelf")
CHECKPOINT = PROJECT_ROOT / "var/models/merl_tcn_v1/model.pt"
needs_checkpoint = pytest.mark.skipif(not CHECKPOINT.is_file(), reason="MERL checkpoint not installed (scripts/models/fetch_merl_tcn.sh)")
ACTION_PROFILE = "merl_shelf_actions_research_v1"


def test_class_segments_bridge_gaps_and_drop_blips() -> None:
    policy = EpisodePolicy(min_segment_steps=2, merge_gap_steps=1)
    s = np.array([0, 0.9, 0, 0, 0.9, 0.9, 0.1, 0.9, 0.9, 0, 0, 0.9])
    # step 1 is a single-step blip (dropped); 4-8 bridge the one-step dip at 6; step 11 is a blip.
    assert class_segments(s, 0.5, policy) == [(4, 8)]
    assert class_segments(s, 0.5, EpisodePolicy(min_segment_steps=1, merge_gap_steps=0)) == [(1, 1), (4, 5), (7, 8), (11, 11)]


def test_episodes_group_nearby_actions_and_pick_primary() -> None:
    t = 100
    probs = np.zeros((t, 5), dtype=np.float32)
    probs[10:14, 0] = 0.9  # reach 2.0-2.8 s
    probs[14:30, 2] = 0.8  # hand in shelf right after
    probs[30:33, 1] = 0.9  # retract
    probs[80:90, 3] = 0.95  # separate inspect episode later
    pts = np.arange(t) * 200
    present = np.ones(t, dtype=bool)
    present[80:90] = False
    episodes = build_episodes(probs, np.full(5, 0.5), CLASSES, pts, present, np.full(t, 0.8, dtype=np.float32), EpisodePolicy())
    assert len(episodes) == 2
    first, second = episodes
    assert [a.label for a in first.actions] == ["reach_to_shelf", "hand_in_shelf", "retract_from_shelf"]
    assert first.primary.label == "hand_in_shelf"
    assert (first.start_ms, first.end_ms) == (2000, 6600)
    assert "limited_temporal_context" in first.warnings
    assert second.primary.label == "inspect_product"
    assert "insufficient_person_visibility" in second.warnings


def test_input_warnings_flag_domain_shift() -> None:
    feats = mf.VideoFeatures(
        features=np.zeros((20, mf.FEATURE_DIM), dtype=np.float16), frame_index=np.arange(20), pts_ms=np.arange(20) * 200,
        person_box=np.full((20, 4), np.nan, dtype=np.float32), person_score=np.zeros(20, dtype=np.float32),
        frame_count=100, fps=25.0, step_frames=5, width=1920, height=1080,
    )
    codes = {w["code"] for w in input_warnings(feats, 61)}
    assert {"research_model_domain", "frame_rate_differs", "aspect_ratio_differs", "insufficient_visibility", "short_video"} <= codes


@needs_checkpoint
def test_runtime_model_loads_with_validation_thresholds(settings) -> None:
    model = ShelfActionModel.load(settings)
    assert model.classes == CLASSES
    assert len(model.thresholds) == 5 and np.all((model.thresholds > 0) & (model.thresholds < 1))
    assert model.receptive_field_steps == 61
    probs = model.predict(np.zeros((30, mf.FEATURE_DIM), dtype=np.float16))
    assert probs.shape == (30, 5) and np.all((probs >= 0) & (probs <= 1))
    # Causality: appending future steps never changes earlier scores.
    x = np.random.default_rng(0).normal(size=(40, mf.FEATURE_DIM)).astype(np.float16)
    assert np.allclose(model.predict(x[:25]), model.predict(x)[:25], atol=1e-5)


@pytest.fixture()
def action_stack(settings):  # noqa: ANN001, ANN201
    from shop_core.detection.person import PersonDetector

    if not CHECKPOINT.is_file():
        pytest.skip("MERL checkpoint not installed")
    return mf.Extractor.from_detector(PersonDetector.load(settings)), ShelfActionModel.load(settings)


def _run(settings, stack, path: Path, sink, **kw):  # noqa: ANN001, ANN202
    from shop_core.pipeline.actions import run_action_analysis

    extractor, model = stack
    return run_action_analysis(settings=settings, extractor=extractor, model=model, path=path, duration_ms=2000,
                               job_id=str(uuid.uuid4()), source_id="s", session_id="x", sink=sink, **kw)


def test_action_pipeline_runs_end_to_end_and_writes_timeline(settings, action_stack, sample_video) -> None:
    settings.ensure_directories()
    sink = InMemorySink()
    result = _run(settings, action_stack, sample_video, sink)
    assert result.status == "completed", result.error_message
    act = result.coverage["action_analysis"]
    assert act["model"]["feature_schema"] == "merl_features_v1"
    assert act["input"]["step_frames"] == 2  # 10 fps fixture -> 2 frames per 5 Hz step
    codes = {w["code"] for w in act["warnings"]}
    assert {"frame_rate_differs", "short_video", "insufficient_visibility"} <= codes
    timeline = json.loads((settings.storage.root / act["timeline"]["path"]).read_text())
    assert timeline["classes"] == list(CLASSES)
    assert len(timeline["scores"]) == 5 and len(timeline["scores"][0]) == len(timeline["pts_ms"]) == act["summary"]["steps"]
    for record in sink.events:
        assert record.event_type == EVENT_TYPE
        assert record.review_score is None
        assert record.actions and all(a["score_semantics"] == "uncalibrated_action_score" for a in record.actions)


def test_action_pipeline_corrupt_video_fails_safely(settings, action_stack, tmp_path) -> None:
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x13" * 4096)
    result = _run(settings, action_stack, bad, InMemorySink())
    assert result.status == "failed"
    assert result.error_code in {"SOURCE_OPEN_FAILED", "SOURCE_FPS_INVALID", "SOURCE_NO_FRAMES"}
    assert result.events_created == 0


def test_action_pipeline_cancellation_publishes_nothing(settings, action_stack, sample_video) -> None:
    sink = InMemorySink()
    sink.cancelled = True
    result = _run(settings, action_stack, sample_video, sink)
    assert result.status == "cancelled"
    assert sink.events == []


def test_missing_checkpoint_is_a_safe_error(settings, tmp_path) -> None:
    from shop_core.registry import ModelUnavailable

    manifest = json.loads((PROJECT_ROOT / "ml/manifests/merl_action_tcn_dev.json").read_text())
    manifest["weights"]["path"] = "var/models/does-not-exist.pt"
    path = PROJECT_ROOT / "var/test-runs/merl-missing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    try:
        settings.models.temporal.manifest = str(path)
        with pytest.raises(ModelUnavailable) as info:
            ShelfActionModel.load(settings)
        assert info.value.code == "MODEL_ARTIFACT_MISSING"
    finally:
        path.unlink(missing_ok=True)


# ---- API contract -------------------------------------------------------------------------------


def _job(client, body):  # noqa: ANN001, ANN202
    return client.post("/api/v1/jobs", json=body)


def test_action_profile_job_contract(client, sample_video_bytes) -> None:
    upload = client.post("/api/v1/uploads", files={"file": ("shelf.avi", sample_video_bytes, "video/x-msvideo")}).json()["upload"]
    body = {"source": {"kind": "file", "upload_id": upload["upload_id"]}, "analysis_profile": ACTION_PROFILE}
    response = _job(client, body)
    if not CHECKPOINT.is_file():
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "ACTION_MODEL_UNAVAILABLE"
        return
    assert response.status_code == 202, response.text
    job = response.json()["job"]
    assert job["analysis_profile"] == ACTION_PROFILE
    assert job["model_bundle"].startswith("merl_action_tcn_v1:")
    assert job["policy_version"] == "merl_action_episodes_v1"
    timeline = client.get(f"/api/v1/jobs/{job['job_id']}/action-timeline")
    assert timeline.status_code == 409
    assert timeline.json()["error"]["code"] == "ACTION_TIMELINE_NOT_READY"


def test_action_profile_rejects_webcam_sources(client, monkeypatch) -> None:
    body = {"source": {"kind": "webcam", "source_id": str(uuid.uuid4())}, "analysis_profile": ACTION_PROFILE}
    response = _job(client, body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "ANALYSIS_PROFILE_SOURCE_UNSUPPORTED"


def test_baseline_job_has_no_action_timeline(app_factory) -> None:
    app, client = app_factory()
    seeded = seed_event(app)
    response = client.get(f"/api/v1/jobs/{seeded['job_id']}/action-timeline")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ACTION_TIMELINE_NOT_APPLICABLE"


def _shelf_event(app) -> str:  # noqa: ANN001
    seeded = seed_event(app, event_type=EVENT_TYPE)
    with app.state.context.db.session() as session:
        row = session.get(EventRow, seeded["event_id"])
        row.actions = [
            {"label": "reach_to_shelf", "start_ms": 0, "end_ms": 800, "mean_score": 0.8},
            {"label": "inspect_product", "start_ms": 800, "end_ms": 3000, "mean_score": 0.7},
        ]
    return seeded["event_id"]


def test_action_feedback_is_recorded_and_drives_insights(app_factory) -> None:
    app, client = app_factory()
    event_id = _shelf_event(app)
    detail = client.get(f"/api/v1/events/{event_id}").json()["event"]
    assert [a["label"] for a in detail["actions"]] == ["reach_to_shelf", "inspect_product"]
    review = {
        "decision": "inconclusive", "reason_codes": ["context_insufficient"], "notes": "",
        "action_feedback": {"verdict": "partially_correct", "observed_actions": ["reach_to_shelf", "retract_from_shelf"]},
    }
    created = client.post(f"/api/v1/events/{event_id}/reviews", json=review)
    assert created.status_code == 201, created.text
    assert created.json()["review"]["action_feedback"]["verdict"] == "partially_correct"
    insights = client.get("/api/v1/research/insights")
    assert insights.status_code == 200
    precision = insights.json()["app"]["reviewer_precision"]["per_class"]
    assert precision["reach_to_shelf"]["confirmed"] == 1
    assert precision["inspect_product"]["rejected"] == 1
    assert precision["retract_from_shelf"]["reviewer_added"] == 1
    assert insights.json()["app"]["reviewer_precision"]["overall"]["n"] == 2


def test_action_feedback_validation(app_factory) -> None:
    app, client = app_factory()
    person = seed_event(app)
    bad_type = client.post(f"/api/v1/events/{person['event_id']}/reviews", json={
        "decision": "benign", "reason_codes": ["activity_benign"], "action_feedback": {"verdict": "correct"}})
    assert bad_type.status_code == 422
    assert bad_type.json()["error"]["code"] == "ACTION_FEEDBACK_NOT_APPLICABLE"
    shelf = _shelf_event(app)
    contradictory = client.post(f"/api/v1/events/{shelf}/reviews", json={
        "decision": "benign", "reason_codes": ["activity_benign"],
        "action_feedback": {"verdict": "no_interaction", "observed_actions": ["inspect_shelf"]}})
    assert contradictory.status_code == 422


def test_event_type_filter(app_factory) -> None:
    app, client = app_factory()
    seed_event(app)
    shelf = _shelf_event(app)
    items = client.get("/api/v1/events?event_type=shelf_interaction_observed").json()["items"]
    assert [i["event_id"] for i in items] == [shelf]
    assert client.get("/api/v1/events?event_type=theft").status_code == 422


def test_research_insights_serves_committed_results(client) -> None:
    payload = client.get("/api/v1/research/insights").json()
    merl = payload["merl"]
    if (PROJECT_ROOT / "ml/results/merl_tcn_v1/analysis.json").is_file():
        assert merl["available"] is True
        assert set(merl["test"]["frame"]) >= set(CLASSES)
        assert merl["runtime_policy_selection"]["selected_on"] == "validation"
    assert payload["app"]["label"] == "actual_measured_result"
    assert payload["retailaction"] is not None
