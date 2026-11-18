"""API integration tests against an isolated SQLite store and real upload probing.

These tests exercise the HTTP contract, error envelope, durable queue, review
history and evidence delivery. They deliberately do not run model inference.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from shop_api.routes import sources as sources_routes
from shop_core.persistence.models import utcnow
from shop_core.persistence.repository import transition_job
from tests.helpers import seed_event

REVIEW_BODY = {
    "decision": "review_needed",
    "reason_codes": ["product_unclear"],
    "notes": "Short descriptive note.",
}


def upload_sample(client, sample_video_bytes: bytes) -> dict:
    response = client.post(
        "/api/v1/uploads",
        files={"file": ("walkthrough.avi", sample_video_bytes, "video/x-msvideo")},
    )
    assert response.status_code == 201, response.text
    return response.json()["upload"]


def create_file_job(client, upload_id: str, *, key: str | None = None, **overrides):
    body = {
        "source": {"kind": "file", "upload_id": upload_id},
        "analysis_profile": "cpu_baseline_v1",
        "sampling_fps": 5.0,
        "privacy_mask_version": "zones_default_v1",
        "retain_evidence": True,
        **overrides,
    }
    headers = {"Idempotency-Key": key} if key else {}
    return client.post("/api/v1/jobs", json=body, headers=headers)


def test_health_live_and_readiness_are_distinct(client) -> None:
    live = client.get("/api/v1/health/live", headers={"X-Request-ID": "req-live.123"})
    assert live.status_code == 200
    assert live.headers["X-Request-ID"] == "req-live.123"
    assert live.json()["request_id"] == "req-live.123"
    assert live.json()["status"] == "ok"

    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 503
    payload = ready.json()
    assert payload["status"] == "degraded"
    assert payload["ready"] is False
    checks = {item["name"]: item for item in payload["checks"]}
    assert checks["database"]["status"] == "pass"
    assert checks["storage"]["status"] == "pass"
    assert checks["person_detector"]["status"] == "pass"
    assert checks["analysis_worker"]["status"] == "fail"
    assert payload["worker"]["status"] == "not_started"


def test_local_bearer_token_protects_privileged_routes(app_factory) -> None:
    _app, client = app_factory(token="local-test-token")
    assert client.get("/api/v1/health/live").status_code == 200
    refused = client.get("/api/v1/capabilities")
    assert refused.status_code == 401
    assert refused.json()["error"]["code"] == "UNAUTHENTICATED"
    assert refused.headers["WWW-Authenticate"] == "Bearer"

    wrong = client.get(
        "/api/v1/capabilities", headers={"Authorization": "Bearer wrong-token"}
    )
    assert wrong.status_code == 401
    assert wrong.json()["error"]["code"] == "INVALID_TOKEN"

    allowed = client.get(
        "/api/v1/capabilities", headers={"Authorization": "Bearer local-test-token"}
    )
    assert allowed.status_code == 200
    assert allowed.json()["service"]["api_version"] == "v1"


def test_error_envelope_and_request_id_are_safe(client) -> None:
    malformed = client.get(
        "/api/v1/events/not-a-uuid", headers={"X-Request-ID": "client-request-42"}
    )
    assert malformed.status_code == 422
    payload = malformed.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["retryable"] is False
    assert payload["request_id"] == "client-request-42"
    assert malformed.headers["X-Request-ID"] == "client-request-42"
    assert payload["error"]["details"]["fields"]
    assert "Traceback" not in malformed.text

    generated = client.get("/api/v1/events", headers={"X-Request-ID": "bad id!"})
    assert generated.headers["X-Request-ID"] != "bad id!"
    assert len(generated.headers["X-Request-ID"]) >= 32

    missing = client.get(f"/api/v1/events/{uuid.uuid4()}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "EVENT_NOT_FOUND"

    unknown = client.get("/api/v1/does-not-exist")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "NOT_FOUND"
    assert unknown.json()["request_id"] == unknown.headers["X-Request-ID"]


def test_capabilities_describe_limits_and_disabled_modules(client, settings) -> None:
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["cpu_verification"]["passed"] is True
    assert payload["cpu_verification"]["label"] == "actual_measured_result"
    assert payload["sources"]["rtsp"]["available"] is False
    assert payload["sources"]["audio"]["available"] is False
    assert payload["models"]["person"]["available"] is True
    assert payload["models"]["person"]["commercial_status"] == "development_only"
    assert payload["models"]["retail"]["available"] is False
    assert payload["models"]["pose"]["available"] is False
    # Research shelf-action TCN: enabled in research mode, available only when its hash-verified
    # checkpoint is installed (scripts/models/fetch_merl_tcn.sh), and always development_only.
    temporal = payload["models"]["temporal"]
    assert temporal["enabled"] is True
    assert temporal["commercial_status"] == "development_only"
    checkpoint = settings.resolve_path("var/models/merl_tcn_v1/model.pt")
    assert temporal["available"] is checkpoint.is_file()
    profiles = {item["profile"]: item for item in payload["limits"]["analysis_profiles"]}
    assert profiles["merl_shelf_actions_research_v1"]["research_mode"] is True
    assert profiles["merl_shelf_actions_research_v1"]["sources"] == ["file"]
    assert payload["models"]["policy"]["no_activity_score_available"] is True
    assert payload["models"]["policy"]["score_semantics"] is None
    assert payload["features"]["automatic_enforcement"] is False
    assert payload["features"]["identity_recognition"] is False
    assert payload["features"]["appearance_matching"] is False
    assert payload["limits"]["label"] == "proposed_starting_setting"
    assert payload["security"]["bearer_token_configured"] is False
    text = json.dumps(payload)
    assert str(settings.storage.root) not in text
    assert "var/" not in text


def test_config_endpoint_redacts_storage_path(client, settings) -> None:
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    payload = response.json()
    assert payload["storage"]["root"] == "managed_local_storage"
    assert payload["has_api_token"] is False
    assert payload["worker"]["status"] == "not_started"
    text = json.dumps(payload)
    assert str(settings.storage.root) not in text
    assert "SHOP_API_TOKEN" not in text


def test_upload_validation_probe_and_deletion(client, settings, sample_video_bytes) -> None:
    unsupported = client.post(
        "/api/v1/uploads", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert unsupported.status_code == 415
    assert unsupported.json()["error"]["code"] == "UNSUPPORTED_EXTENSION"

    empty = client.post(
        "/api/v1/uploads", files={"file": ("empty.avi", b"", "video/x-msvideo")}
    )
    assert empty.status_code == 422
    assert empty.json()["error"]["code"] == "UPLOAD_EMPTY"

    corrupt = client.post(
        "/api/v1/uploads",
        files={"file": ("broken.avi", b"not-a-video-at-all", "video/x-msvideo")},
    )
    assert corrupt.status_code in {415, 422}
    assert corrupt.json()["error"]["code"] in {"MEDIA_UNSUPPORTED", "MEDIA_DECODE_FAILED"}
    assert not list((settings.uploads_dir).glob("*"))

    upload = upload_sample(client, sample_video_bytes)
    assert upload["status"] == "ready"
    assert upload["filename"] == "walkthrough.avi"
    assert upload["bytes"] == len(sample_video_bytes)
    assert upload["probe"]["width"] == 96
    assert upload["probe"]["height"] == 64
    assert upload["probe"]["codec"] == "MJPG"
    assert upload["probe"]["duration_ms"] == 2000
    assert upload["probe"]["first_frame_decoded"] is True
    assert upload["probe"]["container"] == "avi"
    assert len(upload["sha256"]) == 64
    stored = settings.uploads_dir / f"{upload['upload_id']}.avi"
    assert stored.is_file()

    listed = client.get("/api/v1/uploads")
    assert [item["upload_id"] for item in listed.json()["items"]] == [upload["upload_id"]]
    fetched = client.get(f"/api/v1/uploads/{upload['upload_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["upload"]["sha256"] == upload["sha256"]

    deleted = client.delete(f"/api/v1/uploads/{upload['upload_id']}")
    assert deleted.status_code == 202
    assert deleted.json()["status"] == "deleted"
    assert not stored.exists()
    assert client.get(f"/api/v1/uploads/{upload['upload_id']}").status_code == 404
    assert client.get("/api/v1/uploads").json()["items"] == []


def test_source_registration_etag_and_optimistic_patch(client, monkeypatch) -> None:
    refused = client.post("/api/v1/sources", json={"camera_key": "unapproved_camera"})
    assert refused.status_code == 422
    assert refused.json()["error"]["code"] == "CAMERA_NOT_APPROVED"

    created = client.post(
        "/api/v1/sources", json={"camera_key": "camera_01", "label": "Front counter"}
    )
    assert created.status_code == 201
    source = created.json()["source"]
    source_id = source["source_id"]
    assert source["state"] == "idle"
    assert source["approved_camera"] is True
    assert source["camera"]["device_index"] == 0
    first_etag = created.headers["ETag"]
    assert first_etag

    duplicate = client.post("/api/v1/sources", json={"camera_key": "camera_01"})
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "SOURCE_ALREADY_EXISTS"

    listed = client.get("/api/v1/sources").json()
    assert len(listed["items"]) == 1
    approved = {item["key"]: item for item in listed["approved_cameras"]}
    assert approved["camera_01"]["registered"] is True
    assert approved["camera_02"]["registered"] is False

    fetched = client.get(f"/api/v1/sources/{source_id}")
    assert fetched.headers["ETag"] == first_etag

    missing_precondition = client.patch(
        f"/api/v1/sources/{source_id}", json={"label": "New label"}
    )
    assert missing_precondition.status_code == 428
    assert missing_precondition.json()["error"]["code"] == "IF_MATCH_REQUIRED"

    stale = client.patch(
        f"/api/v1/sources/{source_id}",
        json={"label": "New label"},
        headers={"If-Match": '"stale"'},
    )
    assert stale.status_code == 412
    assert stale.json()["error"]["code"] == "STALE_VERSION"

    patched = client.patch(
        f"/api/v1/sources/{source_id}",
        json={"label": "New label"},
        headers={"If-Match": first_etag},
    )
    assert patched.status_code == 200
    assert patched.json()["source"]["label"] == "New label"
    assert patched.headers["ETag"] != first_etag

    monkeypatch.setattr(
        sources_routes.WebcamSource,
        "probe",
        staticmethod(
            lambda device_index: {
                "device_index": device_index,
                "device_present": True,
                "opened": True,
                "frame_read": True,
                "backend": "V4L2",
                "decoded_width": 1280,
                "decoded_height": 720,
                "reason": None,
            }
        ),
    )
    health = client.get(f"/api/v1/sources/{source_id}/health")
    assert health.status_code == 200
    assert health.json()["state"] == "active"
    assert health.json()["probe"]["backend"] == "V4L2"

    monkeypatch.setattr(
        sources_routes,
        "capture_webcam_snapshot",
        lambda device_index, privacy_polygon=None: (
            b"\xff\xd8\xff\xe0preview\xff\xd9",
            {"backend": "V4L2", "captured_at_utc": "2026-01-01T00:00:00Z"},
        ),
    )
    preview = client.get(f"/api/v1/sources/{source_id}/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    assert preview.headers["cache-control"] == "no-store"
    assert preview.headers["X-Shop-Preview-Backend"] == "V4L2"
    assert preview.content.startswith(b"\xff\xd8")


def test_source_api_state_is_derived_from_the_active_job(client) -> None:
    source = client.post("/api/v1/sources", json={"camera_key": "camera_01"}).json()["source"]
    source_id = source["source_id"]

    created = client.post(
        "/api/v1/jobs",
        json={
            "source": {"kind": "webcam", "source_id": source_id},
            "analysis_profile": "cpu_baseline_v1",
            "sampling_fps": 5.0,
            "privacy_mask_version": "zones_default_v1",
            "retain_evidence": True,
        },
    )
    assert created.status_code == 202, created.text
    job_id = created.json()["job"]["job_id"]

    listed = client.get("/api/v1/sources").json()["items"][0]
    assert listed["state"] == "queued"
    assert listed["active_job"] == {
        "job_id": job_id,
        "status": "queued",
        "cancel_requested": False,
    }
    assert listed["health"]["last_known_state"] == "idle"

    with client.app.state.context.db.session() as session:
        assert transition_job(
            session, job_id=job_id, from_states=("queued",), to_state="running"
        )

    running = client.get(f"/api/v1/sources/{source_id}").json()["source"]
    assert running["state"] == "active"
    assert running["active_job"]["status"] == "running"

    with client.app.state.context.db.session() as session:
        assert transition_job(
            session, job_id=job_id, from_states=("running",), to_state="completed"
        )

    finished = client.get(f"/api/v1/sources/{source_id}").json()["source"]
    assert finished["state"] == "idle"
    assert finished["active_job"] is None


def test_job_validation_idempotency_cancel_and_transitions(
    client, sample_video_bytes
) -> None:
    upload = upload_sample(client, sample_video_bytes)
    upload_id = upload["upload_id"]

    unsupported_profile = create_file_job(client, upload_id, analysis_profile="other")
    assert unsupported_profile.status_code == 422
    assert unsupported_profile.json()["error"]["code"] == "ANALYSIS_PROFILE_UNSUPPORTED"

    too_fast = create_file_job(client, upload_id, sampling_fps=7.5)
    assert too_fast.status_code == 422
    assert too_fast.json()["error"]["code"] == "SAMPLING_FPS_UNSUPPORTED"
    assert too_fast.json()["error"]["details"]["maximum_sampling_fps"] == 5.0

    wrong_zones = create_file_job(client, upload_id, privacy_mask_version="old_zones")
    assert wrong_zones.status_code == 422
    assert wrong_zones.json()["error"]["code"] == "ZONE_VERSION_MISMATCH"

    missing_upload = create_file_job(client, str(uuid.uuid4()))
    assert missing_upload.status_code == 404
    assert missing_upload.json()["error"]["code"] == "UPLOAD_NOT_FOUND"

    short_key = create_file_job(client, upload_id, key="short")
    assert short_key.status_code == 422
    assert short_key.json()["error"]["code"] == "IDEMPOTENCY_KEY_INVALID"

    created = create_file_job(client, upload_id, key="job-create-key-0001")
    assert created.status_code == 202
    assert created.headers["Location"].startswith("/api/v1/jobs/")
    assert created.headers["Retry-After"] == "2"
    job = created.json()["job"]
    job_id = job["job_id"]
    assert job["status"] == "queued"
    assert job["events_count"] == 0
    assert job["progress"]["stage"] == "queued"
    assert job["source"]["kind"] == "file"
    assert job["error"] is None

    replay = create_file_job(client, upload_id, key="job-create-key-0001")
    assert replay.status_code == 202
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert replay.json()["job"]["job_id"] == job_id

    reused = create_file_job(
        client, upload_id, key="job-create-key-0001", retain_evidence=False
    )
    assert reused.status_code == 409
    assert reused.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"

    detail = client.get(f"/api/v1/jobs/{job_id}")
    assert detail.status_code == 200
    transitions = detail.json()["job"]["transitions"]
    assert [(item["from_state"], item["to_state"]) for item in transitions] == [
        (None, "queued")
    ]

    cancelled = client.post(
        f"/api/v1/jobs/{job_id}/cancel", headers={"Idempotency-Key": "cancel-key-0001"}
    )
    assert cancelled.status_code == 202
    assert cancelled.json()["job"]["status"] == "cancelled"
    assert "complete" in cancelled.json()["message"].lower()

    cancel_replay = client.post(
        f"/api/v1/jobs/{job_id}/cancel", headers={"Idempotency-Key": "cancel-key-0001"}
    )
    assert cancel_replay.status_code == 202
    assert cancel_replay.headers["Idempotency-Replayed"] == "true"
    assert cancel_replay.json()["job"]["job_id"] == job_id

    already_terminal = client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert already_terminal.status_code == 200
    assert already_terminal.json()["job"]["status"] == "cancelled"

    detail = client.get(f"/api/v1/jobs/{job_id}").json()["job"]
    assert [(item["from_state"], item["to_state"]) for item in detail["transitions"]] == [
        (None, "queued"),
        ("queued", "cancelled"),
    ]
    listed = client.get("/api/v1/jobs", params={"state": "cancelled"}).json()["items"]
    assert [item["job_id"] for item in listed] == [job_id]


def test_queue_capacity_and_active_upload_hold(app_factory, sample_video_bytes) -> None:
    _app, client = app_factory(runtime__queued_jobs=0)
    upload = upload_sample(client, sample_video_bytes)
    upload_id = upload["upload_id"]
    first = create_file_job(client, upload_id)
    assert first.status_code == 202
    job_id = first.json()["job"]["job_id"]

    second = create_file_job(client, upload_id)
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "QUEUE_FULL"
    assert second.json()["error"]["retryable"] is True
    assert second.headers["Retry-After"] == "10"

    hold = client.delete(f"/api/v1/uploads/{upload_id}")
    assert hold.status_code == 409
    assert hold.json()["error"]["code"] == "UPLOAD_ACTIVE_HOLD"
    assert hold.json()["error"]["details"]["job_id"] == job_id

    assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 202
    third = create_file_job(client, upload_id)
    assert third.status_code == 202

    deleted = client.delete(f"/api/v1/uploads/{upload['upload_id']}")
    assert deleted.status_code == 409  # the third job still holds the upload


def test_event_queue_review_history_and_media(client, settings) -> None:
    seeded = seed_event(client.app, evidence=True)
    event_id = seeded["event_id"]
    clip_bytes = seeded["clip_bytes"]
    thumb_bytes = seeded["thumb_bytes"]

    listed = client.get("/api/v1/events").json()
    assert listed["unreviewed_count"] == 1
    assert listed["has_more"] is False
    item = listed["items"][0]
    assert item["event_id"] == event_id
    assert item["review_score"] is None
    assert item["score_semantics"] is None
    assert item["evidence"]["status"] == "available"
    assert item["evidence"]["media_url"] == f"/api/v1/events/{event_id}/media"
    assert item["source"]["label"] == "seed.avi"

    detail = client.get(f"/api/v1/events/{event_id}")
    assert detail.status_code == 200
    payload = detail.json()["event"]
    assert payload["job_status"] == "completed"
    assert payload["reviews"] == []
    assert payload["quality"]["no_activity_score_available"] is True
    assert payload["evidence"]["asset"]["sha256"] == "a" * 64
    assert payload["evidence"]["asset"]["frame_count"] == 20
    assert payload["evidence"]["asset"]["media_interval_ms"] == {"start": 0, "end": 2000}

    media = client.get(f"/api/v1/events/{event_id}/media")
    assert media.status_code == 200
    assert media.content == clip_bytes
    assert media.headers["content-type"] == "video/webm"
    assert media.headers["ETag"] == '"' + "a" * 64 + '"'
    ranged = client.get(
        f"/api/v1/events/{event_id}/media", headers={"Range": "bytes=0-3"}
    )
    assert ranged.status_code == 206
    assert ranged.content == clip_bytes[:4]
    assert ranged.headers["content-range"] == f"bytes 0-3/{len(clip_bytes)}"
    thumbnail = client.get(f"/api/v1/events/{event_id}/thumbnail")
    assert thumbnail.status_code == 200
    assert thumbnail.content == thumb_bytes

    empty_review = client.post(f"/api/v1/events/{event_id}/reviews", json={"decision": "benign"})
    assert empty_review.status_code == 422
    assert empty_review.json()["error"]["code"] == "VALIDATION_ERROR"

    theft_reason = client.post(
        f"/api/v1/events/{event_id}/reviews",
        json={"decision": "benign", "reason_codes": ["theft_observed"]},
    )
    assert theft_reason.status_code == 422

    unknown_reviewer = client.post(
        f"/api/v1/events/{event_id}/reviews",
        json=REVIEW_BODY,
        headers={"X-Shop-Reviewer": "unknown-person"},
    )
    assert unknown_reviewer.status_code == 403
    assert unknown_reviewer.json()["error"]["code"] == "REVIEWER_NOT_APPROVED"

    first = client.post(
        f"/api/v1/events/{event_id}/reviews",
        json=REVIEW_BODY,
        headers={"Idempotency-Key": "review-key-0001", "X-Shop-Reviewer": "local-reviewer"},
    )
    assert first.status_code == 201
    first_review = first.json()["review"]
    assert first_review["decision"] == "review_needed"
    assert first_review["actor"] == "local-reviewer"
    assert "not a finding of guilt" in first.json()["notice"]
    assert first.json()["review_state"] == "review_needed"

    second = client.post(
        f"/api/v1/events/{event_id}/reviews",
        json={
            "decision": "benign",
            "reason_codes": ["customer_owned_item"],
            "notes": "Customer showed a receipt for the item.",
            "supersedes_review_id": first_review["review_id"],
        },
    )
    assert second.status_code == 201
    assert second.json()["review_state"] == "benign"

    replay = client.post(
        f"/api/v1/events/{event_id}/reviews",
        json=REVIEW_BODY,
        headers={"Idempotency-Key": "review-key-0001", "X-Shop-Reviewer": "local-reviewer"},
    )
    assert replay.status_code == 201
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert replay.json()["review"]["review_id"] == first_review["review_id"]

    detail = client.get(f"/api/v1/events/{event_id}").json()["event"]
    assert detail["review_state"] == "benign"
    assert [review["decision"] for review in detail["reviews"]] == [
        "review_needed",
        "benign",
    ]
    assert detail["reviews"][1]["supersedes_review_id"] == first_review["review_id"]
    assert detail["review_score"] is None

    assert client.get("/api/v1/events", params={"reviewed": "true"}).json()["items"]
    assert client.get("/api/v1/events", params={"reviewed": "false"}).json()["items"] == []
    assert (
        len(client.get("/api/v1/events", params={"review_state": "benign"}).json()["items"])
        == 1
    )
    assert client.get("/api/v1/events", params={"review_state": "unknown"}).status_code == 422

    audit = client.get("/api/v1/audit", params={"action": "event_reviewed"}).json()
    assert len(audit["items"]) == 2

    metrics = client.get("/api/v1/metrics").json()
    assert metrics["summary"]["events_total"] == 1
    assert metrics["summary"]["events_by_review_state"]["benign"] == 1
    assert metrics["summary"]["reviews_today"] >= 2
    assert metrics["recent_events"][0]["event_id"] == event_id
    assert metrics["label"] == "actual_measured_result"


def test_event_pagination_filters_and_unavailable_evidence(client) -> None:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    first = seed_event(
        client.app, evidence=False, created_at=base, label="first.avi"
    )
    second = seed_event(
        client.app, evidence=False, created_at=base + timedelta(minutes=1), label="second.avi"
    )
    third = seed_event(
        client.app,
        evidence=False,
        created_at=base + timedelta(minutes=2),
        label="third.avi",
        review_state="benign",
    )

    page = client.get("/api/v1/events", params={"limit": 2})
    assert page.status_code == 200
    payload = page.json()
    assert payload["has_more"] is True
    assert len(payload["items"]) == 2
    assert [item["event_id"] for item in payload["items"]] == [
        third["event_id"],
        second["event_id"],
    ]
    assert payload["unreviewed_count"] == 2

    cursor = payload["next_cursor"]
    next_page = client.get("/api/v1/events", params={"limit": 2, "cursor": cursor})
    assert next_page.status_code == 200
    assert [item["event_id"] for item in next_page.json()["items"]] == [first["event_id"]]
    assert next_page.json()["has_more"] is False

    invalid_cursor = client.get("/api/v1/events", params={"cursor": "not-a-cursor"})
    assert invalid_cursor.status_code == 400
    assert invalid_cursor.json()["error"]["code"] == "INVALID_CURSOR"

    assert client.get("/api/v1/events", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/events", params={"limit": 101}).status_code == 422

    by_job = client.get(
        "/api/v1/events", params={"job_id": second["job_id"]}
    ).json()["items"]
    assert [item["event_id"] for item in by_job] == [second["event_id"]]

    unreviewed = client.get("/api/v1/events", params={"reviewed": "false"}).json()["items"]
    assert {item["event_id"] for item in unreviewed} == {
        first["event_id"],
        second["event_id"],
    }

    media = client.get(f"/api/v1/events/{first['event_id']}/media")
    assert media.status_code == 410
    assert media.json()["error"]["code"] == "EVIDENCE_UNAVAILABLE"


def test_event_delete_removes_evidence_and_keeps_tombstone(client, settings) -> None:
    seeded = seed_event(client.app, evidence=True)
    event_id = seeded["event_id"]
    folder = settings.storage.root / "evidence" / event_id
    assert any(folder.iterdir())

    deleted = client.delete(f"/api/v1/events/{event_id}")
    assert deleted.status_code == 202
    assert deleted.json()["status"] == "deleted"
    assert deleted.json()["evidence_deleted"] is True
    assert deleted.json()["file_cleanup_failures"] == 0
    assert not list(folder.iterdir())

    assert client.get(f"/api/v1/events/{event_id}").status_code == 404
    assert client.get(f"/api/v1/events/{event_id}/media").status_code == 404
    assert client.get("/api/v1/events").json()["items"] == []

    with_deleted = client.get("/api/v1/events", params={"include_deleted": "true"}).json()
    assert len(with_deleted["items"]) == 1
    tombstone = with_deleted["items"][0]
    assert tombstone["event_id"] == event_id
    assert tombstone["deleted_at"] is not None
    assert tombstone["evidence"]["status"] == "deleted"

    repeated = client.delete(f"/api/v1/events/{event_id}")
    assert repeated.status_code == 202
    assert repeated.json()["status"] == "deleted"

    audit = client.get("/api/v1/audit", params={"action": "event_deleted"}).json()
    assert len(audit["items"]) == 1


def test_retention_maintenance_expires_only_due_evidence(client, settings) -> None:
    expired = seed_event(
        client.app,
        evidence=True,
        retain_until=utcnow() - timedelta(days=1),
        label="expired.avi",
    )
    fresh = seed_event(
        client.app,
        evidence=True,
        retain_until=utcnow() + timedelta(days=1),
        label="fresh.avi",
    )
    expired_folder = settings.storage.root / "evidence" / expired["event_id"]
    fresh_folder = settings.storage.root / "evidence" / fresh["event_id"]

    maintenance = client.post("/api/v1/maintenance/retention")
    assert maintenance.status_code == 202
    payload = maintenance.json()
    assert payload["clips_expired"] == 1
    assert payload["events_updated"] == 1
    assert payload["file_cleanup_failures"] == 0
    assert not list(expired_folder.iterdir())
    assert list(fresh_folder.iterdir())

    expired_event = client.get(f"/api/v1/events/{expired['event_id']}").json()["event"]
    assert expired_event["evidence"]["status"] == "expired"
    assert "retention" in expired_event["evidence"]["reason"].lower()
    expired_media = client.get(f"/api/v1/events/{expired['event_id']}/media")
    assert expired_media.status_code == 410
    assert expired_media.json()["error"]["code"] == "EVIDENCE_UNAVAILABLE"

    fresh_media = client.get(f"/api/v1/events/{fresh['event_id']}/media")
    assert fresh_media.status_code == 200

    repeat = client.post("/api/v1/maintenance/retention")
    assert repeat.json()["clips_expired"] == 0

    audit = client.get("/api/v1/audit", params={"action": "retention_applied"}).json()
    assert len(audit["items"]) == 1
