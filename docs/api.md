# API contract

Base path: `/api/v1`  
Local base URL: `http://127.0.0.1:8000`  
OpenAPI: `/api/openapi.json`  
Interactive docs: `/api/docs`

All timestamps are RFC 3339 UTC. Resource IDs are UUID strings. Media intervals are integer milliseconds relative to the source session. Coordinates, when exposed, are original-pixel continuous `xyxy`. Errors never include raw paths, stack traces, SQL, or secrets.

## Authentication and operator identity

`/api/v1/health/live` and `/api/v1/health/ready` are public. All other routes require a bearer token **only when** `SHOP_API_TOKEN` is configured:

```http
Authorization: Bearer <token>
```

Mutations and reviews also send a reviewer identity:

```http
X-Shop-Reviewer: local-reviewer
```

The value must match an entry in `security.reviewers` in `configs/shop.yaml`. Missing/invalid identity returns `403`. The UI stores the token in browser `localStorage`; the token is never returned by any endpoint.

## Request IDs

Every response includes `X-Request-ID`. Error bodies repeat it as `request_id`. Use it to correlate a user-visible failure with local logs; it contains no personal data.

## Error envelope

```json
{
  "error": {
    "code": "STORAGE_LOW",
    "message": "Analysis cannot start while free storage is below the configured reserve.",
    "retryable": true,
    "details": {}
  },
  "request_id": "4d0f0a2e-2a63-4e3c-8f0a-1f2d3c4b5a69"
}
```

Representative codes:

| HTTP | Code | Meaning |
| --- | --- | --- |
| 400 | `INVALID_CURSOR`, `INVALID_CONTENT_LENGTH` | Malformed cursor or header |
| 401 | `UNAUTHENTICATED` | Missing/invalid bearer token |
| 403 | `REVIEWER_NOT_ALLOWED`, `CAMERA_NOT_APPROVED` | Identity or camera is not in local configuration |
| 404 | `EVENT_NOT_FOUND`, `UPLOAD_NOT_FOUND`, `JOB_NOT_FOUND`, `SOURCE_NOT_FOUND` | Resource unavailable to this local scope |
| 409 | `UPLOAD_ACTIVE_HOLD`, `IDEMPOTENCY_KEY_REUSED`, `SOURCE_ALREADY_EXISTS` | State conflict |
| 412 | `STALE_VERSION` | `If-Match` no longer matches the source ETag |
| 413 | `UPLOAD_TOO_LARGE` | Byte limit exceeded |
| 415 | `UNSUPPORTED_EXTENSION`, `MEDIA_UNSUPPORTED` | Container or codec cannot be validated |
| 422 | `VALIDATION_ERROR`, `ANALYSIS_PROFILE_UNSUPPORTED`, `SAMPLING_FPS_UNSUPPORTED`, `ZONE_VERSION_MISMATCH`, `UPLOAD_NOT_READY` | Request is syntactically valid but not allowed |
| 428 | `IF_MATCH_REQUIRED` | Source PATCH lacks an ETag |
| 429 | `QUEUE_FULL` | Local queue capacity reached; `Retry-After` is present |
| 503 | `STORAGE_LOW`, `MODEL_UNAVAILABLE`, `CAMERA_DISCONNECTED` | Required local dependency is unavailable |
| 500 | `INTERNAL_ERROR` | Unexpected failure; details are logged locally, not returned |

## Idempotency

`POST /jobs`, `POST /jobs/{id}/cancel`, and `POST /events/{id}/reviews` accept:

```http
Idempotency-Key: <8–100 characters>
```

For the same actor, route, key, and body, the original status/body is replayed with `Idempotency-Replayed: true`. Reusing a key with a different body returns `409 IDEMPOTENCY_KEY_REUSED`. Keys are scoped to the local actor and route.

## Endpoints

### Health and capability

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health/live` | Process liveness only; no sensitive configuration |
| GET | `/health/ready` | 200 only when database, storage reserve, person model, and fresh worker heartbeat pass; otherwise 503 `degraded` with checks |
| GET | `/capabilities` | Host CPU profile, source availability, model roles/approval, limits, security, feature flags |
| GET | `/models` | Model manifest status, hashes, approval, rights notes, evaluation state |
| GET | `/metrics` | Database-backed counts, queue/worker state, disk, recent events, uptime |
| GET | `/config` | Redacted effective config; storage root is `managed_local_storage` |

`/capabilities` and `/models` report disabled retail/pose/temporal modules with reasons. They never imply retail accuracy.

### Uploads

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/uploads` | `multipart/form-data` field `file`; 201 with probe and SHA-256 |
| GET | `/uploads` | Latest 100 active uploads |
| GET | `/uploads/{upload_id}` | Validation metadata |
| DELETE | `/uploads/{upload_id}` | 202; 409 if an active job holds it |

Accepted extensions: MP4, MOV, WebM, MKV, AVI, M4V (case-insensitive). The decoder observation is authoritative. Empty, oversized, undecodable, or over-duration media is rejected. Upload is streamed to a hidden partial file and atomically promoted.

```bash
curl -s -X POST \
  -F "file=@sample.avi" \
  -H "X-Shop-Reviewer: local-reviewer" \
  http://127.0.0.1:8000/api/v1/uploads
```

### Sources (webcam only in this build)

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/sources` | Register an approved `camera_key`; 201 + ETag |
| GET | `/sources` | Registered webcams + approved camera list with `registered` flags |
| GET | `/sources/{source_id}` | Configuration, stored health, and derived active-job state |
| PATCH | `/sources/{source_id}` | Label update; requires `If-Match`; 412 on stale ETag |
| GET | `/sources/{source_id}/health` | Probes the local device; `active`/`degraded`/`disconnected` |
| GET | `/sources/{source_id}/preview` | One JPEG snapshot; applies the configured privacy mask; captures and releases the device |

Only camera keys declared under `security.approved_cameras` can be registered. RTSP returns `422`/is reported unavailable. The API never accepts a raw device path or arbitrary URL from the client.

`source.state` reflects a live job when one is active (`queued`, `active`, `cancelling`); the last device probe is retained separately as `health.last_known_state`.

### Jobs

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/jobs` | 202; `Location: /api/v1/jobs/{id}`; queue/storage/model gates |
| GET | `/jobs` | Latest 50; optional `state`, `limit` 1–100 |
| GET | `/jobs/{job_id}` | Status, progress, metrics, coverage, source label, transitions |
| POST | `/jobs/{job_id}/cancel` | 202 for queued/running; 200 if already terminal; idempotent |

Create body:

```json
{
  "source": { "kind": "file", "upload_id": "3b0ed65b-7b33-46ad-bb8e-5c7ef73288d4" },
  "analysis_profile": "cpu_baseline_v1",
  "sampling_fps": 5,
  "privacy_mask_version": "zones_default_v1",
  "retain_evidence": true
}
```

Webcam body uses `{ "kind": "webcam", "source_id": "<uuid>" }` instead.

Validation gates:

- `analysis_profile` must be exactly `cpu_baseline_v1`.
- `sampling_fps` must be `> 0` and no higher than the configured profile maximum.
- `privacy_mask_version` must match the active zones version.
- The person model must be available.
- Free storage must be above the reserve.
- Queue depth must be below `active_jobs + queued_jobs` (currently 1 + 2).

Job states: `queued`, `running`, `cancelling`, `completed`, `failed`, `cancelled`. Progress includes stage, media position, analysed/expected samples, coverage, detection count, event count, decode gaps, and source counters. `metrics.label` is `actual_measured_result`.

### Events

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/events` | Cursor page; filters `limit`, `review_state`, `reviewed`, `job_id`, `include_deleted`; includes `unreviewed_count` |
| GET | `/events/{event_id}` | Full event, source, job status, evidence asset, append-only reviews |
| POST | `/events/{event_id}/reviews` | 201; append-only review; updates derived `review_state` |
| GET | `/events/{event_id}/media` | WebM stream; supports `Range`; 206 partial |
| GET | `/events/{event_id}/thumbnail` | JPEG thumbnail |
| DELETE | `/events/{event_id}` | 202; tombstone + media removal; hides from queue |

Event shape (abridged):

```json
{
  "event_id": "d8ade998-5510-42bf-856c-11962cde081f",
  "media_interval_ms": { "start": 22600, "end": 27200 },
  "event_type": "person_activity_observed",
  "reasons": ["person_track_observed"],
  "review_score": null,
  "score_semantics": null,
  "track_ids": ["t0010"],
  "quality": {
    "observed_samples": 11,
    "observed_duration_ms": 2200,
    "observed_fraction": 0.4783,
    "predicted_gaps": 17,
    "warnings": ["intermittent_track_observations"]
  },
  "required_evidence": ["person_detections", "source_coverage"],
  "available_evidence": ["person_detections", "track_observations"],
  "evidence": {
    "status": "available",
    "clip_id": "7cea016d-cca8-4454-84aa-807fe00c5b93",
    "media_url": "/api/v1/events/d8ade998-5510-42bf-856c-11962cde081f/media",
    "thumbnail_url": "/api/v1/events/d8ade998-5510-42bf-856c-11962cde081f/thumbnail",
    "asset": {
      "duration_ms": 25200,
      "media_interval_ms": { "start": 12500, "end": 37700 },
      "fps": 10,
      "encoder": "opencv_vp8",
      "sha256": "c5d1ba4e923edddc43487ae62892248f9121b7745dc0b277918ad3978343f75f"
    }
  },
  "review_state": "inconclusive",
  "reviews": []
}
```

`review_score` is `null` when no calibrated score exists. A client must render "Not scored", not `0`. `evidence.asset.media_interval_ms` is the retained context window; `media_interval_ms` is the observed interval. Do not substitute one for the other.

Review body:

```json
{
  "decision": "inconclusive",
  "reason_codes": ["context_insufficient"],
  "notes": "Evaluation fixture only; reviewed to verify the workflow.",
  "supersedes_review_id": null
}
```

- `decision`: `benign` | `review_needed` | `inconclusive`.
- `reason_codes`: up to 8 from `context_insufficient`, `activity_benign`, `occlusion`, `product_unclear`, `customer_owned_item`, `equipment_fault`, `other`.
- At least one reason code or non-empty note is required.
- Notes are capped at 2000 characters.
- Corrections append; they do not overwrite history. `supersedes_review_id` must refer to a prior review of the same event.

Media endpoints return `410`/`409`-style safe errors when evidence is deleted, expired, or unavailable. They never redirect to a filesystem path or a cloud URL.

### Audit and retention

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/audit` | Cursor page; filters `limit` 1–100, `action`, `object_type` |
| POST | `/maintenance/retention` | 202; expires due evidence clips, keeps tombstones and audit |

Retention is explicit local maintenance, not an automatic cloud lifecycle. The endpoint deletes at most 500 due clips per call and reports `clips_expired`, `events_updated`, and `file_cleanup_failures`.

## Pagination

Event and audit lists use opaque cursors. Pass the returned `next_cursor` unchanged. `has_more` indicates another page. Invalid cursors return `400 INVALID_CURSOR`. There is no offset pagination, so concurrent inserts do not shift pages unpredictably.

## HTTP behavior

- Upload size is checked from `Content-Length` and enforced again during streaming.
- Source PATCH uses `ETag` + `If-Match`; the response returns the new ETag.
- Media supports `Range: bytes=start-end` with `Accept-Ranges: bytes`.
- `DELETE` endpoints use 202 with a body; they are not 204.
- The SPA is served at `/` with CSP `default-src 'self'`, blob media, and no inline script.
- `/api/docs` is available for local inspection; it is not an authorization mechanism.

## Known contract gaps

These PRD endpoints are not implemented in this baseline:

- `POST /exports` and export jobs.
- `GET /deletions/{id}` as a separate deletion-status resource.
- RTSP source registration and credential handling.
- Role-separated authorization beyond the local reviewer allowlist.

`DELETE /events/{id}` returns the resulting tombstone state directly; retention maintenance is the current deletion-status mechanism. See [Status](status.md).

## Shelf-action research mode (added 18 Nov 2026)

| Method | Path | Notes |
| --- | --- | --- |
| `POST` | `/jobs` | `analysis_profile: "merl_shelf_actions_research_v1"` (file sources only; 422 `ANALYSIS_PROFILE_SOURCE_UNSUPPORTED` for webcams; 503 `ACTION_MODEL_UNAVAILABLE` if the checkpoint fails its checks) |
| `GET` | `/jobs/{id}/action-timeline` | Per-step scores (`shop_action_timeline_v1`): classes, thresholds, `pts_ms`, `scores[K][T]`, `person_present`, episodes. 404 for baseline jobs, 409 until completed, 410 if the file is gone, 500 on SHA-256 mismatch |
| `GET` | `/events?event_type=shelf_interaction_observed` | Filter by event type |
| `POST` | `/events/{id}/reviews` | Optional `action_feedback: {verdict: correct\|partially_correct\|incorrect\|no_interaction\|unclear, observed_actions: [...]}`. Only for shelf-interaction events (422 `ACTION_FEEDBACK_NOT_APPLICABLE` otherwise) |
| `GET` | `/research/insights` | Committed `ml/results/*` (MERL analysis, CPU benchmark, runtime parity, RetailAction status, pose gate, own evaluation) and live aggregates: research jobs, warnings, reviewer precision with Wilson 95% intervals |

Shelf events carry `actions: [{label, start_ms, end_ms, mean_score, peak_score, threshold, score_semantics}]`.
`review_score` stays `null`; action scores are `uncalibrated_action_score`. Job `coverage.action_analysis` holds
the model identity, input properties, warnings, per-class summary and stage timing.
