# Runbook

This runbook covers a single local Linux host. Commands assume the repository root (`/path/to/ShopliftingAI`). No CUDA, cloud service, RTSP camera, or system FFmpeg is required.

## 1. Prerequisites

- Linux x86_64 (developed on Pop!_OS), Python 3.12, Node.js 20+, npm.
- ~10 GB free disk for the environment, checkpoint, and local evidence. Keep more for uploads.
- An approved local camera only if live analysis is needed. `/dev/video0` or `/dev/video1` on the development host.
- The person checkpoint at `var/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth` with SHA-256 `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`.

The repository does not download the checkpoint automatically. Loading refuses a missing or mismatched file.

## 2. Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-cpu.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/pip install -r requirements-dev.txt

cd apps/web
npm ci
npm run build
cd ../..
```

`requirements-cpu.txt` pins the recorded CPU-only Torch/TorchVision pair and the Python dependencies. `torch.version.cuda` must be `None`; `/api/v1/capabilities` proves this at runtime.

Verify the environment:

```bash
.venv/bin/python -c "import torch, torchvision; print(torch.__version__, torch.version.cuda, torchvision.__version__)"
# expected: 2.10.0+cpu None 0.25.0+cpu
```

## 3. Configuration

Edit `configs/shop.yaml`. Important values:

| Key | Default | Meaning |
| --- | --- | --- |
| `storage.root` | `var` | Must resolve inside the project; holds DB, uploads, evidence, heartbeat |
| `runtime.inference_threads` | 4 | CPU threads for inference |
| `runtime.allow_development_models` | true | Allows `development_only` person checkpoint for local development |
| `analysis.sampling_fps` | 5.0 | Maximum analysis cadence |
| `analysis.long_edge_px` | 640 | Frame working resolution for detection |
| `analysis.evidence_context_before_ms` | 10000 | Retained context before the event |
| `analysis.evidence_context_after_ms` | 10000 | Retained context after the event |
| `security.reviewers` | `local-reviewer` | Allowed `X-Shop-Reviewer` identities |
| `security.approved_cameras` | camera_01, camera_02 | Allowlisted local devices |
| `models.*.enabled` | person only | Retail, pose, temporal are disabled |

Use another config file with:

```bash
export SHOP_CONFIG=/absolute/path/to/shop.yaml
```

Validation is strict: unknown fields, bad ranges, association weights that do not sum to 1.0, unsafe storage roots, and RTSP/audio enablement are rejected. Numeric values are proposed starting settings, not measured optima.

## 4. Database and startup order

The API applies Alembic migrations on startup when `storage.auto_migrate: true`. The worker does **not** migrate.

Preferred:

```bash
.venv/bin/alembic upgrade head

# terminal 1
.venv/bin/uvicorn shop_api.main:app --host 127.0.0.1 --port 8000

# terminal 2
.venv/bin/python -m shop_worker
```

The worker:

- writes `var/worker-heartbeat.json` at least every few seconds;
- claims one job at a time by default;
- refreshes its lease and honors cancellation;
- marks interrupted jobs failed with `WORKER_INTERRUPTED` rather than replaying them.

Process a single job for debugging:

```bash
.venv/bin/python -m shop_worker --once
.venv/bin/python -m shop_worker --job-id <JOB_ID>
```

Open <http://127.0.0.1:8000>. API docs: <http://127.0.0.1:8000/api/docs>.

### Frontend development

```bash
cd apps/web
npm run dev
# http://127.0.0.1:5173 with /api proxied to 127.0.0.1:8000
```

For the API-served production UI, run `npm run build` and reload <http://127.0.0.1:8000>.

## 5. Authentication

Without `SHOP_API_TOKEN`, the API is open on loopback. Set a token for shared/local multi-user use:

```bash
export SHOP_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv/bin/uvicorn shop_api.main:app --host 127.0.0.1 --port 8000
```

- `GET /api/v1/health/*` remains public.
- Every other `/api/v1` route requires `Authorization: Bearer <token>`.
- Mutations/reviews require `X-Shop-Reviewer` matching `security.reviewers`.
- Enter the token once in **Settings**; it is kept in browser `localStorage`.
- Never commit the token or put it in a URL.

For curl examples below, add the header when auth is enabled:

```bash
-H "Authorization: Bearer $SHOP_API_TOKEN"
```

## 6. Upload and file analysis

```bash
curl -s -X POST \
  -F "file=@/path/to/recording.mp4" \
  -H "X-Shop-Reviewer: local-reviewer" \
  http://127.0.0.1:8000/api/v1/uploads
```

Take `upload.upload_id`, then:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -H "X-Shop-Reviewer: local-reviewer" \
  -H "Idempotency-Key: local-job-0001" \
  -d '{
    "source": {"kind": "file", "upload_id": "<UPLOAD_ID>"},
    "analysis_profile": "cpu_baseline_v1",
    "sampling_fps": 5,
    "privacy_mask_version": "zones_default_v1",
    "retain_evidence": true
  }'
```

Poll the job or use the UI:

```bash
curl -s http://127.0.0.1:8000/api/v1/jobs/<JOB_ID>
curl -s "http://127.0.0.1:8000/api/v1/events?job_id=<JOB_ID>"
```

Cancellation:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/jobs/<JOB_ID>/cancel \
  -H "X-Shop-Reviewer: local-reviewer"
```

A queued job moves to `cancelled`; a running job enters `cancelling`, stops at a sample boundary, flushes the current segment, and then becomes `cancelled`. Repeating cancel on a terminal job returns 200 with its state.

If the worker is not running, the job stays `queued` and readiness is degraded. Do not create a second worker to "fix" this; investigate the existing one first.

## 6a. Shelf-action research analysis

```bash
scripts/models/fetch_merl_tcn.sh          # var/models/merl_tcn_v1/model.pt, SHA-256 checked against the manifest
scripts/demo/run_demo.sh [video.mp4]      # upload + queue + wait + summary through the live API
```

In the UI: **Add footage → Shelf actions (MERL research model) → Queue shelf-action analysis**. The job
page shows the action timeline (click a point to open the nearest episode). Review each episode and
answer the action-feedback question; results appear on **Research insights**.

- Profile `merl_shelf_actions_research_v1`: file uploads only; cadence fixed at 5 Hz steps.
- Timeline: `GET /api/v1/jobs/{id}/action-timeline` (409 until completed; integrity-checked).
- Insights: `GET /api/v1/research/insights`. Regenerate offline results with
  `ml/merl/analyze.py`, `ml/merl/runtime_e2e.py`, `ml/retailaction/pose_compat.py` and
  `ml/retailaction/status_snapshot.py`.
- A missing checkpoint disables only this profile (`ACTION_MODEL_UNAVAILABLE`); baseline jobs keep working.

## 7. Webcam analysis

The camera must be listed under `security.approved_cameras`. Register it once:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/sources \
  -H "Content-Type: application/json" \
  -H "X-Shop-Reviewer: local-reviewer" \
  -d '{"camera_key":"camera_01"}'
```

Check and preview:

```bash
curl -s http://127.0.0.1:8000/api/v1/sources
curl -s http://127.0.0.1:8000/api/v1/sources/<SOURCE_ID>/health
curl -s http://127.0.0.1:8000/api/v1/sources/<SOURCE_ID>/preview -o /tmp/shop-preview.jpg
```

Preview captures one frame and releases the device. A preview and an analysis job cannot hold the camera at the same time; stop one before starting the other.

Create a live job:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -H "X-Shop-Reviewer: local-reviewer" \
  -d '{
    "source": {"kind": "webcam", "source_id": "<SOURCE_ID>"},
    "analysis_profile": "cpu_baseline_v1",
    "sampling_fps": 5,
    "privacy_mask_version": "zones_default_v1",
    "retain_evidence": true
  }'
```

Live analysis runs until cancelled or the 30-minute safety duration. It uses a bounded freshness queue, drops stale frames, and keeps a bounded JPEG ring for evidence. A live job has no finite coverage denominator; dropped/stale counts are reported separately.

Rename a source with ETag:

```bash
ETAG=$(curl -s -D - -o /dev/null http://127.0.0.1:8000/api/v1/sources/<SOURCE_ID> | awk -F': ' 'tolower($1)=="etag"{gsub("\r",""); print $2}')
curl -s -X PATCH http://127.0.0.1:8000/api/v1/sources/<SOURCE_ID> \
  -H "Content-Type: application/json" \
  -H "If-Match: $ETAG" \
  -H "X-Shop-Reviewer: local-reviewer" \
  -d '{"label":"Front entrance"}'
```

A missing `If-Match` returns 428; a stale value returns 412.

## 8. Review operations

- The queue defaults to unreviewed segments.
- A review needs a decision plus at least one reason code or a note.
- `benign`, `review_needed`, and `inconclusive` are operational review outcomes, not guilt findings.
- Corrections append; earlier decisions remain visible.
- "Not scored" means the score is `null`; it does not mean zero risk.
- If evidence is unavailable, review the event with the stated quality limitation or mark `inconclusive`; do not infer visual content that is not present.

Delete an event only when the operational record is no longer needed:

```bash
curl -s -X DELETE http://127.0.0.1:8000/api/v1/events/<EVENT_ID> \
  -H "X-Shop-Reviewer: local-reviewer"
```

This hides the event, removes its media, keeps review/audit tombstones, and returns 202.

## 9. Retention and audit

Apply due evidence retention explicitly:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/maintenance/retention \
  -H "X-Shop-Reviewer: local-reviewer"
```

It processes up to 500 expired clips per call, marks evidence `expired`, keeps metadata, and reports cleanup failures. Run it from a local scheduler if desired; it is not a cloud lifecycle rule.

Inspect audit:

```bash
curl -s "http://127.0.0.1:8000/api/v1/audit?limit=50"
curl -s "http://127.0.0.1:8000/api/v1/audit?action=review_created"
```

## 10. Health checks

```bash
curl -s http://127.0.0.1:8000/api/v1/health/live
curl -s -i http://127.0.0.1:8000/api/v1/health/ready
curl -s http://127.0.0.1:8000/api/v1/capabilities
curl -s http://127.0.0.1:8000/api/v1/metrics
```

Readiness checks:

1. SQLite query and schema.
2. Free storage above reserve.
3. Person detector available (manifest, hash, rights for configured use, runtime pair).
4. Worker heartbeat fresh (≤20 s) and not blocked/stopped.

A 503 with `"status":"degraded"` is actionable, not a generic outage. Inspect the failing check's `detail` and `observed` fields.

## 11. Backup

Stop the API and worker, then copy state:

```bash
mkdir -p /safe/backup
cp var/shop.db /safe/backup/shop.db
cp -a var/uploads /safe/backup/uploads
cp -a var/evidence /safe/backup/evidence
cp -a var/thumbnails /safe/backup/thumbnails 2>/dev/null || true
```

Do not copy a live WAL database and assume it is consistent. Stop writers, or use SQLite's online backup API/CLI. The database alone is insufficient for evidence playback; copy the relative media directories too. `var/models` contains checkpoints and should be backed up separately according to its rights constraints.

## 12. Reset local evaluation state

**This destroys local uploads, events, reviews, evidence, and audit.** Back up first.

```bash
# stop API and worker first
rm -f var/shop.db var/worker-heartbeat.json
rm -rf var/uploads var/evidence var/thumbnails
mkdir -p var/uploads var/evidence var/thumbnails

.venv/bin/alembic upgrade head
.venv/bin/uvicorn shop_api.main:app --host 127.0.0.1 --port 8000
.venv/bin/python -m shop_worker
```

Do not delete `var/models` or `var/evaluation/incoming/vtest.avi` unless you intend to reinstall/re-download them.

## 13. Logs and diagnostics

- Uvicorn logs to the terminal; request errors include the request ID in the response.
- Worker logs are JSON lines with `service: shop-worker`.
- `var/worker-heartbeat.json` shows worker state/job/reason.
- Job detail includes `transitions`, `progress`, `metrics`, `coverage`, and a safe `error`.
- `/api/v1/metrics` reports queue depth, worker state, disk, and counts.
- Evidence generation failures are stored as `evidence.status = unavailable` with a reason, not swallowed.

## 14. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `MODEL_ARTIFACT_MISSING` / readiness degraded | Checkpoint absent | Place the file at the manifest path; verify SHA-256 |
| `MODEL_HASH_MISMATCH` | Wrong or corrupted checkpoint | Re-obtain the exact artifact; do not edit the hash to match |
| `MODEL_RIGHTS_UNAPPROVED` | Manifest status not allowed | Review rights; `development_only` needs `allow_development_models: true` |
| Jobs stay queued | Worker absent/stale | Start worker; check heartbeat and `/health/ready` |
| Worker exits immediately | Schema not migrated or config invalid | Run `alembic upgrade head`; read the worker error |
| `STORAGE_LOW` | Free space below reserve | Free disk or back up/remove old evidence; increase reserve only deliberately |
| `QUEUE_FULL` | 1 active + 2 queued | Wait for completion or cancel a queued job; do not raise limits without CPU evidence |
| `CAMERA_DISCONNECTED` / `CAMERA_UNAVAILABLE` | Device busy, unplugged, or permission denied | Close other camera apps; check `/dev/video*` and group permissions; run health/preview |
| Preview fails but file analysis works | Camera held by another process | Release the device; do not run preview and live analysis together |
| Empty/unavailable evidence | Encoder failure, retention disabled, bad seek/window | Read the event's evidence reason; regenerate from a retained source if needed |
| Video does not play | Browser codec support or stale blob | Use Chromium/Firefox with WebM support; retry the clip; check media endpoint status |
| `VALIDATION_ERROR` | Extra/missing fields or bad enum | Compare with `/api/docs`; clients must not send raw paths or IDs they do not own |
| `IDEMPOTENCY_KEY_REUSED` | Same key, different body | Use a new key for a different request |
| 429/503 on upload | Queue/storage pressure | Back off using `Retry-After`; inspect metrics |
| UI shows request reference only | Safe error envelope | Search logs/audit by that request ID |

## 15. Stop and clean up

```bash
# Ctrl-C in each terminal, or:
pkill -f "uvicorn shop_api.main:app"
pkill -f "shop_worker"
```

Do not use `kill -9` while a job is running unless necessary; graceful cancellation keeps the database consistent. If the host reboots, restart the API (migrations), then the worker; interrupted jobs are marked failed rather than silently replayed.

## 16. Operational limits

- One active job and two queued jobs by default.
- One local store; no multi-tenant authorization model.
- RTSP, exports, external deletion queues, and enforcement integrations are not implemented.
- No automatic cloud upload, telemetry, audio, face recognition, or appearance ReID.
- The person checkpoint is development-only. Do not describe the deployment as commercially approved.
