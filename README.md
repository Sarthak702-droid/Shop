# Shop

Local-first, CPU-only retail video analysis with mandatory human review.

Shop analyses prerecorded files or an approved local webcam, detects and tracks people, and publishes short **descriptive** review segments. It never decides that theft, guilt, or intent occurred. Every event is an observation for a person to review, and the comparison score is stored as `null` because no calibrated model exists yet.

> **Current status:** baseline development build. Person detection and session-local tracking work on CPU. Retail-product detection, pose, and the temporal model are disabled. The bundled person checkpoint is **development-only** because its commercial rights are unresolved. See [Status](docs/status.md).

## What works today

- Upload MP4/MOV/WebM/MKV/AVI/M4V, validate it locally, and analyse at 5 Hz on CPU.
- Register an approved webcam, preview one frame, and run bounded live analysis.
- Durable SQLite job queue with a separate worker process, cancellation, lease recovery, and no inference in HTTP handlers.
- Own IoU + Kalman + Hungarian tracker with tentative/confirmed/lost/removed states and session-local IDs.
- Descriptive event policy: `person_activity_observed`, never a theft claim.
- VP8 WebM evidence clips with before/after context, burned-in boxes, thumbnails, SHA-256 integrity, HTTP Range playback, and local deletion.
- Append-only human reviews: `benign`, `review_needed`, or `inconclusive`.
- FastAPI `/api/v1` with typed validation, safe error envelopes, request IDs, optional bearer token, reviewer identity, ETag source updates, idempotency, retention maintenance, and audit records.
- React + TypeScript review UI in the Shop design system, with loading, empty, error, and unavailable-evidence states.

## Quick start

Prerequisites: Linux (developed on Pop!_OS), Python 3.12, Node.js 20+, a CPU. No CUDA is required or used.

```bash
cd /path/to/ShopliftingAI

# 1. Python environment (CPU wheels first, then the project)
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-cpu.txt
.venv/bin/pip install -e . --no-deps
.venv/bin/pip install -r requirements-dev.txt

# 2. Place the person checkpoint described in docs/models.md
#    Expected: var/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth

# 3. Build the review UI
cd apps/web
npm ci
npm run build
cd ../..

# 4. Start the API (terminal 1) and worker (terminal 2)
.venv/bin/uvicorn shop_api.main:app --host 127.0.0.1 --port 8000
.venv/bin/python -m shop_worker
```

Open <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/api/docs>.

For a clean checkout, the API creates `var/shop.db` and applies migrations on startup. The worker never migrates. To prepare state explicitly:

```bash
.venv/bin/alembic upgrade head
```

### Optional local authentication

```bash
export SHOP_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv/bin/uvicorn shop_api.main:app --host 127.0.0.1 --port 8000
```

When `SHOP_API_TOKEN` is set, every `/api/v1` route except `/api/v1/health/*` requires `Authorization: Bearer <token>`. The review UI stores the token in browser `localStorage` and sends it with requests. Reviewer identity is sent as `X-Shop-Reviewer` and must match `security.reviewers` in `configs/shop.yaml`.

## Verify

```bash
# Backend contracts, math, persistence, API integration
.venv/bin/python -m pytest tests -q

# Frontend unit and interaction tests, then production build
cd apps/web
npm test -- --run
npm run build
```

Last recorded run on this host: **84 backend tests passed**, **54 frontend tests passed**, and the production build succeeded (57 modules; 49.32 kB CSS, 373.90 kB JS before gzip). See [Evaluation](docs/evaluation.md) for the measured CPU run and browser checks.

## Project layout

```text
apps/api/shop_api/          FastAPI app, routes, schemas, serializers
apps/worker/shop_worker/    Durable job worker and database sink
apps/web/                   React + TypeScript + Vite review interface
packages/shop_core/         Config, contracts, sources, detection, tracking,
                            policy, evidence, persistence, pipeline
ml/manifests/               Versioned model manifests and disabled-gate records
migrations/                 Alembic schema history
configs/shop.yaml           Local CPU profile and safety limits
tests/                      Backend unit and integration tests
var/                        Local runtime state (not source; see .gitignore)
docs/                       Architecture, API, models, data, evaluation, runbook, status
```

Runtime state includes the SQLite database, uploads, evidence clips, thumbnails, and worker heartbeat. Absolute paths are never returned by the API; configuration exposes the storage root as `managed_local_storage`.

## Documentation

| Document | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Components, processing flow, contracts, tracking, policy, evidence, storage, failure handling |
| [API contract](docs/api.md) | Endpoints, auth, errors, idempotency, media streaming, examples |
| [Model registry](docs/models.md) | Manifests, hashes, rights, approval gates, disabled modules, feature schema |
| [Data lineage](docs/data.md) | Source-to-event lineage, database schema, retention, dataset rights status |
| [Evaluation](docs/evaluation.md) | Measured CPU run, test results, browser verification, metric limits, provenance labels |
| [Runbook](docs/runbook.md) | Install, run, operate, back up, reset, troubleshoot |
| [Status](docs/status.md) | Delivered scope, deferred capabilities, conflicts, missing artifacts, known gaps |

## Data and rights

- Only one evaluation fixture was downloaded: OpenCV's public `vtest.avi`, kept under `var/evaluation/incoming/`. It is used **only** to exercise decoding, tracking, evidence, and review. It is not training data, has no ground-truth theft labels, and its redistribution rights are not cleared.
- No public dataset was bulk-downloaded. UCF Crime, MEVA, SKU110K, RPC, D2S, MOT17, Avenue, and ShanghaiTech remain quarantined or permission-gated in [Data](docs/data.md).
- The person detector uses COCO-derived TorchVision weights. Annotation and image rights differ, and commercial permission is unresolved. The manifest therefore declares `development_only` and `approved_for_commercial_release: false`.
- No audio is captured or processed. Audio is disabled in the configuration contract.

## Safety and non-claims

- Events describe observable activity only: "person activity observed", "review needed", "item visibility lost", "insufficient evidence", "analysis unavailable", or "inconclusive".
- `review_score` is `null` in the baseline. It is never displayed as zero risk or as a calibrated probability.
- Pose keypoints, wrist proximity, product visibility, or leaving a frame cannot establish concealment, intent, ownership, payment, or theft.
- This repository does not include a commercial model, a trained temporal model, or a validated retail accuracy claim.
- Operational review only. No enforcement action is automated or recommended by the software.
