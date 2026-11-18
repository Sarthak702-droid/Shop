# Status, conflicts, and known gaps

Last updated: 18 November 2026.

This is a working local baseline, not a commercial release. The person pipeline, review workflow, evidence, API, and UI are exercised on real CPU inference. Retail detection, pose, the temporal model, RTSP, and exports are not delivered.

Provenance labels: `[Publisher metadata]`, `[Proposed target]`, `[Illustrative assumption]`, `[Actual measured result]`.

## Delivered

| Area | Status | Evidence |
| --- | --- | --- |
| CPU-only runtime | Delivered | Torch `2.10.0+cpu`; CUDA build null; 0 CUDA devices; six CPU checks passed `[Actual measured result]` |
| Configuration and contracts | Delivered | Strict YAML validation, `shop_frame_v1`, model manifest v1, event schema 1.0, feature constants |
| File ingestion | Delivered | Streamed upload, SHA-256, decoder probe, duration/resolution/size gates, atomic promotion |
| Webcam ingestion | Delivered baseline | Allowlisted device registration, health probe, one-frame preview, bounded live pipeline, freshness drops |
| Person detection | Delivered development baseline | SSDLite320 MobileNetV3 Large, local hash-verified checkpoint, CPU inference |
| Tracking | Delivered | Own Kalman + IoU/motion + Hungarian tracker, lifecycle and elapsed-time prediction |
| Descriptive events | Delivered | `person_activity_observed`, null score, quality/coverage, no theft vocabulary |
| Evidence | Delivered | VP8 WebM with before/after context and burned-in boxes, thumbnail, SHA-256, Range playback |
| Privacy zones | Delivered configuration baseline | Configured privacy polygon rasterized before detection, preview, and evidence; analysis ROI filters detections by box center; no polygon editor |
| Durable jobs | Delivered | SQLite queue, separate worker, leases, heartbeat, cancellation, interrupted-job fail-closed |
| API | Delivered baseline | `/api/v1`, strict schemas, request IDs, safe errors, auth option, idempotency, ETag, media, reviews, audit, retention |
| Review UI | Delivered | React/TypeScript/Vite, Shop tokens, responsive desktop/narrow, all page states, null-score presentation |
| Reviews | Delivered | Append-only benign/review_needed/inconclusive outcomes, reason codes, correction history |
| MERL temporal research pipeline | Delivered (research) | Drive-only feature extraction, causal TCN training, test evaluation, VIA annotation round-trip; see `docs/data-collection.md` |
| Shelf-action research mode | Delivered (research) | `merl_shelf_actions_research_v1` profile: upload → MERL features (bit-identical to training) → causal TCN → episodes → evidence → reviewer action feedback; job timeline UI; Research insights page. See `docs/final-report.md` |
| Tests | Delivered | 151 backend, 61 frontend, production build, browser checks `[Actual measured result]` |
| Documentation | Delivered | README, architecture, API, models, data, evaluation, runbook, this status |

## Phase status

| Phase | Status | Notes |
| --- | --- | --- |
| 0. Environment, contracts, CPU proof | Complete | CPU verification, strict config, manifests, migrations |
| 1. File ingestion, sampling, person detection, tracking | Complete baseline | One real file run; thresholds not tuned |
| 2. Descriptive policy, quality, evidence | Complete baseline | Null score; evidence generation measured |
| 3. Durable jobs, API, review UI | Complete baseline | Functional E2E; auth/retention gaps below |
| 4. Retail detection | Blocked | No product bounding-box labels (MERL has action intervals only), trained checkpoint, or evaluation |
| 5. Pose | Blocked | No installed/approved checkpoint; CPU feasibility unmeasured |
| 6. Temporal features and TCN | **Research model integrated (research mode)** | MERL Shopping causal TCN: test frame mAP 0.592 (chance 0.154, no-context baseline 0.450), segment mAP@0.5 0.289 (0.304 with the validation-chosen runtime policy). Enabled only for the `merl_shelf_actions_research_v1` file profile; `development_only`, uncalibrated, shelf interactions only |
| 7. RTSP and multi-camera | Deferred | Explicitly disabled; no credential or stream adapter |
| 8. Exports, deletion-status resource, legal hold | Not implemented | Deletion is a local tombstone; no export pipeline |
| 9. Commercial hardening | Not started | Packaging, service units, multi-user roles, formal accessibility/security review |

## Conflicts and deviations

| Topic | Source expectation | Implemented decision | Why / impact |
| --- | --- | --- | --- |
| Torch versions | PRD named Torch 2.7 / TorchVision 0.22 as a conservative reference `[Publisher metadata]` | Torch 2.10.0+cpu / TorchVision 0.25.0+cpu | The local CPU wheels were available and verified; manifests record the actual pair. Re-validate if pinning to the older pair |
| Authentication | PRD says privileged auth is mandatory | Bearer auth is implemented but opt-in via `SHOP_API_TOKEN`; default is open on loopback | Local single-user convenience. Unsuitable for a shared host; see gap below |
| API deletion status | PRD lists `GET /deletions/{id}` and 409 holds | Event DELETE returns a tombstone; upload DELETE has an active-job hold; no separate deletion resource or legal hold | Keeps the local baseline small; PRD endpoint remains unimplemented |
| Exports | PRD lists `POST /exports` | Not implemented | No export scope/approval workflow exists |
| Privacy masks/ROI | PRD expects configurable masks and zone checks on preview | Config-defined polygon is applied in preview, detection, and evidence; ROI filters detections by center; no drawing UI or preview overlay | Application is verified by tests; editing still requires the config file. Default mask is empty, so operators must configure it |
| Review score | PRD/ML describe future calibrated scores | Baseline always stores `null` | No calibration/data. UI renders "Not scored", never 0 |
| Dataset acquisition | Data guide names own footage as the primary source, public sets as gated auxiliaries | MERL Shopping (not in the guide) acquired to Google Drive as the first training set; COCO evaluation pack downloaded then removed by the owner | Unblocks temporal-model research. Research/educational licence keeps any model trained on it `development_only` |
| Evidence codec | No system FFmpeg/FFprobe on host | OpenCV bundled FFmpeg; VP8 WebM; `avc1` unsupported | Playback works in Chromium; format is reviewer context, not a forensic master |
| UI concept | Brand concept contains illustrative numbers | UI uses database/API values only; no fabricated incidents | Concept image is not a measured deployment |
| Browser verification | In-app browser unavailable (`[browser.disconnected]`) | Playwright headless Chromium, desktop 1440×1000 and narrow 390×844 | Real browser engine, automated screenshots; not the in-app browser |

## Missing artifacts and blocked dependencies

- **Commercial person checkpoint rights.** The COCO-derived TorchVision weights are `development_only`; commercial release approval is unresolved.
- **Approved retail detector.** No trained artifact, no visible-product/bag/basket labels, no rights decision.
- **Pose checkpoint.** Reference manifest only; checkpoint absent; CPU feasibility unknown.
- **Temporal model.** The MERL research checkpoint is integrated for file uploads in research mode (`merl_features_v1`, not `shop_features_v1`). Still missing: calibration, own-footage evaluation (protocol ready in `docs/own-evaluation.md`), webcam/live support.
- **Dataset manifests.** MERL Shopping has a dataset manifest (rights, MD5 per file, class mapping) and a split manifest. A Shop-specific own-footage dataset and its consent/purpose record do not exist.
- **Store acceptance data.** No approved target-store footage, benign baseline, supported camera views, or action annotations.
- **Privacy zone authoring.** Frames are masked from config, but there is no polygon editor, preview overlay, or per-source zone validation UI.
- **Exports and legal hold.** No export scope, approval, encryption, or hold workflow.
- **RTSP.** No adapter, credential store, reconnect policy, or timestamp gate.
- **Multi-user roles.** Only a local reviewer allowlist; no admin/auditor/reviewer separation.
- **Packaging.** No systemd unit, container, wheelhouse, or installer; commands are manual.
- **Brand masters.** The supplied brand kit states logo vector masters, reversed artwork, and an app-icon master are not included.
- **Formal evaluations.** No mAP, event PR, calibration, false-alert-per-hour, thermal soak, or multi-hour soak report.
- **Container metadata.** Uploaded `content_type` may be `application/octet-stream`; the decoder probe is authoritative, not the MIME header.

## Known functional gaps

1. **Auth is not mandatory by default.** A shared host needs `SHOP_API_TOKEN`, firewall rules, and a reverse proxy/TLS decision. Reviewer identity is an allowlist, not full role-based authorization.
2. **Metadata/audit retention is declared, not enforced.** `storage.metadata_days` and `storage.audit_days` are exposed in config but no expiry job applies them. Only evidence expiry is implemented.
3. **Privacy zones have no editor or preview overlay.** A configured polygon is applied before detection, preview encoding, and evidence annotation, and an ROI limits analyzed detections. The default config is empty, and operators must edit YAML directly. Verify the polygon on the preview before trusting it in a privacy-sensitive store.
4. **No legal hold.** Event deletion has no hold state; upload deletion only checks active jobs.
5. **No exports or deletion status resource.** `POST /exports` and `GET /deletions/{id}` from the PRD are absent.
6. **No explicit `GET /sources/{id}` ETag response body version field** beyond the `ETag` header. PATCH is enforced.
7. **Review state is denormalized.** It is updated in the review transaction, but bulk/manual DB edits could desynchronize it; no reconciliation job exists.
8. **Live evidence is bounded and lossy.** JPEG ring/buffer limits can drop or truncate context under load; reasons are recorded but not every drop is individually auditable.
9. **File evidence is post-analysis and slow.** One 25 s clip took ~9.7 s on this host; no queue/backpressure policy beyond job concurrency exists.
10. **No worker auto-migration.** Starting the worker against a stale schema fails; this is deliberate but operationally sharp.
11. **No worker crash replay.** Interrupted jobs fail closed with `WORKER_INTERRUPTED`; a manual new job is required.
12. **One local store.** `store_id` is constant; no multi-store scoping or object-level authorization beyond local deployment.
13. **No automatic retention scheduler.** Retention is operator-triggered via API.
14. **No OpenAPI contract-diff CI.** OpenAPI is generated from typed models, but no CI comparison against the PRD table exists.
15. **No packaging/CI pipeline.** Tests are run manually; no GitHub Actions/workflow or wheelhouse lock file beyond pins.
16. **Accessibility is verified structurally, not formally.** 44 px controls, labels, focus styles, and no-overflow checks passed; no screen-reader/WCAG audit.
17. **Person detection accuracy is unmeasured.** The benchmark reports pipeline latency/coverage, not precision/recall/mAP.

## Actual checks run

`[Actual measured result]`:

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest tests -q` | 84 passed; one Starlette/httpx deprecation warning |
| `cd apps/web && npm test -- --run` | 56 passed across 8 files |
| `cd apps/web && npm run build` | Passed; 57 modules; CSS 49.45 kB; JS 373.81 kB |
| Real file E2E | Upload → queued job → worker → 398/398 samples → 1 event → evidence available → review |
| Measured run | 41,953 ms wall; RTF 0.528; 612,372 KB peak RSS; 108 detections; 16 tracks |
| Evidence playback | HTTP 200/206, browser `readyState = 4`, duration 25.2 s, no media error |
| Browser desktop 1440×1000 | All pages, real data, no horizontal overflow, 44 px controls |
| Browser narrow 390×844 | Queue, detail, upload, settings; cards/forms stack; no overflow |
| Simulated states | Loading skeleton, honest empty, safe 503 error with request reference + retry |
| Console | No unexpected errors; only the deliberately simulated 503 responses |
| CPU verification | Passed all six checks |

The evaluation fixture DB now contains upload `3b0ed65b-7b33-46ad-bb8e-5c7ef73288d4`, job `b8e76e74-c00e-4660-afe4-97dccb667e36`, event `d8ade998-5510-42bf-856c-11962cde081f`, and review `4721608a-bd60-44c9-8d09-4d3d605cbfd4`. The review is marked `inconclusive` and notes that it is an evaluation fixture, not a real incident. The database is local runtime state, not source.

## Next steps

1. Add a polygon editor and preview overlay so operators can author and visually confirm privacy zones without editing YAML; keep the existing mask/ROI application tests.
2. Decide the auth deployment model: mandatory token for anything beyond loopback, plus TLS/reverse-proxy guidance and role separation.
3. Implement metadata/audit expiry and a retention scheduler, or remove the unused settings from the public config until enforced.
4. Improve the MERL model: add a hand/pose signal, calibrate scores on validation, and try a higher step rate for short reaches. Re-evaluate once on the test split.
5. Record consented own footage (`raw/own/`) with bag, visibility-loss and benign labels, the classes MERL cannot supervise; measure person detection and event recall/false alerts on it.
6. Train/validate a retail visible-product detector only after data/rights/CPU gates pass; do not reuse COCO classes.
7. Implement feature extraction and freeze `shop_features_v1` only after pose/product validity semantics are tested; train the TCN under grouped splits and calibrate.
8. Add exports, deletion-status/legal-hold semantics, RTSP reconnect/timestamp handling, and multi-camera scoping.
9. Add CI, an OpenAPI contract diff, a wheelhouse lock, service units, and a formal accessibility/security review.

## Bottom line

The baseline demonstrates a trustworthy local review loop: CPU inference, honest descriptive events, reviewable evidence, append-only human decisions, and safe failure states. It does **not** demonstrate shoplifting detection accuracy, commercial model rights, retail detection, pose, temporal inference, or production-grade privacy administration. Those remain gated and must not be described as delivered.

## Shelf-action research integration — 18 November 2026

`[Actual measured result]`: 151 backend tests and 61 frontend tests pass; production build passes.
Live demo (API + worker) on held-out MERL test video `30_1_crop.mp4`: 728/728 steps, 10 episodes, all
with evidence, RTF 0.80–0.92 including evidence clips. Runtime features are bit-identical to training
features on 4 held-out videos. CPU benchmark: feature extraction + TCN RTF 0.353 (i7-1065G7, 4 threads),
446 MB peak. Error analysis, RetailAction pose-compatibility gate (failed; four blockers) and the own
evaluation protocol are in `docs/final-report.md`. RetailAction remains disabled in the app.

## RetailAction T4 preparation — 18 November 2026

A separate offline `take`/`put`/`touch` clip-presence experiment is prepared under
`ml/retailaction/`, with `notebooks/Shop_RetailAction_T4.ipynb` and the workflow in
`docs/retailaction-t4.md`. It is deliberately incompatible with MERL's 1,201-feature pipeline.
Both source views stay in one example; missing/off-image pose observations and unknown time
have masks; inconsistent metadata is quarantined with explicit quality gates. GPU training
requires an allocated Colab T4 and all Drive archives verified. No GPU training or final test
has been executed by this preparation, and no new runtime model has been enabled. Existing
MERL checkpoints and CPU inference configuration are unchanged.
