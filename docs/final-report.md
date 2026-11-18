# Shop: final research report

Last updated: 18 November 2026. Scope: personal/research project, not a commercial release.

Shop is a local, CPU-only review assistant for prerecorded supermarket video. It finds observable
shelf interactions, shows when they happened with an evidence clip, and asks a person to review
them. It never decides that theft occurred.

Provenance labels follow `docs/status.md`. Every number below is `[Actual measured result]` unless
marked otherwise, and each one comes from a file under `ml/results/` produced by the named script.

## 1. Research scope (frozen)

| In scope (v1) | Out of scope (future work) |
| --- | --- |
| Prerecorded video, single fixed camera | RTSP, multi-camera, live action inference |
| CPU-only inference | GPU deployment |
| Five observable shelf actions (MERL classes) | Product detection, bag association, concealment |
| Action timeline, evidence clips, model version, warnings | Calibrated probabilities, alerting |
| Human review with action feedback | Billing, packaging, multi-user roles |
| Measured results and stated limits | Any claim of theft or intent detection |

The existing webcam baseline (person activity) is kept unchanged.

## 2. Architecture

```text
Upload ─► SHA-256 + decode probe ─► SQLite job (profile = merl_shelf_actions_research_v1)
                                          │
                       separate worker process (never inside an HTTP request)
                                          ▼
 decode every frame ─► privacy mask ─► SSDLite320 person box + backbone maps (once per 5 Hz step)
                    └► frame-difference motion over all frames of the step
                                          ▼
              merl_features_v1 (1,201 per step) ─► causal TCN (12 s context) ─► 5 action scores
                                          ▼
     validation thresholds ─► per-class segments (≥2 steps, 1-step gaps bridged)
                            ─► interaction episodes (actions ≤1 s apart, ≤20 s each)
                                          ▼
   events (shelf_interaction_observed, review_score = null) + 640 px VP8 evidence with ±2 s context
   + per-step score timeline (timelines/<job>.json, SHA-256 recorded)
                                          ▼
   React UI: job timeline · per-action summary · warnings · evidence · reviewer action feedback
                                          ▼
   Research insights page: offline evaluation joined with live reviewer precision
```

Key engineering decisions:

- **One feature implementation.** `shop_core/actions/merl_features.py` is used by both training
  (`ml/merl/features.py`) and the worker. On held-out videos the runtime features are
  **bit-identical** to the training features (max abs difference 0.0 on 4 of 4 videos checked).
- **Dedicated detector copy.** Feature extraction lowers SSDLite's score floor to 0.15. It loads its
  own detector instance, so baseline jobs keep their 0.5 threshold.
- **Fail closed.** A missing or hash-mismatched checkpoint, corrupt video, unreadable frame rate or
  model error becomes an explicit job error (`ACTION_MODEL_UNAVAILABLE`, `MODEL_HASH_MISMATCH`,
  `SOURCE_OPEN_FAILED`, `SOURCE_FPS_INVALID`, `TEMPORAL_MODEL_FAILED` …). It is never shown as
  "no actions found".
- **Domain warnings** are computed per job: frame rate ≠ 30 fps, aspect ratio far from 920×680,
  person visible in < 30% of steps, video shorter than the 12 s context. Each episode is also
  flagged when the person is visible for < 40% of its duration.
- **Scores stay uncalibrated** and are labelled `uncalibrated_action_score`. The event review score
  remains `null`.

## 3. Experiments

### 3.1 MERL Shopping causal TCN (integrated)

Data: 106 videos, 41 subjects, one overhead camera. Official subject-grouped split: train 1–20
(60 videos), validation 21–26 (18), test 27–41 (28). Labels at 5 Hz step centres.

Training: `ml/merl/train.py`. Masked class-weighted BCE; early stopping on validation mAP; best epoch 11
(training curves on the Research insights page and in `analysis.json`).

Selection discipline: checkpoint, per-class thresholds and the runtime episode policy were all chosen
on **validation only**. The policy grid (min segment 1–3 steps × gap bridging 0–2) picked min 2,
bridge 1 (validation segment mAP 0.330). Test was then scored once with everything frozen
(`ml/merl/analyze.py`; it reproduces the original `evaluate.py` numbers exactly).

| Class | Val frame AP | Test frame AP | Chance | Test P | Test R | Test seg AP (runtime policy) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reach_to_shelf | 0.563 | 0.569 | 0.136 | 0.546 | 0.508 | 0.264 |
| retract_from_shelf | 0.570 | 0.608 | 0.139 | 0.456 | 0.744 | 0.402 |
| hand_in_shelf | 0.516 | 0.552 | 0.098 | 0.363 | 0.761 | 0.243 |
| inspect_product | 0.695 | 0.577 | 0.162 | 0.518 | 0.551 | 0.230 |
| inspect_shelf | 0.671 | 0.653 | 0.234 | 0.607 | 0.716 | 0.383 |
| **mean** | **0.603** | **0.592** | 0.154 | | | **0.304** |

No-context per-step baseline: frame mAP 0.450, segment mAP 0.124. Temporal context more than
doubles segment mAP.

### 3.2 Error analysis (test)

- **Neighbouring-action confusion dominates.** 28% of `retract_from_shelf` steps are predicted as
  `hand_in_shelf`, and 18% of `reach_to_shelf` steps likewise. 89% of the 760 false-positive segments
  overlap a *different* annotated action. Reach → hand in shelf → retract is one continuous motion,
  and most errors sit on its label boundaries.
- **Short actions hurt localisation.** Median annotated reach/retract lasts 1.2 s (10th percentile
  0.8 s = 4 steps), so a one-step boundary error costs a lot of temporal IoU.
- **Person detection is the main failure driver.** The COCO detector finds a person in only 60% of
  overhead test steps. Without a detection, `hand_in_shelf` AP falls from 0.627 to 0.298 and
  `inspect_product` from 0.662 to 0.310. Weakest test videos (27_1, 29_3: mAP ≈ 0.42) have
  30–41% person visibility; the strongest (38_1: 0.73) have 76%.
- **Missed segments** are mostly long `hand_in_shelf`/`inspect_product` intervals with no person
  detection, or scores that cross the threshold only briefly (fragmented).
- **Calibration.** Expected calibration error 0.11–0.27 per class. Scores are not probabilities.

Failed examples with video and time are listed in `ml/results/merl_tcn_v1/report.md` and on the
Research insights page.

### 3.3 RetailAction (take / put / touch): separate improvement experiment

| Step | State | Evidence |
| --- | --- | --- |
| 1. Archive upload + SHA-256 | validation.tar and test.tar verified on Drive. train.tar (24.3 GiB) verified locally against the publisher hash; the Drive upload was at 68% on 18 Nov | `raw/retailaction/TRANSFER_STATUS.json` |
| 2. Data report | Partial: 17,222 train clips; 29 quarantined (pose/timestamp mismatch); 15 without usable pose. Usable positives: take 15,219 · put 339 · **touch 60** | `var/retailaction/preprocessing_pilot.json` |
| 3. T4 learning pilot | Pending: needs Colab T4 after upload | `notebooks/Shop_RetailAction_T4.ipynb` |
| 4–6. Train, validate, final test | Pending | `docs/retailaction-t4.md` |
| 7. Raw-video pose compatibility | **Failed**; see below | `ml/results/retailaction/pose_compatibility.json` |

`touch` has 254× fewer positives than `take`. Its AP will be weak with wide uncertainty, and labels
are kept separate as planned.

**Pose compatibility gate** (`ml/retailaction/pose_compat.py`, 164,339 supplied joints read from the
verified original archive):

1. COCO-17 pose models do not output `left_hand`, `right_hand`, `top_of_head` or the feet. Coverage is
   79% including derived neck and waist, and the hands are exactly what take/put/touch rely on.
2. 99.8% of supplied coordinates lie on a 1/64 grid with maximum 0.9375. They are heatmap-cell
   positions inside the 600×600 shopper crop, not full-frame pixels.
3. Supplied joint scores reach 1.48 (5.1% above 1.0). They are raw heatmap peaks, while Keypoint R-CNN
   emits unbounded logits.
4. Keypoint R-CNN (ResNet-50 FPN) costs 2,081 ms per 600×600 frame on this CPU (0.48 fps), ten times
   over the 200 ms budget for 5 Hz. This was timed with random weights, so it measures cost only.

Therefore a RetailAction model trained on supplied poses **must not** be enabled in the app. Required
first: a hand-capable pose model (or retraining on the COCO subset), shared person-box-normalised
coordinates with validity masks, measured agreement (PCK) between extracted and supplied poses on
RetailAction's own videos, then retraining. Project completion does not depend on this experiment.

### 3.4 Cross-camera transfer: MERL model on RetailAction views

`ml/retailaction/merl_transfer.py`: 104 clips from the verified RetailAction train archive (take 30,
put 30, touch 14 (all single-label touch clips with complete timestamps), no action 30), both fisheye ceiling views, giving
208 videos from other stores. These were run through the integrated runtime path in two conditions: **as uploaded** (the mp4
declares 6 fps, 32 frames) and **time-corrected** (frames held for their real timestamps at 30 fps). Three of these clips were
also run through the live app, and each raised the frame-rate, aspect-ratio and short-video warnings.

| | As uploaded | Time-corrected |
| --- | ---: | ---: |
| Clip AP, action vs no action (chance 0.712) | 0.680 | 0.673 |
| Episode published: take / put / touch | 62% / 67% / 54% | 28% / 43% / 32% |
| Episode published: no action (false) | 65% | 43% |
| rank0 / rank1 agreement | 53% | 62% |

**Result: the model does not transfer.** Where it fails:

- No transfer: clip-level AP 0.680 (as uploaded) and 0.673 (time-corrected) vs 0.712 for a random scorer.
- Not discriminative: an interaction episode is published for 61% of action clips and 65% of no-action clips (as uploaded).
- Wrong timing: the strongest interaction step falls inside the labelled interval in 14% of action clips, although labels cover 32% of each clip.
- Views disagree: rank0 and rank1 of the same clip agree on 'interaction / no interaction' in only 53% of clips.
- Frame timing breaks the motion features: frames are 0.25 s apart vs 0.033 s in MERL, so global frame-difference energy is 7.4x MERL's; reach_to_shelf fires in 73% of all clips.
- Appearance shift: global backbone features sit 2.70 standard deviations from MERL's training mean vs 0.82 for MERL test (fisheye ceiling cameras, rotated/upside-down shoppers, other stores).
- Re-timing to 30 fps fixes the motion statistics (motion z 1.80 -> 0.63, MERL 0.20) and cuts false episodes to 43%, but AP stays at chance: appearance and viewpoint, not only timing, block transfer.
- Clips are 7.9 s long on average, shorter than the model's 12 s causal context, so most predictions use a partially empty context.
- Person visibility matters: visible <50% of steps AP 0.52 vs chance 0.63 (43 videos); visible >=90% AP 0.81 vs chance 0.70 (100 videos). Only fully visible shoppers carry a weak signal; the overall detector rate (74% of steps vs MERL 56%) hides views where a top-down or rotated shopper is missed.

Implication: Shop's domain warnings are justified, and the MERL model's output on another store's cameras is not evidence
of a shelf interaction. Transfer needs labelled data from the target camera, not just format conversion.
Report: `ml/results/retailaction/merl_transfer.md`.

## 4. CPU benchmark

`ml/merl/runtime_e2e.py` on 3 held-out MERL test videos (397.6 s of footage), Intel i7-1065G7,
4 inference threads, torch 2.10.0+cpu:

| Measure | Value |
| --- | --- |
| Real-time factor, feature extraction + TCN | mean 0.353, max 0.383 (≈2.8× faster than real time) |
| Per 5 Hz step | decode 8.0 ms · detector/backbone 57.0 ms · motion 0.5 ms · TCN 0.023 ms |
| Peak RSS (benchmark process) | 446 MB |
| Model load (TCN + detector, hash-verified) | 225 ms |

Full app path (worker job on held-out `30_1_crop.mp4`, 2 m 25 s, including evidence clips): wall
117–134 s, RTF 0.80–0.92, worker peak RSS ≈ 0.87–1.0 GB. That RSS covers the whole worker
lifetime, with both detectors loaded. Measured while a 26 GB Drive upload competed for the same
machine. Evidence encoding was the largest cost; reducing action evidence to 640 px and ±2 s context
cut it from 146 s to 55 s (16 → 6 s per clip) and peak RSS from 1.6 GB to 1.0 GB.

## 5. Reliability checks

| Failure | Behaviour | Test |
| --- | --- | --- |
| Corrupt / non-video upload | 422 `MEDIA_UNSUPPORTED` at upload; nothing queued | `tests/test_api.py`, verified on the live API |
| Corrupt video reaching the worker | Job `failed` with `SOURCE_*` code, no events | `tests/test_actions.py` |
| Checkpoint missing / hash mismatch | Profile shows unavailable; job creation 503 `ACTION_MODEL_UNAVAILABLE`; worker fails the job with the code | `tests/test_actions.py`, `tests/test_registry.py` |
| Cancellation during extraction | Job `cancelled`, no partial events published | `tests/test_actions.py` |
| Timeline file tampered / missing | 500 integrity error / 410, never silently served | `routes/jobs.py` |
| Webcam with the research profile | 422 `ANALYSIS_PROFILE_SOURCE_UNSUPPORTED` | `tests/test_actions.py` |

## 6. Demo (reproducible)

```bash
scripts/models/fetch_merl_tcn.sh                 # verify checkpoint SHA-256
.venv/bin/uvicorn shop_api.main:app --port 8000  # terminal 1
.venv/bin/python -m shop_worker                  # terminal 2
scripts/demo/run_demo.sh                         # held-out MERL test video 30_1
scripts/demo/run_demo.sh var/evaluation/incoming/vtest.avi   # out-of-domain street clip
```

Recorded run on 18 Nov 2026:

- `30_1_crop.mp4`: 728/728 steps, 10 interaction episodes, all with evidence, person visible 91%,
  no domain warning.
- `vtest.avi` (street scene, 10 fps, no shelves): 1 episode of 0.6 s in 79 s, with the
  `frame_rate_differs` warning raised. The model stays mostly quiet on footage with no shelf
  interaction. This is a single qualitative observation, not a false-alarm rate.

Then open `http://127.0.0.1:8000/review`, review each episode, and answer *Do the predicted actions
match the clip?*. Verdicts feed per-class reviewer precision with Wilson 95% intervals on
**Research insights**.

## 7. Own evaluation set

Protocol and tooling are ready (`docs/own-evaluation.md`, `ml/eval/own_eval.py`). It covers staged
reach, take-and-return, inspect, normal pass, occlusion and low-light recordings, with a grouped
session/person split and consent record enforced. **No recordings exist yet**, so transfer to another
store or camera angle is unmeasured. This is the most important remaining measurement.

## 8. Limitations

- One mock store, one overhead camera, 41 actors. No other viewpoint has measured accuracy.
- Five observable shelf actions only. No product, bag or concealment reasoning; the app never infers them.
- Uncalibrated scores; thresholds tuned on MERL validation.
- The COCO person detector misses the overhead shopper in ~40% of steps.
- MERL is licensed for research and education only; the checkpoint is `development_only`.
- Reviewer precision on own footage starts at n = 0 and needs dozens of reviews before it is informative.

## 9. Future work (priority order)

1. Record and annotate the staged evaluation set; run `own_eval.py`.
2. Replace or fine-tune person detection for overhead views, or add a hand-capable pose signal. This
   is the largest measured error source.
3. Calibrate scores on validation (temperature or isotonic) and re-evaluate once.
4. Finish the RetailAction pipeline (pilot → full training → validation → frozen test), then address
   the four pose-compatibility blockers before any integration.
5. Product detection, bag association and multi-camera support, only with rights-cleared labelled data.

## 10. Completion checklist

| Deliverable | Condition | Status |
| --- | --- | --- |
| Working application | Upload → job → timeline → evidence → review | ✅ Verified on live API + worker |
| Action model | At least one trained model integrated | ✅ MERL TCN (`merl_shelf_actions_research_v1`) |
| Evaluation | Held-out results | ✅ 28-video MERL test; ⏳ own staged set not recorded |
| CPU benchmark | Speed and memory recorded | ✅ `ml/results/benchmarks/cpu_e2e.json` |
| Reliability | Safe errors on corrupt video / model failure | ✅ Tests + live check |
| Reproducibility | Setup, model files, run instructions | ✅ `fetch_merl_tcn.sh`, `run_demo.sh`, runbook |
| Final report | Architecture, experiments, limitations, future work | ✅ This document |
| Demo | Reproducible end to end | ✅ `scripts/demo/run_demo.sh` |
| RetailAction | Separate improvement experiment | ⏳ Gates 3–6 need Colab T4; pose gate failed (documented) |
