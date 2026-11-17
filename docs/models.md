# Model registry and manifests

Every analysable artifact is declared by a versioned manifest under `ml/manifests/`, validated at runtime, and exposed through `/api/v1/models` and `/api/v1/capabilities`. Loading fails closed. A missing, corrupted, mismatched, or unapproved artifact is reported as unavailable; random weights are never substituted for a real checkpoint, and no code path downloads weights at startup.

Provenance labels: `[Publisher metadata]`, `[Proposed target]`, `[Illustrative assumption]`, `[Actual measured result]`.

## Manifest contract

Schema: `shop_model_manifest_v1`. Required or meaningful fields:

| Field | Purpose |
| --- | --- |
| `model_id`, `version`, `role` | Stable identity and capability role |
| `architecture`, `constructor` | Exact class and constructor arguments; prevents shape drift |
| `weights.path`, `weights.sha256`, `weights.bytes`, `weights.source` | Local artifact and provenance |
| `preprocessing` | Colour, layout, resize, normalization, coordinate output |
| `output` | Class count/order, background/person indices, coordinate convention |
| `runtime` | CPU requirement, expected Torch/TorchVision builds |
| `feature_schema`, `training_dataset_manifest` | Input contract and training lineage hooks |
| `license` | Code, weights, and source-media rights, kept separate |
| `approval` | Commercial status, release approval, approver, notes |
| `evaluation` | Report reference, measured-on-this-host flag, metrics |

Runtime checks in `shop_core.registry`:

1. Manifest exists and is valid JSON with the supported schema.
2. Toggle is enabled in `configs/shop.yaml`.
3. Weight path exists and SHA-256 matches.
4. Commercial status is `approved` or `development_only`; `development_only` also requires `runtime.allow_development_models: true`.
5. Torch/TorchVision major/minor pair matches the manifest; CUDA-requiring artifacts are rejected.
6. Only then is the artifact marked available.

Any failure returns a safe reason and code (`MODEL_ARTIFACT_MISSING`, `MODEL_HASH_MISMATCH`, `MODEL_RIGHTS_UNAPPROVED`, `MODEL_RUNTIME_INCOMPATIBLE`, etc.). `/health/ready` fails when the person detector is unavailable.

## Current model status

| Role | Manifest | Enabled | Available | Rights | Evaluation |
| --- | --- | --- | --- | --- | --- |
| Person detector | `person_ssdlite320_coco_v1.json` | Yes | Yes on this host | `development_only`; commercial release **not approved** | Hash/runtime checked; no retail-person accuracy claim |
| Retail detector | `retail_detector_untrained.json` | No | No | `not_approved`; no artifact | Untrained; no checkpoint |
| Pose reference | `pose_keypointrcnn_unavailable.json` | No | No | `not_approved`; checkpoint not installed | CPU feasibility unmeasured |
| Temporal TCN | `temporal_tcn_untrained.json` | No | No | `pending`; no artifact | Architecture reference only |
| MERL action TCN (research) | `merl_action_tcn_dev.json` | No | Checkpoint on Drive | `development_only` (MERL research licence) | Test frame mAP 0.592, segment mAP@0.5 0.289 on 28 MERL videos `[Actual measured result]` |

`commercial_status` is a manifest declaration, not an accuracy or legal conclusion. The API also states that evaluation is unmeasured unless `details.measured_on_this_host` is true.

## Person detector

- Model: TorchVision `ssdlite320_mobilenet_v3_large`, person role only.
- Weights: local COCO-derived checkpoint, expected at `var/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth`.
- SHA-256: `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`.
- Bytes: 14 069 355 `[Publisher metadata]`.
- Source: `https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth` `[Publisher metadata]`.
- Constructor: `num_classes=91`, `weights=None`, `weights_backbone=None`, `trainable_backbone_layers=6`, `FrozenBatchNorm2d`. The local state dict is loaded explicitly, so the constructor cannot download weights.
- Preprocessing: BGR→RGB, HWC uint8→CHW float32 / 255, TorchVision internal resize to 320×320; output boxes are original-image pixels.
- Runtime: Torch `2.10.0+cpu`, TorchVision `0.25.0+cpu`, device `cpu`, no CUDA.
- Detection threshold: 0.50 `[Proposed target]`; not tuned on an approved validation split.
- COCO label index: person = 1, background = 0.
- Published architecture size: approximately 3.44 M parameters `[Publisher metadata]`; no throughput claim is derived from that number.
- Commercial status: `development_only`, `approved_for_commercial_release: false`. COCO annotation rights (CC BY 4.0) differ from image rights, and TorchVision requires users to determine pretrained-weight permissions. Commercial deployment needs a completed checkpoint and a dataset rights review.

The measured CPU result in [Evaluation](evaluation.md) is a pipeline benchmark on one OpenCV sample, not a person-detection accuracy result.

## Why retail detection is disabled

The proposed role is visible product, bag, and shopping basket detection using a separately trained SSDLite320 instance. No such checkpoint exists. A generic COCO checkpoint must not be presented as a retail detector, and boxes around bags or products cannot prove concealment or ownership. Enablement requires:

- An approved, consented/labelled retail dataset with train/validation/test separation and a rights decision.
- A trained checkpoint with class order frozen: `background`, `visible_product`, `bag`, `shopping_basket`.
- Measured small-object precision/recall, bag/basket confusion, and calibration on the target camera views.
- Association quality and downstream event false-alert evaluation.
- A manifest with hash, source, license, approval, and evaluation report.

## Why pose is disabled

`keypointrcnn_resnet50_fpn` is recorded only as an offline reference. The checkpoint is not installed, CPU feasibility is unmeasured, and live pose is off. Wrist proximity to a bag or wrist motion near a shelf cannot establish contact, concealment, intent, or theft. If pose is enabled later, it must use the fixed COCO 17-keypoint order, per-person crops, a quality gate, and a feature version tied to the pose artifact hash.

## Temporal TCN reference

The proposed temporal activity model is a **causal** TCN, not a generic pretrained theft classifier.

| Property | Value |
| --- | --- |
| Feature version | `shop_features_v1` |
| Input | 145 features × 29 samples |
| Sampling | 5 Hz |
| Timestamp span | (29−1)/5 = 5.6 s |
| Architecture | 3 residual blocks, 2 causal convolutions per dilation |
| Dilations | 1, 2, 4 |
| Kernel size | 3 |
| Hidden channels | 32 |
| Receptive field | 1 + 2×(3−1)×(1+2+4) = 29 samples |
| Classes | `reach_to_shelf`, `product_pickup_observed`, `return_to_shelf`, `hand_near_bag`, `item_visibility_lost` |
| Dropout | 0.1 |
| Checkpoint | none |
| Calibration | none |
| Status | `proposed_not_trained` |

Feature slices `[start, end)`:

| Slice | Count | Meaning |
| --- | --- | --- |
| 0–68 | 68 | 17 COCO joints × normalized x, y, confidence, validity |
| 68–102 | 34 | 17 joint velocities × x, y |
| 102–119 | 17 | Velocity validity per joint |
| 119–121 | 2 | Left/right wrist nearest supported-product distance |
| 121–123 | 2 | Distance validity masks |
| 123–125 | 2 | Wrist–product compatibility scores |
| 125–127 | 2 | Compatibility validity masks |
| 127–133 | 6 | Each wrist membership in shelf, bag, basket regions |
| 133–139 | 6 | Region-membership validity masks |
| 139–145 | 6 | `torso_scale_valid`, `valid_joint_fraction`, `product_observation_valid`, clipped track age, clipped dt, time since real person detection |

Missing values are zero-filled only after the associated validity mask is zero. "Unknown" and "observed outside region" must remain distinguishable. Track age is clipped at 30 s and elapsed intervals at 1 s `[Illustrative assumption]`; these bounds must be validated and then frozen with the feature version.

A randomly initialized TCN is never exposed as predictions or probabilities. Enabling it requires approved labelled data, frozen features, training/validation/test splits by recording group, calibration, CPU evaluation, event-level ablation against the descriptive policy, and a manifest with the checkpoint hash and feature-schema hash. Any feature order/shape change requires a new feature version and retraining.

## Model bundle identity

Jobs and events store a bundle string:

```text
ssdlite320_mobilenet_v3_large_person:coco_v1_local_dev|policy:policy_baseline_v1
```

This ties an event to the detector identity/version and policy version. It does not imply that pose, retail, or temporal stages ran. `event.model_bundle` and `event.policy_version` are the authoritative provenance fields for a review.

## Adding a checkpoint safely

1. Obtain the artifact through an approved rights process; record source, license, and intended use.
2. Place it under `var/models/` (inside the project storage root) and compute SHA-256.
3. Add or update a manifest with exact constructor, preprocessing, output, feature schema, and runtime pair.
4. Set `approval.commercial_status` honestly. Use `development_only` only when that is the real permission.
5. Keep the module disabled until data, evaluation, calibration, and CPU gates pass.
6. Run `/api/v1/models` and `/api/v1/health/ready`; verify the hash and reasons.
7. Record the measured evaluation in `evaluation.report` and set `measured_on_this_host` only if it is true.

Do not edit a manifest to claim an accuracy number that was not measured on this host, and do not reuse a COCO checkpoint as a retail or activity classifier.
