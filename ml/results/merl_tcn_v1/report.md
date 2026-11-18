# MERL TCN error analysis: `tcn-20261117T221342Z`

Checkpoint SHA-256 `af66b3cb78f44bb0…`, best epoch 11. Split (videos): train 60, val 18, test 28.

## Per-class results

| Class | Val frame AP | Test frame AP | Chance | Test P | Test R | Test F1 | Test seg AP (eval) | Test seg AP (runtime) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reach_to_shelf | 0.563 | 0.569 | 0.136 | 0.546 | 0.508 | 0.526 | 0.255 | 0.264 |
| retract_from_shelf | 0.570 | 0.608 | 0.139 | 0.456 | 0.744 | 0.565 | 0.406 | 0.402 |
| hand_in_shelf | 0.516 | 0.552 | 0.098 | 0.363 | 0.761 | 0.492 | 0.261 | 0.243 |
| inspect_product | 0.695 | 0.577 | 0.162 | 0.518 | 0.551 | 0.534 | 0.200 | 0.230 |
| inspect_shelf | 0.671 | 0.653 | 0.234 | 0.607 | 0.716 | 0.657 | 0.321 | 0.383 |

Frame mAP: val 0.603, test 0.592 (chance 0.154). Segment mAP@0.5: val 0.314, test 0.289; runtime episode policy on test 0.304.

Runtime policy chosen on validation: {'min_segment_steps': 2, 'merge_gap_steps': 1, 'val_segment_map': 0.3296}.

## Confusion (test, single-label and background steps, row-normalised)

| GT \ Pred | reach_to_shelf | retract_from_shelf | hand_in_shelf | inspect_product | inspect_shelf | background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reach_to_shelf | 0.40 | 0.11 | 0.18 | 0.07 | 0.17 | 0.07 |
| retract_from_shelf | 0.04 | 0.55 | 0.28 | 0.06 | 0.04 | 0.04 |
| hand_in_shelf | 0.03 | 0.15 | 0.67 | 0.04 | 0.05 | 0.05 |
| inspect_product | 0.03 | 0.08 | 0.17 | 0.51 | 0.11 | 0.09 |
| inspect_shelf | 0.04 | 0.05 | 0.02 | 0.12 | 0.68 | 0.08 |
| background | 0.03 | 0.04 | 0.02 | 0.03 | 0.20 | 0.68 |

## Visibility

Person detected in 60% of test steps.

| Class | AP, person visible | AP, person not detected |
| --- | ---: | ---: |
| reach_to_shelf | 0.6115 | 0.465 |
| retract_from_shelf | 0.6192 | 0.5948 |
| hand_in_shelf | 0.627 | 0.2976 |
| inspect_product | 0.6619 | 0.3096 |
| inspect_shelf | 0.638 | 0.6766 |

## Failure examples (test)

760 false-positive segments (tIoU < 0.1 with any same-class GT); 89% of them overlap a *different* annotated action. 230 missed GT segments; mean person visibility inside misses 0.684.

| Type | Video | Class | Time (s) | Score | Person visible | Note |
| --- | --- | --- | --- | ---: | ---: | --- |
| FP | 33_1_crop.mp4 | inspect_product | 9.6–12.6 | 0.966 | 1.0 | GT here: reach_to_shelf, retract_from_shelf, hand_in_shelf, inspect_product |
| FP | 32_1_crop.mp4 | inspect_product | 37.8–40.4 | 0.949 | 1.0 | GT here: reach_to_shelf, retract_from_shelf, inspect_shelf |
| FP | 32_2_crop.mp4 | reach_to_shelf | 56.6–57.2 | 0.949 | 0.67 | GT here: hand_in_shelf |
| FP | 32_2_crop.mp4 | inspect_product | 31.2–32.8 | 0.936 | 0.62 | GT here: reach_to_shelf, inspect_shelf |
| FP | 28_1_crop.mp4 | inspect_product | 113.0–116.2 | 0.935 | 1.0 | GT here: reach_to_shelf, retract_from_shelf, inspect_shelf |
| FP | 39_1_crop.mp4 | inspect_product | 36.6–37.4 | 0.933 | 1.0 | GT here: retract_from_shelf, inspect_shelf |
| Miss | 29_3_crop.mp4 | hand_in_shelf | 58.8–85.8 | max 0.644 (thr 0.638) | 0.0 | |
| Miss | 29_2_crop.mp4 | inspect_product | 11.4–21.8 | max 0.472 (thr 0.734) | 0.19 | |
| Miss | 27_1_crop.mp4 | inspect_product | 108.2–118.0 | max 0.595 (thr 0.734) | 0.0 | |
| Miss | 32_1_crop.mp4 | inspect_shelf | 24.6–34.2 | max 0.626 (thr 0.623) | 0.31 | |
| Miss | 37_1_crop.mp4 | hand_in_shelf | 55.0–64.2 | max 0.314 (thr 0.638) | 0.02 | |
| Miss | 30_3_crop.mp4 | inspect_product | 29.4–38.0 | max 0.943 (thr 0.734) | 0.49 | |

## Calibration (test, expected calibration error)

reach_to_shelf 0.265, retract_from_shelf 0.183, hand_in_shelf 0.159, inspect_product 0.174, inspect_shelf 0.109. Scores are not probabilities.

## Weakest / strongest test videos (frame mAP)

- 27_1_crop.mp4: mAP 0.417, person visible 41%
- 29_3_crop.mp4: mAP 0.420, person visible 30%
- 31_3_crop.mp4: mAP 0.485, person visible 41%
- 27_2_crop.mp4: mAP 0.714, person visible 82%
- 28_2_crop.mp4: mAP 0.717, person visible 52%
- 38_1_crop.mp4: mAP 0.729, person visible 76%

- Frame metrics: per 5 Hz step. Segment metrics: temporal IoU >= 0.5.
- Thresholds and runtime policy were selected on validation; test is reported, not tuned.
- Scores are uncalibrated observable-action scores, never theft or intent probabilities.
- MERL licence: research and educational use only; the checkpoint is development_only.
