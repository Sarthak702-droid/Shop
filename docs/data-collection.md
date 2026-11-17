# Data collection and storage

Last updated: 18 November 2026. Rules follow `Shop_Training_and_Testing_Data_Download_Guide.pdf` and
`Shop_ML_Model_Selection_and_Implementation_Guide.pdf`.

## What data is held

`[Actual measured result]`:

| Dataset | Location | Size | Labels | Licence | Use in Shop |
| --- | --- | --- | --- | --- | --- |
| **MERL Shopping** | Drive `raw/merl_shopping/` | 106 videos, 1.73 GiB | 5,377 action intervals, 5 classes | Research and educational only | Temporal (TCN) training and evaluation for shelf interactions |
| OpenCV `vtest.avi` | Local `var/evaluation/incoming/` | 7.8 MB | none | not cleared | Pipeline smoke test |
| SSDLite320 person checkpoint | Local `var/models/` | 14 MB | (pretrained on COCO) | `development_only` | Person detection, no retraining |
| Own store footage | Drive `raw/own/` (empty) | — | — | must be authorised | Primary commercial source, still to be recorded |

COCO val2017 was downloaded for detector evaluation and later removed by the owner. Its manifest has been deleted.

### MERL Shopping Dataset

Source: <https://www.merl.com/demos/merl-shopping-dataset>. Paper: Singh et al., CVPR 2016 (MERL TR2016-080).

- **Media.** 106 videos (`xx_yy_crop.mp4`, xx = subject 1–41, yy = session 1–3) from a fixed overhead camera in a
  mock grocery store. Each is about 2 minutes, 30 fps, 920×680 (verified on `1_1_crop.mp4`: 3,920 frames).
- **Labels.** `xx_yy_label.mat`, a cell array `tlabs`. `tlabs{k}` is a K × 2 array of start and end **frame numbers at
  30 fps** for action k. The largest label frame across all files is 5,951. The publisher does not state whether frames
  are 0- or 1-based, so they are kept exactly as published.
- **Split** (publisher, grouped by subject): train subjects 1–20 (60 videos), validation 21–26 (18), test 27–41 (28).
- **Licence** (publisher ReadMe, *Legalese*): "Permission to use these data and their documentation without fee for
  research and educational purposes, is hereby granted." Commercial use is **not** granted. Any checkpoint trained on
  MERL is therefore `development_only` with `approved_for_commercial_release: false`.

| `tlabs{k}` | MERL class | Intervals | Relation to Shop `ACTION_LABELS` |
| --- | --- | --- | --- |
| 1 | Reach To Shelf | 1,711 | Same as `reach_to_shelf` |
| 2 | Retract From Shelf | 1,621 | Related to `return_to_shelf`, not equal |
| 3 | Hand In Shelf | 562 | Auxiliary |
| 4 | Inspect Product | 674 | Weak proxy for `product_pickup_observed` |
| 5 | Inspect Shelf | 809 | Auxiliary negative context |

MERL gives **no** supervision for `hand_near_bag` or `item_visibility_lost`, and has no product bounding boxes. Those
still need own footage.

## Google Drive layout

Remote `shopdrive:` (rclone, full Drive scope, so browser-uploaded files are visible) and root folder
`ShopliftingAI-Data/`, shared at
<https://drive.google.com/drive/folders/1yEeTJq500whh0TgA9WS0Mt9pBjJfSm1n>:

```text
ShopliftingAI-Data/
├── README.txt
├── raw/
│   ├── merl_shopping/
│   │   ├── videos/            106 × xx_yy_crop.mp4
│   │   ├── labels/            106 × xx_yy_label.mat
│   │   └── ReadMe.md          publisher ReadMe (licence text)
│   └── own/                   future authorised store footage
├── annotations/
│   └── merl_shopping_v1/intervals.csv   one row per interval, frames + ms + split + Shop mapping
├── splits/merl_shopping_official.json   frozen subject-grouped train/val/test
├── manifests/merl_shopping.json         rights, MD5 of all 212 files, class mapping, counts
├── derived/features/          cached per-window features (training input)
├── evaluation/                predictions and reports
└── checkpoints/               trained weights + manifest
```

Removed as unneeded: `raw/coco`, `raw/meva`, `research/*` (UCF, SKU-110K, RPC, D2S, MOT17, Avenue, Campus),
`quarantine/`, `reports/`, `derived/frames/`. They were all empty, except COCO, which the owner had already deleted.
`drive_setup.sh` recreates the current layout if needed.

## Commands

```bash
scripts/data/status.sh            # local disk + per-folder sizes on Drive
scripts/data/drive_setup.sh       # (re)create the folder tree and README.txt; idempotent
scripts/data/upload_own_footage.sh ~/Videos/session1 2026-11-20_storeA_cam1_actors

# Regenerate annotations / split / manifest from the .mat labels
rclone copy shopdrive:ShopliftingAI-Data/raw/merl_shopping/labels /tmp/merl/labels
rclone md5sum shopdrive:ShopliftingAI-Data/raw/merl_shopping --exclude ReadMe.md > /tmp/merl/md5.txt
.venv/bin/python -I scripts/data/prepare_merl.py /tmp/merl/labels /tmp/merl/out --md5 /tmp/merl/md5.txt
rclone copy /tmp/merl/out shopdrive:ShopliftingAI-Data

# Fresh Drive only: rebuild raw/merl_shopping from the official MERL server (~3.6 GB temp)
scripts/data/fetch_merl_shopping.sh
```

rclone is installed at `~/.local/bin/rclone`. Its shared Google client ID is being retired during 2026. Create your
own client ID (<https://rclone.org/drive/#making-your-own-client-id>) and run
`rclone config update shopdrive client_id=… client_secret=…` to avoid interruption.

## Annotation (VIA)

Tool: **VGG Image Annotator 3.0.13** (`via_video_annotator.html`, BSD-2). It's a single 441 KB HTML file that runs
offline in any browser, with no server, account or install. CVAT and Label Studio need Docker or a Python server
and much more disk. VIA's project format is plain JSON, so the scripts read and write it directly.

On Drive under `annotations/via/`:

| File | Purpose |
| --- | --- |
| `via_video_annotator.html`, `VIA_LICENSE.txt` | The tool |
| `merl_shopping/xx_yy_via.json` (106) | MERL labels pre-loaded as timeline segments, for review or correction |
| `shop_own_footage_template.json` | Shop's 5 actions as temporal segments + `visible_product / bag / basket / hand` boxes |
| `HOW_TO_ANNOTATE.txt` | Step-by-step guide |

Verified in headless Chromium: project `1_1` loads with the video and all 65 segments on five action rows.
Edited projects convert back with `ml/merl/annotation.py import`, and the round-trip is frame-exact
(`tests/test_via.py`).

## Training and evaluation

Everything reads from and writes to Drive. Local files are short-lived temporaries.

```bash
# 1. Features: 106 videos -> derived/features/merl_features_v1/*.npz   (~2 h on this CPU, resumable)
.venv/bin/python ml/merl/features.py

# 2. Train (train split), select on validation -> checkpoints/merl_tcn_v1/<run>/
.venv/bin/python -u ml/merl/train.py --model tcn     2>&1 | tee /tmp/merl_train.log
.venv/bin/python -u ml/merl/train.py --model linear  # no-context baseline

# 3. Test once (subjects 27-41) -> evaluation/merl_tcn_v1/<run>/report.md
.venv/bin/python ml/merl/evaluate.py --run latest
```

| Stage | Detail |
| --- | --- |
| Features `merl_features_v1` (1,201 per step, 5 Hz) | SSDLite backbone ROI around the largest person (672) + global map (480) + person geometry (11) + frame-difference motion over all 6 frames (38) |
| Labels | Step label = MERL interval covering the step's centre frame |
| Model | `shop_core.temporal.tcn.ActionTCN`: causal, 4 residual blocks, dilations 1-2-4-8, 64 channels |
| Loss | Masked BCE with positive-class weights (multi-label: classes overlap) |
| Selection | Early stopping and per-class thresholds on the 18 validation videos only |

### Results

| Model | Run | Test frame mAP | Test segment mAP@0.5 |
| --- | --- | ---: | ---: |
| Chance (class prevalence) | — | 0.154 | — |
| No-context baseline (per-step linear) | `linear-20261117T222354Z` | 0.450 | 0.124 |
| **Causal TCN** (61-step / 12 s receptive field) | `tcn-20261117T221342Z` | **0.592** | **0.289** |

TCN per class on the 28 test videos `[Actual measured result]`:

| Class | Frame AP | Chance | Frame F1 | Segment AP | Segment F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| reach_to_shelf | 0.569 | 0.136 | 0.526 | 0.255 | 0.347 |
| retract_from_shelf | 0.608 | 0.139 | 0.565 | 0.406 | 0.431 |
| hand_in_shelf | 0.552 | 0.098 | 0.492 | 0.261 | 0.244 |
| inspect_product | 0.577 | 0.162 | 0.534 | 0.200 | 0.163 |
| inspect_shelf | 0.653 | 0.234 | 0.657 | 0.321 | 0.298 |

Time context is what makes this work: segment mAP more than doubles over the per-step baseline. Short hand motions
(reach, retract) gain the most.

Limits: one overhead camera in one mock store; the COCO person detector finds a person in only ~55% of steps from this
angle (score ≥ 0.15); there is no pose model; the scores are uncalibrated. The checkpoint is `development_only`
(`ml/manifests/merl_action_tcn_dev.json`) and is **not** enabled in the Shop runtime.

## Own footage still required

MERL covers reaching into shelves and inspecting products from one overhead view. For bag proximity, item visibility
loss, product boxes, other camera angles and a real benign baseline, record authorised footage as the data guide
describes (section 13):

1. Written store permission and consenting adult actors. Store a consent and purpose note in `manifests/`.
2. Record at deployment camera height: pickups, returns, basket and bag use, phone/wallet handling, hand near bag, an
   item becoming obscured, staff restocking, crowding, and **long stretches of normal shopping**.
3. Name sessions `YYYY-MM-DD_store_camera_kind`, upload with `upload_own_footage.sh`, and label in CVAT.
