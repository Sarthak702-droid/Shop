"""Annotation in VGG Image Annotator 3 (VIA, BSD-2, a single offline HTML file).

* ``build``: writes one VIA video project per MERL video (pre-filled with the publisher's intervals),
  a blank Shop template for own footage, and the VIA tool itself to ``annotations/via/`` on Drive.
* ``import``: converts edited VIA projects back to ``intervals.csv`` rows for training.

Usage (repository root)::

    .venv/bin/python ml/merl/annotation.py build --via-html /path/to/via_video_annotator.html
    .venv/bin/python ml/merl/annotation.py import project1.json project2.json --out intervals_own.csv

VIA project format 3.1.1: temporal segments are ``metadata`` with ``z=[t0, t1]`` seconds and
``av={"1": "<option id>"}``; the attribute is anchored to ``FILE1_Z2_XY0`` (temporal segment in video).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FPS = 30.0
MERL_CLASSES = ("reach_to_shelf", "retract_from_shelf", "hand_in_shelf", "inspect_product", "inspect_shelf")
SHOP_ACTIONS = ("reach_to_shelf", "product_pickup_observed", "return_to_shelf", "hand_near_bag", "item_visibility_lost")
SHOP_OBJECTS = ("visible_product", "bag", "basket", "hand")
DRIVE_VIA = "annotations/via"
ACTION_AID = "1"
OBJECT_AID = "2"


def _project(name: str, video: str | None, actions: tuple[str, ...], objects: bool) -> dict:
    attribute = {
        ACTION_AID: {
            "aname": "action", "anchor_id": "FILE1_Z2_XY0", "type": 4,
            "desc": "Observable action (temporal segment). Describes activity, never intent or theft.",
            "options": {str(i): a for i, a in enumerate(actions)}, "default_option_id": "",
        }
    }
    if objects:
        attribute[OBJECT_AID] = {
            "aname": "object", "anchor_id": "FILE1_Z1_XY1", "type": 4,
            "desc": "Visible object box in a frame", "options": {str(i): o for i, o in enumerate(SHOP_OBJECTS)},
            "default_option_id": "",
        }
    files, views, vids = {}, {}, []
    if video:
        files["1"] = {"fid": "1", "fname": video, "type": 4, "loc": 1, "src": ""}
        views["1"] = {"fid_list": ["1"]}
        vids = ["1"]
    return {
        "project": {"pid": "__VIA_PROJECT_ID__", "rev": "__VIA_PROJECT_REV_ID__", "rev_timestamp": "__VIA_PROJECT_REV_TIMESTAMP__",
                    "pname": name, "creator": "Shop (scripts derived from VGG Image Annotator)",
                    "created": int(time.time() * 1000), "data_format_version": "3.1.1", "vid_list": vids},
        "config": {"file": {"loc_prefix": {"1": "", "2": "", "3": "", "4": ""}},
                   "ui": {"file_content_align": "center", "file_metadata_editor_visible": True,
                          "spatial_metadata_editor_visible": True, "temporal_segment_metadata_editor_visible": True,
                          "spatial_region_label_attribute_id": OBJECT_AID if objects else "",
                          "gtimeline_visible_row_count": str(len(actions))}},
        "attribute": attribute, "file": files, "metadata": {}, "view": views,
    }


def merl_project(video: str, intervals: list[tuple[int, int, int]]) -> dict:
    """intervals: (class_idx, start_frame, end_frame)."""
    p = _project(f"MERL {video}", video, MERL_CLASSES, objects=False)
    for n, (k, s, e) in enumerate(sorted(intervals, key=lambda r: (r[1], r[0])), 1):
        p["metadata"][f"1_{n:05d}"] = {"vid": "1", "flg": 0, "z": [round(s / FPS, 3), round((e + 1) / FPS, 3)],
                                       "xy": [], "av": {ACTION_AID: str(k)}}
    return p


def project_intervals(project: dict) -> list[dict]:
    """Temporal segments of a VIA 3 project -> rows (video, action, start_s, end_s, start_frame, end_frame)."""
    options = project["attribute"][ACTION_AID]["options"]
    files = {vid: project["file"][view["fid_list"][0]]["fname"] for vid, view in project["view"].items()}
    rows = []
    for meta in project["metadata"].values():
        if len(meta.get("z", [])) != 2 or meta.get("xy"):
            continue  # boxes and single-frame marks are not intervals
        opt = meta.get("av", {}).get(ACTION_AID)
        if opt not in options:
            continue
        t0, t1 = sorted(float(t) for t in meta["z"])
        rows.append({"video": files[meta["vid"]], "action": options[opt], "start_s": round(t0, 3), "end_s": round(t1, 3),
                     "start_frame": int(round(t0 * FPS)), "end_frame": max(int(round(t1 * FPS)) - 1, int(round(t0 * FPS)))})
    return sorted(rows, key=lambda r: (r["video"], r["start_s"], r["action"]))


def build(via_html: Path) -> None:
    from ml.merl import drive

    with tempfile.TemporaryDirectory(prefix="shop-via-") as tmp:
        t = Path(tmp)
        rows = list(csv.DictReader(drive.download("annotations/merl_shopping_v1/intervals.csv", t / "i.csv").open()))
        by_video: dict[str, list[tuple[int, int, int]]] = {}
        for r in rows:
            by_video.setdefault(r["video"], []).append((int(r["merl_class_id"]) - 1, int(r["start_frame"]), int(r["end_frame"])))
        out = t / "via"
        (out / "merl_shopping").mkdir(parents=True)
        for video, ivs in by_video.items():
            (out / "merl_shopping" / f"{video.removesuffix('.mp4')}_via.json").write_text(json.dumps(merl_project(video, ivs)))
        (out / "shop_own_footage_template.json").write_text(json.dumps(_project("Shop own footage", None, SHOP_ACTIONS, objects=True), indent=1))
        shutil.copy(via_html, out / "via_video_annotator.html")
        lic = via_html.parent / "LICENSE"
        if lic.exists():
            shutil.copy(lic, out / "VIA_LICENSE.txt")
        (out / "HOW_TO_ANNOTATE.txt").write_text(HOWTO)
        drive.upload_dir(out, DRIVE_VIA)
    print(f"{len(by_video)} MERL projects + template + VIA tool -> {drive.path(DRIVE_VIA)}")


HOWTO = """Annotating with VIA (VGG Image Annotator 3.0.13, BSD-2, runs offline in any browser)
==================================================================================

Review or correct MERL labels
  1. Download via_video_annotator.html, the video (raw/merl_shopping/videos/xx_yy_crop.mp4) and
     merl_shopping/xx_yy_via.json into one folder.
  2. Open via_video_annotator.html in Firefox or Chrome. Click the folder icon (Open project) and pick
     the JSON. If VIA reports the video file as missing, use "Add local files" and pick the .mp4.
  3. Each action row on the timeline is one MERL class. Drag edges to fix boundaries, press "a" to add
     a segment at the playhead, Backspace to delete the selected one.
  4. Save (disk icon) and upload the JSON back to annotations/via/merl_shopping/.

Label own footage
  1. Open via_video_annotator.html, then Open project -> shop_own_footage_template.json.
  2. Add your video ("+" / Add local files). Mark temporal segments with the "action" attribute
     (reach_to_shelf, product_pickup_observed, return_to_shelf, hand_near_bag, item_visibility_lost).
     Draw boxes for visible_product / bag / basket / hand on key frames.
  3. Save the project as <session>_via.json into annotations/via/own/.
  4. Convert for training:
     .venv/bin/python ml/merl/annotation.py import <session>_via.json --out intervals_own.csv

Rules: describe what is visible. Never label intent, theft, suspicion, identity or protected traits.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--via-html", type=Path, required=True)
    imp = sub.add_parser("import")
    imp.add_argument("projects", nargs="+", type=Path)
    imp.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.via_html)
    else:
        rows = [r for p in args.projects for r in project_intervals(json.loads(p.read_text()))]
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["video", "action", "start_s", "end_s", "start_frame", "end_frame"])
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} intervals -> {args.out}")


if __name__ == "__main__":
    main()
