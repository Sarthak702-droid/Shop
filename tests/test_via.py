from ml.merl.annotation import FPS, MERL_CLASSES, SHOP_ACTIONS, _project, merl_project, project_intervals


def test_merl_intervals_roundtrip_through_via_exactly():
    ivs = [(0, 156, 183), (3, 217, 371), (1, 406, 416)]
    project = merl_project("1_1_crop.mp4", ivs)
    assert project["project"]["data_format_version"] == "3.1.1"
    assert project["attribute"]["1"]["anchor_id"] == "FILE1_Z2_XY0"
    rows = project_intervals(project)
    got = sorted((MERL_CLASSES.index(r["action"]), r["start_frame"], r["end_frame"]) for r in rows)
    assert got == sorted(ivs)
    assert all(r["video"] == "1_1_crop.mp4" for r in rows)
    assert rows[0]["start_s"] == round(156 / FPS, 3)


def test_shop_template_has_actions_and_object_boxes_without_theft_vocabulary():
    template = _project("t", None, SHOP_ACTIONS, objects=True)
    assert list(template["attribute"]["1"]["options"].values()) == list(SHOP_ACTIONS)
    assert template["attribute"]["2"]["anchor_id"] == "FILE1_Z1_XY1"
    text = str(template).lower()
    assert "theft" not in text.replace("never intent or theft", "") and "suspect" not in text


def test_boxes_are_not_imported_as_intervals():
    project = merl_project("v.mp4", [(0, 30, 59)])
    project["metadata"]["box"] = {"vid": "1", "flg": 0, "z": [1.0], "xy": [2, 1, 2, 3, 4], "av": {"1": "0"}}
    assert len(project_intervals(project)) == 1
