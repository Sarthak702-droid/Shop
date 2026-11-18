"""Manifest integrity, rights gates and CPU-only runtime verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shop_core.config import PROJECT_ROOT, redacted_config
from shop_core.contracts import (
    ACTION_LABELS,
    FEATURE_COUNT,
    FEATURE_SAMPLING_HZ,
    FEATURE_VERSION,
    FEATURE_WINDOW_SAMPLES,
)
from shop_core.registry import (
    ModelUnavailable,
    host_profile,
    inspect_model,
    load_manifest,
    sha256_file,
    verify_cpu_runtime,
)

MANIFESTS = PROJECT_ROOT / "ml" / "manifests"


def test_sha256_file_matches_known_digest(storage_root: Path) -> None:
    path = storage_root / "payload.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()


def test_manifest_loader_fails_closed(storage_root: Path) -> None:
    with pytest.raises(ModelUnavailable) as missing:
        load_manifest(storage_root / "missing.json")
    assert missing.value.code == "MODEL_MANIFEST_MISSING"

    broken = storage_root / "broken.json"
    broken.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ModelUnavailable) as invalid:
        load_manifest(broken)
    assert invalid.value.code == "MODEL_MANIFEST_INVALID"

    wrong_schema = storage_root / "wrong-schema.json"
    wrong_schema.write_text(json.dumps({"schema_version": "v0"}), encoding="utf-8")
    with pytest.raises(ModelUnavailable) as schema:
        load_manifest(wrong_schema)
    assert schema.value.code == "MODEL_SCHEMA_MISMATCH"


def test_person_manifest_matches_installed_checkpoint(settings) -> None:
    manifest = load_manifest(MANIFESTS / "person_ssdlite320_coco_v1.json")
    assert manifest["schema_version"] == "shop_model_manifest_v1"
    assert manifest["role"] == "person_detector"
    assert manifest["architecture"] == (
        "torchvision.models.detection.ssdlite320_mobilenet_v3_large"
    )
    assert manifest["constructor"]["weights"] is None
    assert manifest["constructor"]["weights_backbone"] is None
    assert manifest["output"]["person_class_index"] == 1
    assert manifest["output"]["coordinate_convention"].startswith("continuous")
    weights = manifest["weights"]
    installed = PROJECT_ROOT / weights["path"]
    assert installed.is_file()
    assert sha256_file(installed) == weights["sha256"]
    assert installed.stat().st_size == weights["bytes"]
    assert manifest["approval"]["commercial_status"] == "development_only"
    assert manifest["approval"]["approved_for_commercial_release"] is False
    assert manifest["evaluation"]["measured_on_this_host"] is False
    assert manifest["runtime"]["requires_cuda"] is False


def test_temporal_manifest_declares_reference_tcn_contract() -> None:
    manifest = load_manifest(MANIFESTS / "temporal_tcn_untrained.json")
    constructor = manifest["constructor"]
    assert manifest["feature_version"] == FEATURE_VERSION == "shop_features_v1"
    assert manifest["feature_count"] == FEATURE_COUNT == 145
    assert manifest["sample_rate_hz"] == FEATURE_SAMPLING_HZ == 5
    assert manifest["window_samples"] == FEATURE_WINDOW_SAMPLES == 29
    assert manifest["receptive_field_samples"] == 29
    assert constructor["dilations"] == [1, 2, 4]
    assert constructor["convolutions_per_dilation"] == 2
    assert constructor["kernel_size"] == 3
    assert constructor["features"] == 145
    assert constructor["classes"] == len(ACTION_LABELS) == 5
    assert manifest["class_order"] == list(ACTION_LABELS)
    # Receptive field: 1 + 2 * (k - 1) * sum(dilations)
    assert 1 + 2 * (constructor["kernel_size"] - 1) * sum(constructor["dilations"]) == 29
    assert manifest["weight_artifact"]["path"] is None
    assert manifest["commercial_approval"] == "pending"


def test_retail_and_pose_manifests_have_no_unapproved_artifact() -> None:
    retail = load_manifest(MANIFESTS / "retail_detector_untrained.json")
    pose = load_manifest(MANIFESTS / "pose_keypointrcnn_unavailable.json")
    assert retail["weights"]["path"] is None
    assert retail["weights"]["sha256"] is None
    assert retail["approval"]["commercial_status"] == "not_approved"
    assert pose["weights"]["path"] is None
    assert pose["approval"]["commercial_status"] == "not_approved"
    assert pose["feature_schema"] == "shop_features_v1"
    assert pose["evaluation"]["measured_on_this_host"] is False


def test_inspect_person_model_reports_development_only_capability(settings) -> None:
    status = inspect_model(settings, "person_detector")
    assert status.enabled is True
    assert status.available is True
    assert status.commercial_status == "development_only"
    assert status.details["measured_on_this_host"] is False
    assert status.to_public()["capability"] == "available"
    assert "commercial" not in status.reason.lower()


def test_development_only_checkpoint_can_be_blocked_by_configuration(settings) -> None:
    settings.runtime.allow_development_models = False
    status = inspect_model(settings, "person_detector")
    assert status.available is False
    assert status.details["code"] == "MODEL_DEVELOPMENT_ONLY_BLOCKED"
    assert status.to_public()["capability"] == "analysis_unavailable"


def test_hash_mismatch_is_refused(storage_root: Path, settings) -> None:
    manifest = json.loads((MANIFESTS / "person_ssdlite320_coco_v1.json").read_text())
    manifest["weights"]["sha256"] = "0" * 64
    path = storage_root / "mismatch.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    settings.models.person_detector.manifest = str(path)
    status = inspect_model(settings, "person_detector")
    assert status.available is False
    assert status.details["code"] == "MODEL_HASH_MISMATCH"
    assert status.details["expected_sha256"] == "0" * 64


def test_missing_artifact_and_disabled_modules_are_unavailable(settings) -> None:
    manifest = json.loads((MANIFESTS / "person_ssdlite320_coco_v1.json").read_text())
    manifest["weights"]["path"] = "var/models/does-not-exist.pth"
    path = PROJECT_ROOT / "var" / "test-runs" / "missing-artifact.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        settings.models.person_detector.manifest = str(path)
        status = inspect_model(settings, "person_detector")
        assert status.available is False
        assert status.details["code"] == "MODEL_ARTIFACT_MISSING"
    finally:
        path.unlink(missing_ok=True)

    for toggle in ("retail_detector", "pose"):
        status = inspect_model(settings, toggle)
        assert status.enabled is False
        assert status.available is False
        assert status.reason == "Module is disabled by configuration."

    settings.models.temporal.enabled = False
    status = inspect_model(settings, "temporal")
    assert status.available is False
    assert status.reason == "Module is disabled by configuration."


def test_shelf_action_checkpoint_hash_mismatch_fails_closed(settings) -> None:
    manifest = json.loads((MANIFESTS / "merl_action_tcn_dev.json").read_text())
    fake = PROJECT_ROOT / "var" / "test-runs" / "fake-merl.pt"
    path = PROJECT_ROOT / "var" / "test-runs" / "merl-hash.json"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_bytes(b"not a checkpoint")
    manifest["weights"]["path"] = "var/test-runs/fake-merl.pt"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        settings.models.temporal.manifest = str(path)
        status = inspect_model(settings, "temporal")
        assert status.available is False
        assert status.details["code"] == "MODEL_HASH_MISMATCH"
    finally:
        path.unlink(missing_ok=True)
        fake.unlink(missing_ok=True)


def test_cpu_runtime_verification_is_explicit() -> None:
    verification = verify_cpu_runtime()
    assert verification.passed is True
    names = {item["name"] for item in verification.checks}
    assert {
        "torch_build_cuda",
        "torch_cuda_available",
        "cuda_device_count",
        "tensor_device",
        "torchvision_build",
    } <= names
    by_name = {item["name"]: item for item in verification.checks}
    assert by_name["tensor_device"]["observed"] == "cpu"
    assert by_name["cuda_device_count"]["observed"] == 0


def test_host_profile_reports_cpu_and_no_cuda(settings) -> None:
    profile = host_profile()
    assert profile["device"] == "cpu"
    assert profile["cuda_build"] is None
    assert profile["logical_cores"] and profile["logical_cores"] >= 1


def test_redacted_config_hides_absolute_storage_path(settings) -> None:
    payload = redacted_config(settings)
    assert payload["storage"]["root"] == "managed_local_storage"
    assert str(settings.storage.root) not in json.dumps(payload)
    assert payload["has_api_token"] is False
