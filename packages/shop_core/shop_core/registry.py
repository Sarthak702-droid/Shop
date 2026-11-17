"""Model manifest validation and CPU-only runtime verification.

Loading fails closed. A missing, corrupted, schema-incompatible or unapproved
artifact never becomes random weights and never silently downloads a substitute.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shop_core.config import PROJECT_ROOT, Settings


class ModelUnavailable(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class CpuVerification:
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(item["passed"] for item in self.checks)

    def add(self, name: str, passed: bool, observed: Any, expectation: str) -> None:
        self.checks.append(
            {"name": name, "passed": bool(passed), "observed": observed, "expectation": expectation}
        )


def verify_cpu_runtime() -> CpuVerification:
    """Evidence that the runtime is actually CPU-only, not only 'CUDA unavailable'."""
    import torch
    import torchvision

    result = CpuVerification()
    result.add("torch_build_cuda", torch.version.cuda is None, torch.version.cuda, "torch.version.cuda is None")
    result.add(
        "torch_cuda_available",
        torch.cuda.is_available() is False,
        torch.cuda.is_available(),
        "CUDA is not available",
    )
    result.add(
        "cuda_device_count",
        torch.cuda.device_count() == 0,
        torch.cuda.device_count(),
        "zero CUDA devices",
    )
    cuda_flag = getattr(torch.backends, "cuda", None)
    cuda_compiled = bool(getattr(cuda_flag, "is_built", lambda: False)())
    result.add("torch_cuda_compiled", cuda_compiled is False, cuda_compiled, "CUDA support not compiled")
    tensor = torch.ones(2, 2) @ torch.ones(2, 2)
    result.add("tensor_device", tensor.device.type == "cpu", tensor.device.type, "cpu")
    result.add(
        "torchvision_build",
        "+cpu" in torchvision.__version__,
        torchvision.__version__,
        "CPU wheel suffix",
    )
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelUnavailable(
            "MODEL_MANIFEST_MISSING",
            "Model manifest is missing; analysis capability is unavailable.",
            {"manifest": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)},
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelUnavailable("MODEL_MANIFEST_INVALID", "Model manifest is not valid JSON.") from error
    if payload.get("schema_version") != "shop_model_manifest_v1":
        raise ModelUnavailable(
            "MODEL_SCHEMA_MISMATCH",
            "Model manifest schema version is not supported.",
            {"found": payload.get("schema_version")},
        )
    return payload


@dataclass
class ModelStatus:
    role: str
    model_id: str
    version: str
    enabled: bool
    available: bool
    commercial_status: str
    architecture: str | None
    sha256: str | None
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_public(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "model_id": self.model_id,
            "version": self.version,
            "enabled": self.enabled,
            "available": self.available,
            "capability": "available" if self.available else "analysis_unavailable",
            "commercial_status": self.commercial_status,
            "architecture": self.architecture,
            "sha256": self.sha256,
            "reason": self.reason,
            "details": self.details,
        }


def inspect_model(settings: Settings, toggle_name: str) -> ModelStatus:
    toggle = getattr(settings.models, toggle_name)
    role = {
        "person_detector": "person_detector",
        "retail_detector": "retail_detector",
        "pose": "pose_reference",
        "temporal": "temporal_activity_model",
    }[toggle_name]
    try:
        manifest = load_manifest(settings.model_manifest_path(toggle))
    except ModelUnavailable as error:
        return ModelStatus(
            role=role,
            model_id="unavailable",
            version="unavailable",
            enabled=toggle.enabled,
            available=False,
            commercial_status="unknown",
            architecture=None,
            sha256=None,
            reason=error.message,
            details={"code": error.code, **error.details},
        )

    approval = manifest.get("approval", {})
    commercial_status = approval.get("commercial_status") or manifest.get("commercial_approval", "unknown")
    weights = manifest.get("weights") or {}
    asset_path = weights.get("path")
    expected_sha = weights.get("sha256") or manifest.get("checkpoint_sha256")
    model_id = manifest.get("model_id") or manifest.get("bundle_id") or "unknown"
    version = manifest.get("version") or manifest.get("status") or "unknown"
    evaluation = manifest.get("evaluation") or {}
    manifest_details = {
        "approval_notes": approval.get("notes") or manifest.get("notes"),
        "evaluation_report": evaluation.get("report") or manifest.get("evaluation_report"),
        "measured_on_this_host": evaluation.get("measured_on_this_host", False),
        "feature_schema": manifest.get("feature_schema") or manifest.get("feature_version"),
        "feature_count": manifest.get("feature_count"),
        "sample_rate_hz": manifest.get("sample_rate_hz"),
        "window_samples": manifest.get("window_samples"),
        "class_order": manifest.get("class_order"),
        "dilations": (manifest.get("constructor") or {}).get("dilations") if isinstance(manifest.get("constructor"), dict) else None,
        "training_dataset_manifest": manifest.get("training_dataset_manifest"),
        "license": manifest.get("license"),
        "weights_source": weights.get("source"),
        "preprocessing": manifest.get("preprocessing"),
        "output": manifest.get("output"),
        "constructor": manifest.get("constructor"),
    }

    if not toggle.enabled:
        return ModelStatus(
            role=role,
            model_id=model_id,
            version=version,
            enabled=False,
            available=False,
            commercial_status=commercial_status,
            architecture=manifest.get("architecture"),
            sha256=expected_sha,
            reason="Module is disabled by configuration.",
            details=manifest_details,
        )
    if not asset_path:
        return ModelStatus(
            role=role,
            model_id=model_id,
            version=version,
            enabled=True,
            available=False,
            commercial_status=commercial_status,
            architecture=manifest.get("architecture"),
            sha256=None,
            reason="No approved weight artifact is installed for this module.",
            details={"code": "MODEL_ARTIFACT_ABSENT", **manifest_details},
        )

    path = settings.resolve_path(asset_path)
    if not path.is_file():
        return ModelStatus(
            role=role,
            model_id=model_id,
            version=version,
            enabled=True,
            available=False,
            commercial_status=commercial_status,
            architecture=manifest.get("architecture"),
            sha256=expected_sha,
            reason="Model weight artifact is missing; readiness is degraded.",
            details={"code": "MODEL_ARTIFACT_MISSING", **manifest_details},
        )
    actual_sha = sha256_file(path)
    if expected_sha and actual_sha != expected_sha:
        return ModelStatus(
            role=role,
            model_id=model_id,
            version=version,
            enabled=True,
            available=False,
            commercial_status=commercial_status,
            architecture=manifest.get("architecture"),
            sha256=actual_sha,
            reason="Model artifact hash does not match its manifest; loading refused.",
            details={"code": "MODEL_HASH_MISMATCH", "expected_sha256": expected_sha, **manifest_details},
        )
    if commercial_status not in {"approved", "development_only"}:
        return ModelStatus(
            role=role,
            model_id=model_id,
            version=version,
            enabled=True,
            available=False,
            commercial_status=commercial_status,
            architecture=manifest.get("architecture"),
            sha256=actual_sha,
            reason="Artifact rights are not approved for the configured use.",
            details={"code": "MODEL_RIGHTS_UNAPPROVED", **manifest_details},
        )
    if commercial_status == "development_only" and not settings.runtime.allow_development_models:
        return ModelStatus(
            role=role,
            model_id=model_id,
            version=version,
            enabled=True,
            available=False,
            commercial_status=commercial_status,
            architecture=manifest.get("architecture"),
            sha256=actual_sha,
            reason="Development-only artifact is blocked by configuration.",
            details={"code": "MODEL_DEVELOPMENT_ONLY_BLOCKED", **manifest_details},
        )

    compatible, runtime_reason = _runtime_compatible(manifest)
    if not compatible:
        return ModelStatus(
            role=role,
            model_id=model_id,
            version=version,
            enabled=True,
            available=False,
            commercial_status=commercial_status,
            architecture=manifest.get("architecture"),
            sha256=actual_sha,
            reason=runtime_reason,
            details={"code": "MODEL_RUNTIME_INCOMPATIBLE", **manifest_details},
        )
    return ModelStatus(
        role=role,
        model_id=model_id,
        version=version,
        enabled=True,
        available=True,
        commercial_status=commercial_status,
        architecture=manifest.get("architecture"),
        sha256=actual_sha,
        reason="Artifact hash and runtime schema match. Capability is available.",
        details=manifest_details,
    )


def _runtime_compatible(manifest: dict[str, Any]) -> tuple[bool, str]:
    import torch
    import torchvision

    runtime = manifest.get("runtime") or {}
    expected_torch = runtime.get("torch_build")
    expected_torchvision = runtime.get("torchvision_build")
    if expected_torch:
        expected_pair = ".".join(expected_torch.split(".")[:2])
        actual_pair = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        if expected_pair != actual_pair:
            return False, f"Manifest requires torch {expected_torch}, runtime is {torch.__version__}."
    if expected_torchvision:
        expected_pair = ".".join(expected_torchvision.split(".")[:2])
        actual_pair = ".".join(torchvision.__version__.split("+")[0].split(".")[:2])
        if expected_pair != actual_pair:
            return False, (
                f"Manifest requires torchvision {expected_torchvision}, "
                f"runtime is {torchvision.__version__}."
            )
    if runtime.get("requires_cuda"):
        return False, "Manifest requires CUDA, which this CPU-only profile forbids."
    return True, "compatible"


def host_profile() -> dict[str, Any]:
    import torch
    import torchvision

    cpu_model = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        physical = int(
            next(
                line.split(":")[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("cpu cores")
            )
        )
    except (OSError, StopIteration, ValueError):
        physical = os.cpu_count() or 0
    memory_kb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_kb = int(line.split()[1])
                break
    except OSError:
        pass
    return {
        "os": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cores": os.cpu_count(),
        "physical_cores": physical,
        "memory_total_kb": memory_kb,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": "cpu",
        "cuda_build": torch.version.cuda,
    }
