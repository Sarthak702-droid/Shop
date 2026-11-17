"""Validated application configuration.

All numeric values are proposed starting settings. Validation rejects unsafe or
incoherent combinations instead of silently substituting different values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RuntimeConfig(StrictModel):
    device: Literal["cpu"] = "cpu"
    inference_threads: int = Field(default=4, ge=1, le=64)
    active_jobs: int = Field(default=1, ge=1, le=1)
    queued_jobs: int = Field(default=2, ge=0, le=32)
    allow_development_models: bool = True


class StorageConfig(StrictModel):
    root: Path = Path("var")
    auto_migrate: bool = True
    max_upload_bytes: int = Field(default=2 * 1024**3, ge=1024, le=64 * 1024**3)
    max_upload_duration_ms: int = Field(default=2 * 60 * 60 * 1000, ge=1000, le=24 * 60 * 60 * 1000)
    max_upload_width: int = Field(default=3840, ge=320, le=8192)
    max_upload_height: int = Field(default=2160, ge=240, le=8192)
    min_free_bytes: int = Field(default=1024**3, ge=0)
    evidence_days: int = Field(default=7, ge=1, le=3650)
    metadata_days: int = Field(default=30, ge=1, le=3650)
    audit_days: int = Field(default=90, ge=1, le=3650)

    @field_validator("root")
    @classmethod
    def _resolve_root(cls, value: Path) -> Path:
        value = value.expanduser()
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        resolved = value.resolve()
        if PROJECT_ROOT not in resolved.parents and resolved != PROJECT_ROOT:
            raise ValueError("storage root must stay inside the project directory")
        return resolved


class AnalysisConfig(StrictModel):
    long_edge_px: int = Field(default=640, ge=320, le=1920)
    sampling_fps: float = Field(default=5.0, gt=0, le=15, allow_inf_nan=False)
    person_score_threshold: float = Field(default=0.50, ge=0, le=1)
    tracker_iou_weight: float = Field(default=0.60, ge=0, le=1)
    tracker_motion_weight: float = Field(default=0.40, ge=0, le=1)
    tracker_gate_chi2: float = Field(default=9.4877, gt=0, allow_inf_nan=False)
    tracker_process_noise: float = Field(default=12.0, gt=0, allow_inf_nan=False)
    tracker_measurement_noise: float = Field(default=18.0, gt=0, allow_inf_nan=False)
    tracker_confirm_hits: int = Field(default=3, ge=1, le=10)
    tracker_max_prediction_ms: int = Field(default=1000, ge=0, le=10000)
    tracker_tentative_max_prediction_ms: int = Field(default=250, ge=0, le=10000)
    tracker_remove_after_ms: int = Field(default=2500, ge=0, le=60000)
    max_person_crops: int = Field(default=4, ge=1, le=32)
    live_queue_frames: int = Field(default=2, ge=1, le=8)
    evidence_context_before_ms: int = Field(default=10000, ge=0, le=120000)
    evidence_context_after_ms: int = Field(default=10000, ge=0, le=120000)
    event_min_dwell_ms: int = Field(default=4000, ge=0, le=600000)
    event_gap_split_ms: int = Field(default=3000, ge=0, le=600000)
    max_events_per_job: int = Field(default=200, ge=0, le=5000)

    @model_validator(mode="after")
    def _weights_sum(self) -> "AnalysisConfig":
        total = self.tracker_iou_weight + self.tracker_motion_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("tracker association weights must sum to 1.0")
        if self.tracker_max_prediction_ms < self.tracker_tentative_max_prediction_ms:
            raise ValueError("confirmed prediction horizon must not be shorter than tentative")
        return self


class QualityConfig(StrictModel):
    minimum_valid_fraction: float = Field(default=0.70, ge=0, le=1)


class PolicyConfig(StrictModel):
    version: str = Field(default="policy_baseline_v1", min_length=1, max_length=100)
    score_semantics: Literal["uncalibrated_ranking", "calibrated_action_probability"] | None = None
    high_threshold: float = Field(default=0.80, ge=0, le=1)
    low_threshold: float = Field(default=0.60, ge=0, le=1)
    open_dwell_ms: int = Field(default=1000, ge=0)
    close_dwell_ms: int = Field(default=1500, ge=0)
    cooldown_ms: int = Field(default=10000, ge=0)

    @model_validator(mode="after")
    def _thresholds(self) -> "PolicyConfig":
        if self.low_threshold > self.high_threshold:
            raise ValueError("policy low_threshold must not exceed high_threshold")
        return self


class CameraConfig(StrictModel):
    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    label: str = Field(min_length=1, max_length=100)
    device_index: int = Field(ge=0, le=63)


class SecurityConfig(StrictModel):
    bind_host: Literal["127.0.0.1", "localhost"] = "127.0.0.1"
    audio_enabled: Literal[False] = False
    reviewers: list[str] = Field(default_factory=lambda: ["local-reviewer"], min_length=1)
    approved_cameras: list[CameraConfig] = Field(default_factory=list)

    @field_validator("reviewers")
    @classmethod
    def _reviewers(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item or len(item) > 100 for item in cleaned):
            raise ValueError("reviewer identifiers must be 1-100 characters")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("reviewer identifiers must be unique")
        return cleaned


class ModelToggle(StrictModel):
    enabled: bool
    manifest: str


class ModelsConfig(StrictModel):
    person_detector: ModelToggle
    retail_detector: ModelToggle
    pose: ModelToggle
    temporal: ModelToggle


class ZonesConfig(StrictModel):
    version: str = Field(min_length=1, max_length=100)
    coordinate_space: Literal["normalized_original_image"] = "normalized_original_image"
    analysis_roi: list[list[float]] = Field(default_factory=list)
    privacy_mask: list[list[float]] = Field(default_factory=list)

    @field_validator("analysis_roi", "privacy_mask")
    @classmethod
    def _polygon(cls, value: list[list[float]]) -> list[list[float]]:
        if not value:
            return value
        if len(value) < 3:
            raise ValueError("polygon needs at least three vertices")
        for point in value:
            if len(point) != 2:
                raise ValueError("polygon points must be [x, y]")
            x, y = point
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("polygon coordinates must be inside [0, 1]")
        area = 0.0
        for (x1, y1), (x2, y2) in zip(value, value[1:] + value[:1]):
            area += x1 * y2 - x2 * y1
        if abs(area) < 1e-9:
            raise ValueError("polygon must have non-zero area")
        return value


class Settings(StrictModel):
    schema_version: Literal[1] = 1
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    models: ModelsConfig
    zones: ZonesConfig = Field(default_factory=lambda: ZonesConfig(version="zones_default_v1"))

    config_path: Path | None = Field(default=None, exclude=True)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def uploads_dir(self) -> Path:
        return self.storage.root / "uploads"

    @property
    def evidence_dir(self) -> Path:
        return self.storage.root / "evidence"

    @property
    def thumbnails_dir(self) -> Path:
        return self.storage.root / "thumbnails"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.storage.root / 'shop.db'}"

    @property
    def api_token(self) -> str | None:
        token = os.environ.get("SHOP_API_TOKEN", "").strip()
        return token or None

    def camera(self, key: str) -> CameraConfig | None:
        for camera in self.security.approved_cameras:
            if camera.key == key:
                return camera
        return None

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    def model_manifest_path(self, toggle: ModelToggle) -> Path:
        return self.resolve_path(toggle.manifest)

    def ensure_directories(self) -> None:
        for path in (self.storage.root, self.uploads_dir, self.evidence_dir, self.thumbnails_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path or os.environ.get("SHOP_CONFIG") or PROJECT_ROOT / "configs/shop.yaml")
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(raw)
    settings.config_path = config_path.resolve()
    return settings


def redacted_config(settings: Settings) -> dict:
    """Configuration safe to expose to the local browser.

    Absolute filesystem paths are server internals: the local browser receives a
    stable label instead and never a path it could use to probe the host.
    """
    payload = settings.model_dump(mode="json", exclude={"config_path"})
    payload["storage"]["root"] = "managed_local_storage"
    payload["has_api_token"] = bool(settings.api_token)
    return payload
