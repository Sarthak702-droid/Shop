"""Typed HTTP request contracts.

Extra fields are rejected. Client identifiers are references, never authority:
the service resolves and authorizes every upload, source and event itself.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

REVIEW_REASON_CODES = (
    "context_insufficient",
    "activity_benign",
    "occlusion",
    "product_unclear",
    "customer_owned_item",
    "equipment_fault",
    "other",
)


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileJobSource(StrictRequest):
    kind: Literal["file"]
    upload_id: UUID


class WebcamJobSource(StrictRequest):
    kind: Literal["webcam"]
    source_id: UUID


JobSource = Annotated[FileJobSource | WebcamJobSource, Field(discriminator="kind")]


class JobCreateRequest(StrictRequest):
    source: JobSource
    analysis_profile: str = Field(default="cpu_baseline_v1", min_length=1, max_length=100)
    sampling_fps: float = Field(default=5.0, gt=0, le=30, allow_inf_nan=False)
    privacy_mask_version: str = Field(default="zones_default_v1", min_length=1, max_length=100)
    retain_evidence: bool = True


class ReviewCreateRequest(StrictRequest):
    decision: Literal["benign", "review_needed", "inconclusive"]
    reason_codes: list[Literal[
        "context_insufficient",
        "activity_benign",
        "occlusion",
        "product_unclear",
        "customer_owned_item",
        "equipment_fault",
        "other",
    ]] = Field(default_factory=list, max_length=8)
    notes: str = Field(default="", max_length=2000)
    supersedes_review_id: UUID | None = None

    @model_validator(mode="after")
    def _reason_required(self) -> "ReviewCreateRequest":
        if not self.reason_codes and not self.notes.strip():
            raise ValueError("provide at least one reason code or a short note")
        return self


class SourceCreateRequest(StrictRequest):
    camera_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    label: str | None = Field(default=None, min_length=1, max_length=100)


class SourcePatchRequest(StrictRequest):
    label: str = Field(min_length=1, max_length=100)
