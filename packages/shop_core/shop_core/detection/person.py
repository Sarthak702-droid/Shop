"""Person detection adapter for SSDLite320 MobileNetV3 Large (CPU).

The adapter validates the manifest hash, builds the network with explicitly
disabled pretrained backbones, and loads the approved local state dict. It
never triggers a network download at service startup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from shop_core.config import Settings
from shop_core.contracts import Detection
from shop_core.geometry import clip_box
from shop_core.registry import ModelUnavailable, inspect_model, load_manifest


@dataclass
class DetectionTiming:
    preprocess_ms: float
    detect_ms: float
    person_count: int


class PersonDetector:
    role = "person_detector"

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        model_id: str,
        version: str,
        person_class_index: int,
        threshold: float,
        threads: int,
    ) -> None:
        self._model = model
        self.model_id = model_id
        self.version = version
        self.person_class_index = person_class_index
        self.threshold = threshold
        self.threads = threads
        self.last_timing = DetectionTiming(0.0, 0.0, 0)

    @classmethod
    def load(cls, settings: Settings) -> "PersonDetector":
        status = inspect_model(settings, "person_detector")
        if not status.available:
            raise ModelUnavailable(
                status.details.get("code", "MODEL_UNAVAILABLE"),
                status.reason,
                {"model_id": status.model_id, "commercial_status": status.commercial_status},
            )
        manifest = load_manifest(settings.model_manifest_path(settings.models.person_detector))
        if manifest.get("role") != "person_detector":
            raise ModelUnavailable("MODEL_ROLE_MISMATCH", "Manifest role is not person_detector.")
        output = manifest.get("output") or {}
        class_count = int(output.get("class_count", 91))
        person_index = int(output.get("person_class_index", 1))
        if class_count != 91:
            raise ModelUnavailable(
                "MODEL_SCHEMA_MISMATCH",
                "Person detector manifest class count is not the COCO detector contract.",
                {"class_count": class_count},
            )

        torch.set_num_threads(settings.runtime.inference_threads)
        weights_path = settings.resolve_path(manifest["weights"]["path"])
        # Explicitly disable both pretrained weight paths. TorchVision's default
        # backbone argument can otherwise download ImageNet weights.
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=class_count,
        )
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except Exception as error:  # noqa: BLE001 - converted into a typed model failure
            raise ModelUnavailable(
                "MODEL_LOAD_FAILED",
                "Approved person detector weights could not be loaded into the declared architecture.",
                {"error_type": type(error).__name__},
            ) from error
        model.eval().to("cpu")
        with torch.inference_mode():
            # Shape smoke test. This is not an accuracy measurement.
            probe = model([torch.zeros(3, 320, 320)])
        if not isinstance(probe, list) or "boxes" not in probe[0]:
            raise ModelUnavailable("MODEL_OUTPUT_CONTRACT", "Detector output contract is not understood.")
        return cls(
            model,
            model_id=manifest["model_id"],
            version=manifest["version"],
            person_class_index=person_index,
            threshold=settings.analysis.person_score_threshold,
            threads=settings.runtime.inference_threads,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        started = time.perf_counter()
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("detector expects a BGR HxWx3 frame")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        preprocess_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self._model([tensor.to("cpu")])[0]
        except Exception as error:  # noqa: BLE001 - inference failure must not look like "no people"
            raise RuntimeError("person detector inference failed") from error
        detect_ms = (time.perf_counter() - started) * 1000.0

        height, width = frame_bgr.shape[:2]
        detections: list[Detection] = []
        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            if int(label) != self.person_class_index:
                continue
            if float(score) < self.threshold:
                continue
            try:
                clipped = clip_box([float(v) for v in box], width, height)
            except ValueError:
                continue
            detections.append(
                Detection(
                    box=clipped,
                    score=float(score),
                    class_id=int(label),
                    class_name="person",
                    model_id=self.model_id,
                    model_version=self.version,
                    role="person",
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        self.last_timing = DetectionTiming(preprocess_ms, detect_ms, len(detections))
        return detections


def detection_evidence(detector: PersonDetector) -> dict[str, Any]:
    return {
        "model_id": detector.model_id,
        "version": detector.version,
        "person_class_index": detector.person_class_index,
        "score_threshold": detector.threshold,
        "threads": detector.threads,
        "device": "cpu",
    }
