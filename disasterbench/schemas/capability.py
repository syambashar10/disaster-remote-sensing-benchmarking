"""
Dataset and sample capability records.

Capabilities describe what a dataset/sample can safely support, such as
bounding-box export, segmentation export, mask export, or GeoJSON export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from disasterbench.schemas.verification import VerificationRecord


class CapabilityName(str, Enum):
    """Known framework capability names."""

    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    RASTER_MASK_EXPORT = "raster_mask_export"
    GEOJSON_EXPORT = "geojson_export"
    COCO_DETECTION_EXPORT = "coco_detection_export"
    COCO_SEGMENTATION_EXPORT = "coco_segmentation_export"
    YOLO_DETECTION_EXPORT = "yolo_detection_export"
    YOLO_SEGMENTATION_EXPORT = "yolo_segmentation_export"
    VISION_LANGUAGE_SUPERVISION = "vision_language_supervision"
    UNKNOWN_EXTENSION_ONLY = "unknown_extension_only"


@dataclass
class CapabilityRecord:
    """One supported or unsupported capability."""

    name: CapabilityName
    supported: bool
    reason: str
    required_representations: List[str] = field(default_factory=list)
    verification: Optional[VerificationRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["name"] = self.name.value
        if self.verification is not None:
            data["verification"] = self.verification.to_dict()
        return data
