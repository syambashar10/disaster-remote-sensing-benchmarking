"""
Export capability checker.

This module determines whether a CommonSample can be exported to a target
training/analysis format. It prevents unsafe exports by checking whether the
required annotation representations exist before any exporter writes files.

This checker verifies structural capability only. It does not prove semantic
correctness of labels or whether a lossy conversion is scientifically valid.
Those decisions still require verification gates and human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set

from disasterbench.schemas.annotation import AnnotationRecord
from disasterbench.schemas.media import MediaType
from disasterbench.schemas.sample import CommonSample


class TargetFormat(str, Enum):
    """Supported target export formats."""

    COCO_DETECTION = "coco_detection"
    COCO_SEGMENTATION = "coco_segmentation"
    YOLO_DETECTION = "yolo_detection"
    YOLO_SEGMENTATION = "yolo_segmentation"
    GEOJSON = "geojson"
    RASTER_MASK = "raster_mask"


@dataclass
class ExportCapabilityResult:
    """Result of checking whether a sample can be exported."""

    target_format: TargetFormat
    supported: bool
    available_representations: Set[str] = field(default_factory=set)
    missing_requirements: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_format": self.target_format.value,
            "supported": self.supported,
            "available_representations": sorted(self.available_representations),
            "missing_requirements": self.missing_requirements,
            "reason": self.reason,
        }


# Each target format can be satisfied by one of several alternative
# representation sets. For example, COCO segmentation can be supported by
# polygon or mask annotations.
TARGET_REQUIREMENTS: Dict[TargetFormat, List[Set[str]]] = {
    TargetFormat.YOLO_DETECTION: [{"bbox"}],
    TargetFormat.YOLO_SEGMENTATION: [{"polygon"}],
    TargetFormat.COCO_DETECTION: [{"bbox"}],
    TargetFormat.COCO_SEGMENTATION: [{"polygon"}, {"mask"}],
    TargetFormat.GEOJSON: [{"geo_polygon"}, {"geo_bbox"}],
    TargetFormat.RASTER_MASK: [{"mask"}, {"raster_label"}],
}


def get_annotation_representations(annotation: AnnotationRecord) -> Set[str]:
    """Return representation keys available on one annotation."""

    return set(annotation.representations.keys())


def get_sample_representations(sample: CommonSample) -> Set[str]:
    """
    Return all representation names available in a sample.

    This includes annotation representation keys and media-level mask presence.
    """

    representations: Set[str] = set()

    for annotation in sample.annotations:
        representations.update(get_annotation_representations(annotation))

    for media in sample.media:
        if media.media_type == MediaType.MASK:
            representations.add("mask")

    return representations


def check_sample_export_capability(
    sample: CommonSample,
    target_format: TargetFormat,
) -> ExportCapabilityResult:
    """
    Check whether one CommonSample can support a target export format.

    The result is structural only. Human review may still be required for
    label mapping, lossy conversion approval, and scientific validity.
    """

    available = get_sample_representations(sample)
    requirement_options = TARGET_REQUIREMENTS[target_format]

    for option in requirement_options:
        if option.issubset(available):
            return ExportCapabilityResult(
                target_format=target_format,
                supported=True,
                available_representations=available,
                missing_requirements=[],
                reason=(
                    f"Sample supports {target_format.value} because it has "
                    f"required representation(s): {sorted(option)}."
                ),
            )

    missing_descriptions = [
        " or ".join(sorted(option)) for option in requirement_options
    ]

    return ExportCapabilityResult(
        target_format=target_format,
        supported=False,
        available_representations=available,
        missing_requirements=missing_descriptions,
        reason=(
            f"Sample does not support {target_format.value}. "
            f"Required representation option(s): {missing_descriptions}. "
            f"Available representation(s): {sorted(available)}."
        ),
    )


def check_dataset_export_capability(
    samples: List[CommonSample],
    target_format: TargetFormat,
) -> ExportCapabilityResult:
    """
    Check whether all provided samples support a target format.

    If even one sample lacks the required representation, the dataset-level
    result is unsupported and explains how many samples failed.
    """

    if not samples:
        return ExportCapabilityResult(
            target_format=target_format,
            supported=False,
            available_representations=set(),
            missing_requirements=["at least one sample"],
            reason="No samples were provided for export capability checking.",
        )

    all_available: Set[str] = set()
    failed_count = 0
    missing_seen: Set[str] = set()

    for sample in samples:
        result = check_sample_export_capability(sample, target_format)
        all_available.update(result.available_representations)
        if not result.supported:
            failed_count += 1
            missing_seen.update(result.missing_requirements)

    if failed_count == 0:
        return ExportCapabilityResult(
            target_format=target_format,
            supported=True,
            available_representations=all_available,
            missing_requirements=[],
            reason=(
                f"All {len(samples)} sample(s) support "
                f"{target_format.value}."
            ),
        )

    return ExportCapabilityResult(
        target_format=target_format,
        supported=False,
        available_representations=all_available,
        missing_requirements=sorted(missing_seen),
        reason=(
            f"{failed_count} out of {len(samples)} sample(s) do not support "
            f"{target_format.value}."
        ),
    )
