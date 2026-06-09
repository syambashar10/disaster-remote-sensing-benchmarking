"""
Annotation and supervision records for the common schema.

The schema preserves both original labels and canonical labels. It also allows
unknown future annotation types to be preserved through raw_payload/extensions
instead of being silently dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from disasterbench.schemas.verification import VerificationRecord


class AnnotationType(str, Enum):
    BBOX = "bbox"
    ROTATED_BBOX = "rotated_bbox"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"
    MASK = "mask"
    RASTER_LABEL = "raster_label"
    POINT = "point"
    LINE = "line"
    GEO_BBOX = "geo_bbox"
    GEO_POLYGON = "geo_polygon"
    GRAPH = "graph"
    UNKNOWN = "unknown"


class SupervisionType(str, Enum):
    CLASSIFICATION = "classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    CAPTION = "caption"
    QUESTION_ANSWER = "question_answer"
    INSTRUCTION_RESPONSE = "instruction_response"
    CHANGE_DESCRIPTION = "change_description"
    REPORT_GENERATION = "report_generation"


@dataclass
class LabelRecord:
    """
    Label information for one annotation.

    original_label is the raw dataset label.
    canonical_label is the standardized label after mapping.
    """

    original_label: Optional[str]
    canonical_label: Optional[str]
    taxonomy_name: Optional[str] = None
    taxonomy_version: Optional[str] = None
    mapping_confidence: Optional[float] = None
    verification: Optional[VerificationRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.verification is not None:
            data["verification"] = self.verification.to_dict()
        return data


@dataclass
class AnnotationRecord:
    """Common schema representation of a spatial or raster annotation."""

    annotation_id: str
    annotation_type: AnnotationType
    label: LabelRecord
    representations: Dict[str, Any] = field(default_factory=dict)
    source_annotation_path: Optional[str] = None
    source_annotation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_payload: Optional[Dict[str, Any]] = None
    verification: List[VerificationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["annotation_type"] = self.annotation_type.value
        data["label"] = self.label.to_dict()
        data["verification"] = [record.to_dict() for record in self.verification]
        return data


@dataclass
class SupervisionRecord:
    """Common schema representation of non-spatial supervision."""

    supervision_id: str
    supervision_type: SupervisionType
    content: Dict[str, Any]
    source_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    verification: List[VerificationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["supervision_type"] = self.supervision_type.value
        data["verification"] = [record.to_dict() for record in self.verification]
        return data
