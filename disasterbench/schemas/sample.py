"""
CommonSample schema.

A CommonSample is the central intermediate representation used by loaders,
converters, validators, filters, exporters, and catalog writers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from disasterbench.schemas.annotation import AnnotationRecord, SupervisionRecord
from disasterbench.schemas.capability import CapabilityRecord
from disasterbench.schemas.media import MediaRecord
from disasterbench.schemas.verification import VerificationRecord


@dataclass
class CommonSample:
    """Schema-centered representation of one dataset sample."""

    dataset_id: str
    sample_id: str
    split: Optional[str] = None
    task_types: List[str] = field(default_factory=list)
    disaster_types: List[str] = field(default_factory=list)
    media: List[MediaRecord] = field(default_factory=list)
    annotations: List[AnnotationRecord] = field(default_factory=list)
    supervision: List[SupervisionRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    capabilities: List[CapabilityRecord] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)
    verification: List[VerificationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["media"] = [item.to_dict() for item in self.media]
        data["annotations"] = [item.to_dict() for item in self.annotations]
        data["supervision"] = [item.to_dict() for item in self.supervision]
        data["capabilities"] = [item.to_dict() for item in self.capabilities]
        data["verification"] = [item.to_dict() for item in self.verification]
        return data

    def annotation_count(self) -> int:
        return len(self.annotations)

    def media_count(self) -> int:
        return len(self.media)

    def has_representation(self, representation_name: str) -> bool:
        return any(
            representation_name in annotation.representations
            for annotation in self.annotations
        )
