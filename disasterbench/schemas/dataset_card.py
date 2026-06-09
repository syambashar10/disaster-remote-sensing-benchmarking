"""
DatasetCard schema.

DatasetCard describes a full dataset or a generated dataset package.
It is different from CommonSample, which describes one sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from disasterbench.schemas.capability import CapabilityRecord
from disasterbench.schemas.verification import VerificationRecord


@dataclass
class DatasetCard:
    """Dataset-level metadata and capability summary."""

    dataset_id: str
    display_name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    source_datasets: List[str] = field(default_factory=list)
    disaster_types: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    native_annotation_types: List[str] = field(default_factory=list)
    derived_annotation_types: List[str] = field(default_factory=list)
    sample_count: Optional[int] = None
    split_counts: Dict[str, int] = field(default_factory=dict)
    class_names_original: List[str] = field(default_factory=list)
    class_names_canonical: List[str] = field(default_factory=list)
    metadata_fields: List[str] = field(default_factory=list)
    capabilities: List[CapabilityRecord] = field(default_factory=list)
    license: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    verification: List[VerificationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = [item.to_dict() for item in self.capabilities]
        data["verification"] = [item.to_dict() for item in self.verification]
        return data
