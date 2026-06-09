"""
Media records for the common schema.

A media record can represent an RGB image, multi-band raster, mask, pre/post
image, or any file used by a dataset sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from disasterbench.schemas.verification import VerificationRecord


class MediaType(str, Enum):
    IMAGE = "image"
    RASTER = "raster"
    MASK = "mask"
    PRE_IMAGE = "pre_image"
    POST_IMAGE = "post_image"
    METADATA = "metadata"
    OTHER = "other"


@dataclass
class MediaRecord:
    """Common schema representation of a media asset."""

    media_id: str
    media_type: MediaType
    path: str
    width: Optional[int] = None
    height: Optional[int] = None
    bands: Optional[int] = None
    dtype: Optional[str] = None
    crs: Optional[str] = None
    transform: Optional[List[float]] = None
    bounds: Optional[List[float]] = None
    gsd: Optional[float] = None
    sensor: Optional[str] = None
    capture_date: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    verification: List[VerificationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["media_type"] = self.media_type.value
        data["verification"] = [record.to_dict() for record in self.verification]
        return data
