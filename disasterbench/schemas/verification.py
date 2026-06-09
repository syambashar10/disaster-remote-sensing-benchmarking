"""
Verification status records for the schema-centered framework.

This module separates automatically verifiable facts from human-reviewed
semantic decisions. It is intentionally lightweight and dependency-free so it
can be used across inspection, parsing, schema conversion, filtering, export,
and catalog generation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationStatus(str, Enum):
    """Status assigned to a dataset fact, mapping, conversion, or output."""

    AUTO_VERIFIED = "auto_verified"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    HUMAN_APPROVED = "human_approved"
    NOT_VERIFIED = "not_verified"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class VerificationEvidence:
    """
    Evidence supporting a verification claim.

    Examples:
        - inspection_report.json
        - class_inventory.csv
        - official dataset documentation
        - visual QA image path
        - conversion audit JSON
    """

    evidence_type: str
    description: str
    path: Optional[str] = None
    value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationRecord:
    """
    Verification record for one decision or computed fact.

    Structural facts such as file counts can be AUTO_VERIFIED.
    Semantic facts such as label meaning should normally be HUMAN_REVIEW_REQUIRED
    unless official documentation proves them directly.
    """

    name: str
    status: VerificationStatus
    description: str
    evidence: List[VerificationEvidence] = field(default_factory=list)
    reviewer: Optional[str] = None
    review_notes: Optional[str] = None

    def requires_human_review(self) -> bool:
        return self.status == VerificationStatus.HUMAN_REVIEW_REQUIRED

    def is_usable_without_human_review(self) -> bool:
        return self.status in {
            VerificationStatus.AUTO_VERIFIED,
            VerificationStatus.HUMAN_APPROVED,
        }

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


def auto_verified(
    name: str,
    description: str,
    evidence: Optional[List[VerificationEvidence]] = None,
) -> VerificationRecord:
    """Create an AUTO_VERIFIED record."""

    return VerificationRecord(
        name=name,
        status=VerificationStatus.AUTO_VERIFIED,
        description=description,
        evidence=evidence or [],
    )


def human_review_required(
    name: str,
    description: str,
    evidence: Optional[List[VerificationEvidence]] = None,
) -> VerificationRecord:
    """Create a HUMAN_REVIEW_REQUIRED record."""

    return VerificationRecord(
        name=name,
        status=VerificationStatus.HUMAN_REVIEW_REQUIRED,
        description=description,
        evidence=evidence or [],
    )


def not_verified(name: str, description: str) -> VerificationRecord:
    """Create a NOT_VERIFIED record."""

    return VerificationRecord(
        name=name,
        status=VerificationStatus.NOT_VERIFIED,
        description=description,
    )
