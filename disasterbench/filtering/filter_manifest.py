"""
Filter manifest records.

A filter manifest explains exactly why samples were included, excluded,
quarantined, or marked for human review after dataset filtering.

This module does not perform filtering by itself. Filter engines will create
FilterManifestEntry records later.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from disasterbench.schemas.verification import VerificationStatus


class FilterDecision(str, Enum):
    """Decision assigned to one sample after filtering."""

    INCLUDED = "included"
    EXCLUDED = "excluded"
    QUARANTINED = "quarantined"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


@dataclass
class FilterReason:
    """One reason explaining a filter decision."""

    code: str
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FilterManifestEntry:
    """Filter decision for one sample."""

    sample_id: str
    decision: FilterDecision
    reasons: List[FilterReason] = field(default_factory=list)
    matched_filters: Dict[str, Any] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    verification_status: str = VerificationStatus.AUTO_VERIFIED.value
    notes: Optional[str] = None

    def requires_human_review(self) -> bool:
        return (
            self.decision == FilterDecision.HUMAN_REVIEW_REQUIRED
            or self.verification_status == VerificationStatus.HUMAN_REVIEW_REQUIRED.value
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        data["reasons"] = [reason.to_dict() for reason in self.reasons]
        return data


@dataclass
class FilterManifest:
    """
    Dataset-level filter manifest.

    This should be written after filtering so the final curated dataset can be
    audited and reproduced.
    """

    dataset_id: str
    filter_config_id: str
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    entries: List[FilterManifestEntry] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_entry(self, entry: FilterManifestEntry) -> None:
        self.entries.append(entry)

    def decision_counts(self) -> Dict[str, int]:
        counts = {decision.value: 0 for decision in FilterDecision}
        for entry in self.entries:
            counts[entry.decision.value] += 1
        return counts

    def included_sample_ids(self) -> List[str]:
        return [
            entry.sample_id
            for entry in self.entries
            if entry.decision == FilterDecision.INCLUDED
        ]

    def excluded_sample_ids(self) -> List[str]:
        return [
            entry.sample_id
            for entry in self.entries
            if entry.decision == FilterDecision.EXCLUDED
        ]

    def requires_human_review(self) -> bool:
        return any(entry.requires_human_review() for entry in self.entries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "filter_config_id": self.filter_config_id,
            "generated_at_utc": self.generated_at_utc,
            "filters_applied": self.filters_applied,
            "decision_counts": self.decision_counts(),
            "requires_human_review": self.requires_human_review(),
            "entries": [entry.to_dict() for entry in self.entries],
            "notes": self.notes,
        }


def write_filter_manifest_json(
    manifest: FilterManifest,
    output_path: str | Path,
) -> Path:
    """Write filter_manifest.json."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest.to_dict(), file, indent=2, ensure_ascii=False)

    return path


def write_filter_manifest_csv(
    manifest: FilterManifest,
    output_path: str | Path,
) -> Path:
    """Write filter_manifest.csv for quick spreadsheet review."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "decision",
                "reason_codes",
                "reason_messages",
                "verification_status",
                "source_files",
                "notes",
            ],
        )
        writer.writeheader()

        for entry in manifest.entries:
            writer.writerow(
                {
                    "sample_id": entry.sample_id,
                    "decision": entry.decision.value,
                    "reason_codes": ";".join(
                        reason.code for reason in entry.reasons
                    ),
                    "reason_messages": ";".join(
                        reason.message for reason in entry.reasons
                    ),
                    "verification_status": entry.verification_status,
                    "source_files": ";".join(entry.source_files),
                    "notes": entry.notes or "",
                }
            )

    return path


def write_standard_filter_manifest_outputs(
    manifest: FilterManifest,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """
    Write standard filter manifest outputs.

    Returns:
        - filter_manifest_json
        - filter_manifest_csv
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = write_filter_manifest_json(
        manifest,
        output_path / "filter_manifest.json",
    )
    csv_path = write_filter_manifest_csv(
        manifest,
        output_path / "filter_manifest.csv",
    )

    return {
        "filter_manifest_json": json_path,
        "filter_manifest_csv": csv_path,
    }
