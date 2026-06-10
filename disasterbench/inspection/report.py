"""
Inspection report objects and writers.

The inspection report is the first evidence artifact for every dataset.
It records what was automatically inspected, what was detected, what anomalies
were found, and which items still require human review.

This module does not inspect files by itself. Scanner modules will populate
these report objects later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from disasterbench.schemas.verification import (
    VerificationRecord,
    VerificationStatus,
)


class InspectionIssueSeverity(str, Enum):
    """Severity levels for inspection issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    BLOCKED = "blocked"


@dataclass
class InspectionIssue:
    """One issue or observation found during dataset inspection."""

    severity: InspectionIssueSeverity
    field: str
    message: str
    evidence_path: Optional[str] = None

    def is_blocking(self) -> bool:
        return self.severity in {
            InspectionIssueSeverity.ERROR,
            InspectionIssueSeverity.BLOCKED,
        }

    def requires_human_review(self) -> bool:
        return self.severity == InspectionIssueSeverity.HUMAN_REVIEW_REQUIRED

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class InspectionArtifact:
    """A file produced by inspection and used as evidence."""

    name: str
    path: str
    description: str
    artifact_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InspectionReport:
    """Standard inspection report for one dataset."""

    dataset_id: str
    dataset_root: str
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    file_counts: Dict[str, int] = field(default_factory=dict)
    folder_counts: Dict[str, int] = field(default_factory=dict)
    image_properties: Dict[str, Any] = field(default_factory=dict)
    raster_properties: Dict[str, Any] = field(default_factory=dict)
    annotation_types_detected: List[str] = field(default_factory=list)
    class_values_detected: List[str] = field(default_factory=list)
    mask_values_detected: List[Any] = field(default_factory=list)
    metadata_fields_detected: List[str] = field(default_factory=list)
    pairing_summary: Dict[str, Any] = field(default_factory=dict)
    duplicate_summary: Dict[str, Any] = field(default_factory=dict)
    quality_summary: Dict[str, Any] = field(default_factory=dict)

    issues: List[InspectionIssue] = field(default_factory=list)
    artifacts: List[InspectionArtifact] = field(default_factory=list)
    verification: List[VerificationRecord] = field(default_factory=list)

    notes: List[str] = field(default_factory=list)

    def add_issue(
        self,
        severity: InspectionIssueSeverity,
        field: str,
        message: str,
        evidence_path: Optional[str] = None,
    ) -> None:
        self.issues.append(
            InspectionIssue(
                severity=severity,
                field=field,
                message=message,
                evidence_path=evidence_path,
            )
        )

    def add_artifact(
        self,
        name: str,
        path: str,
        description: str,
        artifact_type: str,
    ) -> None:
        self.artifacts.append(
            InspectionArtifact(
                name=name,
                path=path,
                description=description,
                artifact_type=artifact_type,
            )
        )

    def has_blocking_issues(self) -> bool:
        return any(issue.is_blocking() for issue in self.issues)

    def requires_human_review(self) -> bool:
        if any(issue.requires_human_review() for issue in self.issues):
            return True

        return any(
            record.status == VerificationStatus.HUMAN_REVIEW_REQUIRED
            for record in self.verification
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["issues"] = [issue.to_dict() for issue in self.issues]
        data["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        data["verification"] = [record.to_dict() for record in self.verification]
        return data


def write_inspection_report_json(
    report: InspectionReport,
    output_path: str | Path,
) -> Path:
    """Write inspection_report.json."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(report.to_dict(), file, indent=2, ensure_ascii=False)

    return path


def build_inspection_summary_markdown(report: InspectionReport) -> str:
    """Build a human-readable inspection summary."""

    lines = [
        f"# Inspection Summary: {report.dataset_id}",
        "",
        f"- Dataset root: `{report.dataset_root}`",
        f"- Generated at UTC: `{report.generated_at_utc}`",
        f"- Blocking issues: `{report.has_blocking_issues()}`",
        f"- Human review required: `{report.requires_human_review()}`",
        "",
        "## File Counts",
        "",
    ]

    if report.file_counts:
        for key, value in sorted(report.file_counts.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- No file counts recorded yet.")

    lines.extend(["", "## Annotation Types Detected", ""])

    if report.annotation_types_detected:
        for item in sorted(report.annotation_types_detected):
            lines.append(f"- {item}")
    else:
        lines.append("- No annotation types recorded yet.")

    lines.extend(["", "## Class Values Detected", ""])

    if report.class_values_detected:
        for item in sorted(report.class_values_detected):
            lines.append(f"- {item}")
    else:
        lines.append("- No class values recorded yet.")

    lines.extend(["", "## Metadata Fields Detected", ""])

    if report.metadata_fields_detected:
        for item in sorted(report.metadata_fields_detected):
            lines.append(f"- {item}")
    else:
        lines.append("- No metadata fields recorded yet.")

    lines.extend(["", "## Issues", ""])

    if report.issues:
        for issue in report.issues:
            lines.append(
                f"- **{issue.severity.value}** `{issue.field}`: {issue.message}"
            )
    else:
        lines.append("- No issues recorded.")

    lines.extend(["", "## Artifacts", ""])

    if report.artifacts:
        for artifact in report.artifacts:
            lines.append(
                f"- `{artifact.path}` ({artifact.artifact_type}): "
                f"{artifact.description}"
            )
    else:
        lines.append("- No artifacts recorded.")

    lines.append("")
    return "\n".join(lines)


def write_inspection_summary_markdown(
    report: InspectionReport,
    output_path: str | Path,
) -> Path:
    """Write inspection_summary.md."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        build_inspection_summary_markdown(report),
        encoding="utf-8",
    )

    return path


def write_standard_inspection_outputs(
    report: InspectionReport,
    output_dir: str | Path,
) -> Dict[str, Path]:
    """Write standard inspection_report.json and inspection_summary.md."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = write_inspection_report_json(
        report,
        output_path / "inspection_report.json",
    )
    summary_path = write_inspection_summary_markdown(
        report,
        output_path / "inspection_summary.md",
    )

    return {
        "inspection_report_json": json_path,
        "inspection_summary_md": summary_path,
    }
