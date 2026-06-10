"""
Label taxonomy and mapping validation.

This module validates versioned class taxonomy files and dataset-specific label
mappings. It prevents raw dataset labels from being silently treated as trusted
canonical labels without confidence scores and human review status.

A taxonomy file can be drafted automatically from inspection results, but final
semantic approval must come from documentation or human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from disasterbench.schemas.verification import VerificationStatus

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


REQUIRED_TAXONOMY_FIELDS = [
    "taxonomy_id",
    "version",
    "labels",
]

REQUIRED_MAPPING_FIELDS = [
    "source_dataset",
    "original_label",
    "canonical_label",
    "confidence",
    "review_status",
]


@dataclass
class TaxonomyValidationIssue:
    """One validation issue for a label taxonomy."""

    severity: str
    field: str
    message: str


@dataclass
class TaxonomyValidationResult:
    """Validation result for a taxonomy YAML/dict."""

    passed: bool = True
    issues: List[TaxonomyValidationIssue] = field(default_factory=list)
    requires_human_review: bool = False

    def add_error(self, field: str, message: str) -> None:
        self.passed = False
        self.issues.append(TaxonomyValidationIssue("error", field, message))

    def add_warning(self, field: str, message: str) -> None:
        self.issues.append(TaxonomyValidationIssue("warning", field, message))

    def add_human_review(self, field: str, message: str) -> None:
        self.requires_human_review = True
        self.issues.append(
            TaxonomyValidationIssue("human_review_required", field, message)
        )


@dataclass
class LabelMappingRecord:
    """Normalized raw-label to canonical-label mapping."""

    source_dataset: str
    original_label: str
    canonical_label: str
    confidence: float
    review_status: str
    evidence: List[str] = field(default_factory=list)
    notes: Optional[str] = None


@dataclass
class LabelTaxonomy:
    """Normalized label taxonomy object."""

    taxonomy_id: str
    version: str
    labels: Dict[str, Any]
    mappings: List[LabelMappingRecord] = field(default_factory=list)
    description: Optional[str] = None
    raw_taxonomy: Dict[str, Any] = field(default_factory=dict)


def load_taxonomy_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a taxonomy YAML file."""

    if yaml is None:
        raise RuntimeError("PyYAML is required to load taxonomy YAML files.")

    taxonomy_path = Path(path)
    with taxonomy_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        raise TypeError("Taxonomy YAML must contain a dictionary at the top level.")

    return loaded


def normalize_taxonomy_dict(taxonomy: Dict[str, Any]) -> LabelTaxonomy:
    """Convert a taxonomy dictionary into a LabelTaxonomy object."""

    mappings = []

    for mapping in taxonomy.get("mappings", []):
        mappings.append(
            LabelMappingRecord(
                source_dataset=str(mapping.get("source_dataset", "")),
                original_label=str(mapping.get("original_label", "")),
                canonical_label=str(mapping.get("canonical_label", "")),
                confidence=float(mapping.get("confidence", 0.0)),
                review_status=str(
                    mapping.get(
                        "review_status",
                        VerificationStatus.NOT_VERIFIED.value,
                    )
                ),
                evidence=list(mapping.get("evidence", [])),
                notes=mapping.get("notes"),
            )
        )

    return LabelTaxonomy(
        taxonomy_id=str(taxonomy.get("taxonomy_id", "")),
        version=str(taxonomy.get("version", "")),
        labels=dict(taxonomy.get("labels", {})),
        mappings=mappings,
        description=taxonomy.get("description"),
        raw_taxonomy=dict(taxonomy),
    )


def validate_taxonomy_dict(taxonomy: Dict[str, Any]) -> TaxonomyValidationResult:
    """
    Validate a label taxonomy dictionary.

    Structural correctness can be automatically checked. Semantic correctness of
    label meaning requires human approval unless official evidence is recorded.
    """

    result = TaxonomyValidationResult()

    for field_name in REQUIRED_TAXONOMY_FIELDS:
        if field_name not in taxonomy:
            result.add_error(field_name, f"Missing required field: {field_name}")

    labels = taxonomy.get("labels", {})
    if "labels" in taxonomy and not isinstance(labels, dict):
        result.add_error("labels", "labels must be a dictionary.")

    mappings = taxonomy.get("mappings", [])
    if mappings and not isinstance(mappings, list):
        result.add_error("mappings", "mappings must be a list.")
        return result

    valid_status_values = {status.value for status in VerificationStatus}
    seen_mappings: Dict[Tuple[str, str], str] = {}

    for index, mapping in enumerate(mappings):
        mapping_field = f"mappings[{index}]"

        if not isinstance(mapping, dict):
            result.add_error(mapping_field, "Each mapping must be a dictionary.")
            continue

        for field_name in REQUIRED_MAPPING_FIELDS:
            if field_name not in mapping:
                result.add_error(
                    f"{mapping_field}.{field_name}",
                    f"Missing required mapping field: {field_name}",
                )

        source_dataset = str(mapping.get("source_dataset", ""))
        original_label = str(mapping.get("original_label", ""))
        canonical_label = str(mapping.get("canonical_label", ""))
        review_status = str(
            mapping.get("review_status", VerificationStatus.NOT_VERIFIED.value)
        )

        confidence = mapping.get("confidence")
        if not isinstance(confidence, (int, float)):
            result.add_error(
                f"{mapping_field}.confidence",
                "confidence must be a number between 0 and 1.",
            )
        elif confidence < 0 or confidence > 1:
            result.add_error(
                f"{mapping_field}.confidence",
                "confidence must be between 0 and 1.",
            )

        if review_status not in valid_status_values:
            result.add_error(
                f"{mapping_field}.review_status",
                (
                    "review_status must be one of: "
                    f"{sorted(valid_status_values)}"
                ),
            )

        if canonical_label and isinstance(labels, dict):
            if canonical_label not in labels:
                result.add_error(
                    f"{mapping_field}.canonical_label",
                    (
                        f"canonical_label '{canonical_label}' is not defined "
                        "in taxonomy labels."
                    ),
                )

        mapping_key = (source_dataset, original_label)
        if mapping_key in seen_mappings:
            previous_canonical = seen_mappings[mapping_key]
            if previous_canonical != canonical_label:
                result.add_human_review(
                    mapping_field,
                    (
                        f"Ambiguous mapping: raw label '{original_label}' from "
                        f"dataset '{source_dataset}' maps to both "
                        f"'{previous_canonical}' and '{canonical_label}'."
                    ),
                )
        else:
            seen_mappings[mapping_key] = canonical_label

        if review_status != VerificationStatus.HUMAN_APPROVED.value:
            result.add_human_review(
                f"{mapping_field}.review_status",
                (
                    "Label mapping is not human-approved. Do not treat this "
                    "mapping as final."
                ),
            )

        if isinstance(confidence, (int, float)) and confidence < 1.0:
            result.add_human_review(
                f"{mapping_field}.confidence",
                (
                    "Label mapping confidence is below 1.0 and should be "
                    "reviewed before use in final exports."
                ),
            )

        evidence = mapping.get("evidence", [])
        if review_status == VerificationStatus.HUMAN_APPROVED.value and not evidence:
            result.add_warning(
                f"{mapping_field}.evidence",
                (
                    "Mapping is human-approved but has no evidence path or "
                    "review note recorded."
                ),
            )

    return result


def validate_taxonomy_yaml(path: str | Path) -> TaxonomyValidationResult:
    """Load and validate a taxonomy YAML file."""

    taxonomy = load_taxonomy_yaml(path)
    return validate_taxonomy_dict(taxonomy)
