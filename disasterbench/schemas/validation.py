"""
Schema validation utilities.

These checks validate structural requirements only. Semantic correctness, such
as label meaning or task meaning, must still go through human review when not
proven by official documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from disasterbench.schemas.sample import CommonSample


@dataclass
class ValidationIssue:
    """One schema validation issue."""

    severity: str
    field: str
    message: str


@dataclass
class ValidationResult:
    """Validation result for one sample or dataset object."""

    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_error(self, field: str, message: str) -> None:
        self.passed = False
        self.issues.append(ValidationIssue("error", field, message))

    def add_warning(self, field: str, message: str) -> None:
        self.issues.append(ValidationIssue("warning", field, message))


def validate_common_sample(sample: CommonSample) -> ValidationResult:
    """Validate required structural fields in a CommonSample."""

    result = ValidationResult(passed=True)

    if not sample.dataset_id:
        result.add_error("dataset_id", "dataset_id is required.")

    if not sample.sample_id:
        result.add_error("sample_id", "sample_id is required.")

    if not sample.media:
        result.add_warning("media", "sample has no media records.")

    for media in sample.media:
        if not media.media_id:
            result.add_error("media.media_id", "media_id is required.")
        if not media.path:
            result.add_error("media.path", "media path is required.")
        if media.width is not None and media.width <= 0:
            result.add_error("media.width", "media width must be positive.")
        if media.height is not None and media.height <= 0:
            result.add_error("media.height", "media height must be positive.")
        if media.bands is not None and media.bands <= 0:
            result.add_error("media.bands", "media bands must be positive.")

    for annotation in sample.annotations:
        if not annotation.annotation_id:
            result.add_error(
                "annotation.annotation_id",
                "annotation_id is required.",
            )
        if annotation.label.canonical_label is None:
            result.add_warning(
                "annotation.label.canonical_label",
                "canonical label is missing; label mapping may require human review.",
            )

    return result
