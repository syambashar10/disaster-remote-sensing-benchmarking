"""
Dataset configuration validation.

Dataset configs are the bridge between raw inspection results and trusted
dataset loading. A config may be drafted automatically, but semantic fields
such as label meanings, task type, metadata meaning, lossy conversions, and
license interpretation must be human-reviewed before being treated as final.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from disasterbench.schemas.verification import VerificationStatus


REQUIRED_CONFIG_FIELDS = [
    "dataset_id",
    "display_name",
    "dataset_root",
    "task_types",
    "native_annotation_types",
]

LIST_FIELDS = [
    "aliases",
    "task_types",
    "disaster_types",
    "native_annotation_types",
    "supported_exports",
    "unsupported_exports",
    "lossy_conversions",
]

DICT_FIELDS = [
    "paths",
    "splits",
    "label_mappings",
    "metadata_mappings",
    "quality_thresholds",
]

SEMANTIC_REVIEW_FIELDS = [
    "task_types",
    "disaster_types",
    "label_mappings",
    "metadata_mappings",
    "supported_exports",
    "unsupported_exports",
    "lossy_conversions",
    "quality_thresholds",
    "license",
]


@dataclass
class ConfigValidationIssue:
    """One dataset config validation issue."""

    severity: str
    field: str
    message: str


@dataclass
class ConfigValidationResult:
    """Validation result for one dataset config."""

    passed: bool = True
    issues: List[ConfigValidationIssue] = field(default_factory=list)
    requires_human_review: bool = False

    def add_error(self, field: str, message: str) -> None:
        self.passed = False
        self.issues.append(ConfigValidationIssue("error", field, message))

    def add_warning(self, field: str, message: str) -> None:
        self.issues.append(ConfigValidationIssue("warning", field, message))

    def add_human_review(self, field: str, message: str) -> None:
        self.requires_human_review = True
        self.issues.append(
            ConfigValidationIssue("human_review_required", field, message)
        )


@dataclass
class DatasetConfig:
    """Normalized dataset config object used by the framework."""

    dataset_id: str
    display_name: str
    dataset_root: str
    aliases: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    disaster_types: List[str] = field(default_factory=list)
    native_annotation_types: List[str] = field(default_factory=list)
    supported_exports: List[str] = field(default_factory=list)
    unsupported_exports: List[str] = field(default_factory=list)
    lossy_conversions: List[str] = field(default_factory=list)
    paths: Dict[str, Any] = field(default_factory=dict)
    splits: Dict[str, Any] = field(default_factory=dict)
    label_mappings: Dict[str, Any] = field(default_factory=dict)
    metadata_mappings: Dict[str, Any] = field(default_factory=dict)
    quality_thresholds: Dict[str, Any] = field(default_factory=dict)
    license: Optional[str] = None
    human_review_status: str = VerificationStatus.NOT_VERIFIED.value
    raw_config: Dict[str, Any] = field(default_factory=dict)


def load_dataset_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a JSON dataset config from disk."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_dataset_config(config: Dict[str, Any]) -> DatasetConfig:
    """
    Convert a config dictionary into a DatasetConfig object.

    This does not prove the config is correct. Run validate_dataset_config_dict
    to check required fields, types, and human review status.
    """

    return DatasetConfig(
        dataset_id=str(config.get("dataset_id", "")),
        display_name=str(config.get("display_name", "")),
        dataset_root=str(config.get("dataset_root", "")),
        aliases=list(config.get("aliases", [])),
        task_types=list(config.get("task_types", [])),
        disaster_types=list(config.get("disaster_types", [])),
        native_annotation_types=list(config.get("native_annotation_types", [])),
        supported_exports=list(config.get("supported_exports", [])),
        unsupported_exports=list(config.get("unsupported_exports", [])),
        lossy_conversions=list(config.get("lossy_conversions", [])),
        paths=dict(config.get("paths", {})),
        splits=dict(config.get("splits", {})),
        label_mappings=dict(config.get("label_mappings", {})),
        metadata_mappings=dict(config.get("metadata_mappings", {})),
        quality_thresholds=dict(config.get("quality_thresholds", {})),
        license=config.get("license"),
        human_review_status=str(
            config.get(
                "human_review_status",
                VerificationStatus.NOT_VERIFIED.value,
            )
        ),
        raw_config=dict(config),
    )


def validate_dataset_config_dict(config: Dict[str, Any]) -> ConfigValidationResult:
    """
    Validate one dataset config dictionary.

    Structural checks can pass automatically. Semantic fields still require
    human review unless the config explicitly records human approval.
    """

    result = ConfigValidationResult()

    for field_name in REQUIRED_CONFIG_FIELDS:
        if field_name not in config:
            result.add_error(field_name, f"Missing required field: {field_name}")

    for field_name in LIST_FIELDS:
        if field_name in config and not isinstance(config[field_name], list):
            result.add_error(field_name, f"{field_name} must be a list.")

    for field_name in DICT_FIELDS:
        if field_name in config and not isinstance(config[field_name], dict):
            result.add_error(field_name, f"{field_name} must be a dictionary.")

    human_review_status = config.get(
        "human_review_status",
        VerificationStatus.NOT_VERIFIED.value,
    )

    valid_status_values = {status.value for status in VerificationStatus}
    if human_review_status not in valid_status_values:
        result.add_error(
            "human_review_status",
            (
                "human_review_status must be one of: "
                f"{sorted(valid_status_values)}"
            ),
        )

    if human_review_status != VerificationStatus.HUMAN_APPROVED.value:
        for field_name in SEMANTIC_REVIEW_FIELDS:
            if field_name in config and config[field_name]:
                result.add_human_review(
                    field_name,
                    (
                        f"{field_name} has semantic meaning and must be "
                        "human-reviewed before being treated as final."
                    ),
                )

    if "label_mappings" in config:
        label_mappings = config["label_mappings"]
        if isinstance(label_mappings, dict):
            for raw_label, mapping in label_mappings.items():
                if isinstance(mapping, dict):
                    if "canonical_label" not in mapping:
                        result.add_human_review(
                            f"label_mappings.{raw_label}",
                            (
                                "Label mapping is missing canonical_label. "
                                "This must be reviewed before export."
                            ),
                        )
                    if mapping.get("review_status") != (
                        VerificationStatus.HUMAN_APPROVED.value
                    ):
                        result.add_human_review(
                            f"label_mappings.{raw_label}.review_status",
                            (
                                "Label mapping is not human-approved. "
                                "Do not treat it as final."
                            ),
                        )

    return result


def validate_dataset_config_file(
    config_path: str | Path,
) -> ConfigValidationResult:
    """Load and validate a dataset config JSON file."""

    config = load_dataset_config(config_path)
    return validate_dataset_config_dict(config)
