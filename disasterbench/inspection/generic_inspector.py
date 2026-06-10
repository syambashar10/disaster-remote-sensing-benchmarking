"""
Generic dataset inspection runner.

This module performs safe structural inspection from a dataset config without
guessing semantic meanings. It checks dataset roots, configured paths, file
counts, extension counts, and flags semantic config fields for human review.

Dataset-specific scanners can later extend this with deeper checks such as
image dimensions, mask unique values, JSON label inventories, duplicate checks,
and visual QA.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from disasterbench.configs import load_dataset_config
from disasterbench.inspection.report import (
    InspectionIssueSeverity,
    InspectionReport,
    write_standard_inspection_outputs,
)
from disasterbench.packaging import create_standard_output_package
from disasterbench.schemas.verification import (
    VerificationRecord,
    VerificationStatus,
    auto_verified,
    human_review_required,
)


IMAGE_LIKE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RASTER_LIKE_EXTENSIONS = {".tif", ".tiff"}
ANNOTATION_LIKE_EXTENSIONS = {".json", ".geojson", ".txt", ".csv", ".xml"}


SEMANTIC_REVIEW_KEYS = {
    "task_type",
    "task_types",
    "problem_type",
    "raw_annotation_type",
    "native_annotation_types",
    "damage_classes",
    "recommended_formats",
    "unsupported_formats",
    "lossy_formats",
    "supported_exports",
    "unsupported_exports",
    "lossy_conversions",
    "license",
    "metadata_columns",
}


SEMANTIC_NESTED_KEYS = {
    "original_mask_values",
    "binary_water_mapping",
    "label_mappings",
    "metadata_mappings",
    "quality_thresholds",
}


PATH_KEY_SUFFIXES = (
    "_path",
    "_folder",
    "_dir",
    "_file",
)


def get_dataset_id(config: Dict[str, Any]) -> str:
    """Return dataset_id with a safe fallback."""

    return str(config.get("dataset_id") or "unknown_dataset")


def get_dataset_root(config: Dict[str, Any]) -> str:
    """
    Return dataset root from either v0.3-style or legacy v0.2-style config.

    v0.3 preferred field: dataset_root
    v0.2 legacy field: dataset_path
    """

    return str(config.get("dataset_root") or config.get("dataset_path") or "")


def is_probable_path_key(key: str) -> bool:
    """Return True if a config key likely stores a file/folder path."""

    key_lower = key.lower()
    return key_lower.endswith(PATH_KEY_SUFFIXES)


def collect_configured_paths(
    config: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Collect configured file/folder paths from known config patterns.

    This supports both:
        - paths: {...}
        - split_structure: {...}
        - nested legacy fields such as sentinel1.image_folder
    """

    paths: List[Dict[str, str]] = []

    def add_path(field: str, value: Any) -> None:
        if isinstance(value, str) and value.strip():
            paths.append({"field": field, "path": value})

    for section_name in ["paths", "split_structure", "splits"]:
        section = config.get(section_name)
        if isinstance(section, dict):
            for key, value in section.items():
                add_path(f"{section_name}.{key}", value)

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                field_name = f"{prefix}.{key}" if prefix else key

                if is_probable_path_key(key):
                    add_path(field_name, nested_value)

                walk(field_name, nested_value)

        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(f"{prefix}[{index}]", item)

    walk("", config)

    # Remove dataset root fields because they are checked separately.
    filtered = []
    seen = set()

    for item in paths:
        if item["field"] in {"dataset_root", "dataset_path"}:
            continue

        key = (item["field"], item["path"])
        if key not in seen:
            seen.add(key)
            filtered.append(item)

    return filtered


def resolve_config_path(dataset_root: Path, configured_path: str) -> Path:
    """Resolve absolute or dataset-root-relative configured paths."""

    path = Path(configured_path)

    if path.is_absolute():
        return path

    return dataset_root / path


def add_semantic_review_flags(
    report: InspectionReport,
    config: Dict[str, Any],
) -> None:
    """
    Add human-review-required records for semantic config fields.

    These are not errors. They are reminders that semantic truth must come from
    official documentation or mentor/human approval.
    """

    for key in sorted(SEMANTIC_REVIEW_KEYS):
        value = config.get(key)
        if value:
            report.verification.append(
                human_review_required(
                    name=f"semantic_config_field:{key}",
                    description=(
                        f"Config field '{key}' contains semantic meaning and "
                        "must be confirmed from documentation or human review."
                    ),
                )
            )

    def walk(prefix: str, value: Any) -> None:
        if not isinstance(value, dict):
            return

        for key, nested_value in value.items():
            field_name = f"{prefix}.{key}" if prefix else key

            if key in SEMANTIC_NESTED_KEYS and nested_value:
                report.verification.append(
                    human_review_required(
                        name=f"semantic_config_field:{field_name}",
                        description=(
                            f"Config field '{field_name}' contains semantic "
                            "meaning and must be confirmed before final use."
                        ),
                    )
                )

            if isinstance(nested_value, dict):
                walk(field_name, nested_value)

    walk("", config)


def inspect_dataset_config(
    config: Dict[str, Any],
    config_path: Optional[str | Path] = None,
) -> InspectionReport:
    """
    Inspect one dataset config structurally.

    This function does not approve labels, mask values, task meanings, license,
    or lossy conversion decisions.
    """

    dataset_id = get_dataset_id(config)
    dataset_root_value = get_dataset_root(config)

    report = InspectionReport(
        dataset_id=dataset_id,
        dataset_root=dataset_root_value,
    )

    if config_path is not None:
        report.add_artifact(
            name="source_config",
            path=str(config_path),
            description="Dataset config used for generic inspection.",
            artifact_type="config",
        )

    report.metadata_fields_detected = sorted(str(key) for key in config.keys())

    if not dataset_root_value:
        report.add_issue(
            severity=InspectionIssueSeverity.BLOCKED,
            field="dataset_root",
            message=(
                "No dataset_root or legacy dataset_path field was found in "
                "the config."
            ),
        )
        report.verification.append(
            VerificationRecord(
                name="dataset_root_exists",
                status=VerificationStatus.BLOCKED,
                description="Dataset root could not be checked because it is missing.",
            )
        )
        add_semantic_review_flags(report, config)
        return report

    dataset_root = Path(dataset_root_value)

    if not dataset_root.exists():
        report.add_issue(
            severity=InspectionIssueSeverity.BLOCKED,
            field="dataset_root",
            message=f"Dataset root does not exist: {dataset_root}",
        )
        report.verification.append(
            VerificationRecord(
                name="dataset_root_exists",
                status=VerificationStatus.BLOCKED,
                description=f"Dataset root does not exist: {dataset_root}",
            )
        )
        add_semantic_review_flags(report, config)
        return report

    if not dataset_root.is_dir():
        report.add_issue(
            severity=InspectionIssueSeverity.BLOCKED,
            field="dataset_root",
            message=f"Dataset root exists but is not a directory: {dataset_root}",
        )
        report.verification.append(
            VerificationRecord(
                name="dataset_root_is_directory",
                status=VerificationStatus.FAILED,
                description=f"Dataset root is not a directory: {dataset_root}",
            )
        )
        add_semantic_review_flags(report, config)
        return report

    all_files = [path for path in dataset_root.rglob("*") if path.is_file()]
    extension_counts = Counter(
        path.suffix.lower() if path.suffix else "[no_extension]"
        for path in all_files
    )

    report.file_counts = {
        "total_files": len(all_files),
        **dict(sorted(extension_counts.items())),
    }

    folder_counts: Counter[str] = Counter()
    for path in all_files:
        relative = path.relative_to(dataset_root)

        if len(relative.parts) <= 1:
            top_level = "."
        else:
            top_level = relative.parts[0]

        folder_counts[top_level] += 1

    report.folder_counts = dict(sorted(folder_counts.items()))

    image_like_extensions = sorted(
        extension for extension in extension_counts if extension in IMAGE_LIKE_EXTENSIONS
    )
    raster_like_extensions = sorted(
        extension for extension in extension_counts if extension in RASTER_LIKE_EXTENSIONS
    )
    annotation_like_extensions = sorted(
        extension
        for extension in extension_counts
        if extension in ANNOTATION_LIKE_EXTENSIONS
    )

    report.image_properties = {
        "image_like_file_count": sum(
            extension_counts[extension] for extension in image_like_extensions
        ),
        "image_like_extensions": image_like_extensions,
    }
    report.raster_properties = {
        "raster_like_file_count": sum(
            extension_counts[extension] for extension in raster_like_extensions
        ),
        "raster_like_extensions": raster_like_extensions,
    }

    report.annotation_types_detected = annotation_like_extensions

    configured_paths = collect_configured_paths(config)
    configured_path_checks = []

    for item in configured_paths:
        resolved_path = resolve_config_path(dataset_root, item["path"])
        exists = resolved_path.exists()

        configured_path_checks.append(
            {
                "field": item["field"],
                "configured_path": item["path"],
                "resolved_path": str(resolved_path),
                "exists": exists,
            }
        )

        if not exists:
            report.add_issue(
                severity=InspectionIssueSeverity.WARNING,
                field=item["field"],
                message=f"Configured path does not exist: {resolved_path}",
            )

    report.pairing_summary["configured_paths"] = configured_path_checks

    report.verification.append(
        auto_verified(
            name="file_extension_counts",
            description=(
                "File extension counts were computed by recursively scanning "
                "the dataset root."
            ),
        )
    )
    report.verification.append(
        auto_verified(
            name="configured_path_existence",
            description=(
                "Configured file/folder path existence was checked relative to "
                "the dataset root."
            ),
        )
    )

    add_semantic_review_flags(report, config)

    report.notes.append(
        "Generic inspection is structural only. It does not approve label "
        "meanings, mask value meanings, task meanings, license interpretation, "
        "or lossy conversion validity."
    )

    return report


def run_dataset_inspection(
    config_path: str | Path,
    output_root: str | Path = "outputs/runs",
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run generic inspection from a config file and write standard outputs.

    Returns the output package, report object, and written artifact paths.
    """

    config = load_dataset_config(config_path)
    dataset_id = get_dataset_id(config)

    package = create_standard_output_package(
        dataset_id=dataset_id,
        output_root=output_root,
        config=config,
        run_id=run_id,
        extra_metadata={
            "tool": "generic_inspector",
            "config_path": str(config_path),
        },
    )

    report = inspect_dataset_config(
        config=config,
        config_path=config_path,
    )

    inspection_outputs = write_standard_inspection_outputs(
        report,
        package.directories["inspection"],
    )

    return {
        "package": package,
        "report": report,
        "inspection_outputs": inspection_outputs,
    }
