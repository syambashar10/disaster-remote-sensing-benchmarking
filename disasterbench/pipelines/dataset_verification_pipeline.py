"""
Dataset verification pipeline.

This module connects reusable inspectors into one report. It does not replace
format-specific inspectors. It orchestrates them based on a dataset config.

Current supported checks:
- Generic dataset inspection
- Configured file pairing checks
- Configured raster consistency checks

Future checks:
- JSON annotation inspection from config
- Raster/mask inspection from config
- Suggested config generation
- Capability report
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from disasterbench.configs import load_dataset_config
from disasterbench.inspection.generic_inspector import inspect_dataset_config
from disasterbench.inspection.file_pairing_inspector import inspect_file_pairing
from disasterbench.inspection.paired_raster_consistency import (
    inspect_paired_raster_consistency,
)
from disasterbench.packaging import create_standard_output_package


@dataclass
class PipelineCheckRecord:
    name: str
    status: str
    output_path: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DatasetVerificationPipelineResult:
    dataset_id: str
    run_id: str
    output_root: str
    checks: List[PipelineCheckRecord]
    human_review_required: bool
    blocking_issues: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_dataset_path(dataset_root: str | Path, configured_path: str | Path) -> Path:
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return Path(dataset_root) / path


def write_json(data: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def run_file_pairing_checks(
    config: Dict[str, Any],
    dataset_root: str | Path,
    output_dir: str | Path,
) -> List[PipelineCheckRecord]:
    records: List[PipelineCheckRecord] = []

    pairing_checks = config.get("pairing_checks", [])

    if not pairing_checks:
        records.append(
            PipelineCheckRecord(
                name="file_pairing_checks",
                status="skipped",
                summary={"reason": "No pairing_checks configured."},
            )
        )
        return records

    for index, check in enumerate(pairing_checks):
        name = check.get("name", f"pairing_check_{index}")
        output_path = Path(output_dir) / f"{name}.json"

        try:
            reference_root = resolve_dataset_path(dataset_root, check["reference_root"])
            candidate_roots = {
                candidate_name: resolve_dataset_path(dataset_root, candidate_path)
                for candidate_name, candidate_path in check["candidate_roots"].items()
            }

            result = inspect_file_pairing(
                reference_root=reference_root,
                reference_extensions=check.get("reference_extensions"),
                candidate_roots=candidate_roots,
                candidate_extensions=check.get("candidate_extensions"),
                strip_suffixes=check.get("strip_suffixes", []),
                max_examples=check.get("max_examples", 20),
            )

            write_json(result.to_dict(), output_path)

            missing_counts = {
                key: len(value)
                for key, value in result.missing_from_candidates.items()
            }
            extra_counts = {
                key: len(value)
                for key, value in result.extra_in_candidates.items()
            }

            has_issue = (
                result.has_missing_pairs()
                or result.has_extra_pairs()
                or result.has_duplicates()
                or bool(result.missing_roots)
            )

            records.append(
                PipelineCheckRecord(
                    name=name,
                    status="failed" if has_issue else "passed",
                    output_path=str(output_path),
                    summary={
                        "reference_file_count": result.reference_file_count,
                        "candidate_file_counts": result.candidate_file_counts,
                        "matched_counts": result.matched_counts,
                        "missing_counts": missing_counts,
                        "extra_counts": extra_counts,
                        "duplicate_reference_stems": len(result.duplicate_reference_stems),
                        "duplicate_candidate_stems": {
                            key: len(value)
                            for key, value in result.duplicate_candidate_stems.items()
                        },
                        "missing_roots": result.missing_roots,
                    },
                )
            )

        except Exception as error:
            records.append(
                PipelineCheckRecord(
                    name=name,
                    status="error",
                    output_path=str(output_path),
                    error=str(error),
                )
            )

    return records


def run_paired_raster_checks(
    config: Dict[str, Any],
    dataset_root: str | Path,
    output_dir: str | Path,
    max_pairs: Optional[int] = None,
) -> List[PipelineCheckRecord]:
    records: List[PipelineCheckRecord] = []

    raster_checks = config.get("paired_raster_checks", [])

    if not raster_checks:
        records.append(
            PipelineCheckRecord(
                name="paired_raster_checks",
                status="skipped",
                summary={"reason": "No paired_raster_checks configured."},
            )
        )
        return records

    for index, check in enumerate(raster_checks):
        name = check.get("name", f"paired_raster_check_{index}")
        output_path = Path(output_dir) / f"{name}.json"

        try:
            reference_root = resolve_dataset_path(dataset_root, check["reference_root"])
            candidate_root = resolve_dataset_path(dataset_root, check["candidate_root"])

            result = inspect_paired_raster_consistency(
                reference_root=reference_root,
                candidate_root=candidate_root,
                reference_extensions=check.get("reference_extensions", [".tif", ".tiff"]),
                candidate_extensions=check.get("candidate_extensions", [".tif", ".tiff"]),
                strip_suffixes=check.get("strip_suffixes", []),
                max_pairs=max_pairs if max_pairs is not None else check.get("max_pairs"),
                transform_tolerance=check.get("transform_tolerance", 1e-9),
            )

            write_json(result.to_dict(), output_path)

            has_issue = result.has_mismatches()

            records.append(
                PipelineCheckRecord(
                    name=name,
                    status="failed" if has_issue else "passed",
                    output_path=str(output_path),
                    summary={
                        "total_reference_files": result.total_reference_files,
                        "total_candidate_files": result.total_candidate_files,
                        "paired_stem_count": result.paired_stem_count,
                        "checked_pair_count": result.checked_pair_count,
                        "shape_mismatch_count": result.shape_mismatch_count,
                        "crs_mismatch_count": result.crs_mismatch_count,
                        "transform_mismatch_count": result.transform_mismatch_count,
                        "invalid_pair_count": result.invalid_pair_count,
                        "missing_candidate_stems": len(result.missing_candidate_stems),
                        "extra_candidate_stems": len(result.extra_candidate_stems),
                        "read_errors": len(result.read_errors),
                    },
                )
            )

        except Exception as error:
            records.append(
                PipelineCheckRecord(
                    name=name,
                    status="error",
                    output_path=str(output_path),
                    error=str(error),
                )
            )

    return records


def run_dataset_verification_pipeline(
    config_path: str | Path,
    output_root: str | Path = "outputs/runs",
    run_id: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> DatasetVerificationPipelineResult:
    config = load_dataset_config(config_path)
    dataset_id = str(config.get("dataset_id") or "unknown_dataset")
    dataset_root = str(config.get("dataset_root") or config.get("dataset_path") or "")

    package = create_standard_output_package(
        dataset_id=dataset_id,
        output_root=output_root,
        config=config,
        run_id=run_id,
        extra_metadata={
            "tool": "dataset_verification_pipeline",
            "config_path": str(config_path),
            "max_pairs": max_pairs,
        },
    )

    checks: List[PipelineCheckRecord] = []

    generic_report = inspect_dataset_config(config=config, config_path=config_path)
    generic_output = Path(package.directories["inspection"]) / "generic_inspection_report.json"
    write_json(generic_report.to_dict(), generic_output)

    checks.append(
        PipelineCheckRecord(
            name="generic_inspection",
            status="failed" if generic_report.has_blocking_issues() else "passed",
            output_path=str(generic_output),
            summary={
                "dataset_root": generic_report.dataset_root,
                "file_counts": generic_report.file_counts,
                "folder_counts": generic_report.folder_counts,
                "issue_count": len(generic_report.issues),
                "human_review_required": generic_report.requires_human_review(),
            },
        )
    )

    verification_dir = Path(package.root_dir) / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)

    checks.extend(
        run_file_pairing_checks(
            config=config,
            dataset_root=dataset_root,
            output_dir=verification_dir / "file_pairing",
        )
    )
    checks.extend(
        run_paired_raster_checks(
            config=config,
            dataset_root=dataset_root,
            output_dir=verification_dir / "paired_rasters",
            max_pairs=max_pairs,
        )
    )

    blocking_issues = any(check.status in {"failed", "error"} for check in checks)
    human_review_required = generic_report.requires_human_review()

    result = DatasetVerificationPipelineResult(
        dataset_id=dataset_id,
        run_id=package.run_id,
        output_root=str(package.root_dir),
        checks=checks,
        human_review_required=human_review_required,
        blocking_issues=blocking_issues,
    )

    write_json(
        result.to_dict(),
        Path(package.root_dir) / "verification_report.json",
    )

    return result
