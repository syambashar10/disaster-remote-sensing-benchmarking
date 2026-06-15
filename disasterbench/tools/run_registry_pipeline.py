from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from disasterbench.capabilities.capability_matrix import (
    build_capability_matrix,
    write_capability_matrix_json,
    write_capability_matrix_markdown,
)
from disasterbench.tools.export_registry_manifests import export_one_dataset
from disasterbench.tools.validate_common_manifest import validate_manifest
from disasterbench.tools.validate_dataset_registry import validate_one_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full DisasterBench registry pipeline."
    )
    parser.add_argument("--registry", required=True, help="Path to dataset registry JSON.")
    parser.add_argument("--output-root", required=True, help="Root output directory for pipeline artifacts.")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum samples per dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry_path = Path(args.registry)
    output_root = Path(args.output_root)

    validation_dir = output_root / "validation"
    manifests_dir = output_root / "common_manifests"
    reports_dir = output_root / "reports"

    validation_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    dataset_entries = registry.get("datasets", [])

    dataset_validation_results: List[Dict[str, Any]] = []
    manifest_export_results: List[Dict[str, Any]] = []
    manifest_validation_results: List[Dict[str, Any]] = []

    for entry in dataset_entries:
        dataset_validation_results.append(
            validate_one_dataset(entry, max_samples=args.max_samples)
        )

    dataset_validation_report = {
        "registry": str(registry_path),
        "dataset_count": len(dataset_validation_results),
        "valid_dataset_count": sum(1 for item in dataset_validation_results if item.get("valid")),
        "invalid_dataset_count": sum(1 for item in dataset_validation_results if not item.get("valid")),
        "results": dataset_validation_results,
        "valid": all(item.get("valid") for item in dataset_validation_results),
    }

    dataset_validation_path = validation_dir / "dataset_config_loader_validation.json"
    dataset_validation_path.write_text(
        json.dumps(dataset_validation_report, indent=2),
        encoding="utf-8",
    )

    for entry in dataset_entries:
        manifest_export_results.append(
            export_one_dataset(
                dataset_id=entry["dataset_id"],
                config_path=entry["config_path"],
                output_root=manifests_dir,
                max_samples=args.max_samples,
            )
        )

    manifest_export_report = {
        "registry": str(registry_path),
        "manifest_root": str(manifests_dir),
        "dataset_count": len(manifest_export_results),
        "valid_dataset_count": sum(1 for item in manifest_export_results if item.get("valid")),
        "invalid_dataset_count": sum(1 for item in manifest_export_results if not item.get("valid")),
        "results": manifest_export_results,
        "valid": all(item.get("valid") for item in manifest_export_results),
    }

    manifest_export_path = manifests_dir / "registry_common_manifest_export_summary.json"
    manifest_export_path.write_text(
        json.dumps(manifest_export_report, indent=2),
        encoding="utf-8",
    )

    for entry in dataset_entries:
        dataset_id = entry["dataset_id"]
        manifest_path = manifests_dir / f"{dataset_id}_common_manifest.jsonl"

        if not manifest_path.exists():
            manifest_validation_results.append(
                {
                    "dataset_id": dataset_id,
                    "manifest_path": str(manifest_path),
                    "valid": False,
                    "error": "manifest_file_not_found",
                }
            )
            continue

        report = validate_manifest(manifest_path)
        report["dataset_id"] = dataset_id
        manifest_validation_results.append(report)

    manifest_validation_report = {
        "registry": str(registry_path),
        "manifest_root": str(manifests_dir),
        "dataset_count": len(manifest_validation_results),
        "valid_dataset_count": sum(1 for item in manifest_validation_results if item.get("valid")),
        "invalid_dataset_count": sum(1 for item in manifest_validation_results if not item.get("valid")),
        "results": manifest_validation_results,
        "valid": all(item.get("valid") for item in manifest_validation_results),
    }

    manifest_validation_path = validation_dir / "registry_manifest_validation_summary.json"
    manifest_validation_path.write_text(
        json.dumps(manifest_validation_report, indent=2),
        encoding="utf-8",
    )

    capability_report = build_capability_matrix(registry_path)

    capability_json_path = reports_dir / "dataset_capability_matrix.json"
    capability_markdown_path = reports_dir / "dataset_capability_matrix.md"

    write_capability_matrix_json(capability_report, capability_json_path)
    write_capability_matrix_markdown(capability_report, capability_markdown_path)

    final_report = {
        "registry": str(registry_path),
        "output_root": str(output_root),
        "max_samples": args.max_samples,
        "dataset_count": len(dataset_entries),
        "dataset_config_validation": {
            "path": str(dataset_validation_path),
            "valid": dataset_validation_report["valid"],
            "valid_dataset_count": dataset_validation_report["valid_dataset_count"],
            "invalid_dataset_count": dataset_validation_report["invalid_dataset_count"],
        },
        "common_manifest_export": {
            "path": str(manifest_export_path),
            "valid": manifest_export_report["valid"],
            "valid_dataset_count": manifest_export_report["valid_dataset_count"],
            "invalid_dataset_count": manifest_export_report["invalid_dataset_count"],
        },
        "common_manifest_validation": {
            "path": str(manifest_validation_path),
            "valid": manifest_validation_report["valid"],
            "valid_dataset_count": manifest_validation_report["valid_dataset_count"],
            "invalid_dataset_count": manifest_validation_report["invalid_dataset_count"],
        },
        "capability_matrix": {
            "json_path": str(capability_json_path),
            "markdown_path": str(capability_markdown_path),
            "dataset_count": capability_report["dataset_count"],
            "output_format_count": capability_report["output_format_count"],
        },
        "valid": (
            dataset_validation_report["valid"]
            and manifest_export_report["valid"]
            and manifest_validation_report["valid"]
        ),
    }

    final_report_path = output_root / "registry_pipeline_report.json"
    final_report_path.write_text(
        json.dumps(final_report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
