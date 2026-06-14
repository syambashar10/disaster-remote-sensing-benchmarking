from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from disasterbench.tools.validate_common_manifest import validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all common manifests exported for datasets in a registry."
    )
    parser.add_argument("--registry", required=True, help="Path to dataset registry JSON.")
    parser.add_argument("--manifest-root", required=True, help="Directory containing exported *_common_manifest.jsonl files.")
    parser.add_argument("--output", required=True, help="Output registry manifest validation report JSON.")
    parser.add_argument("--max-errors", type=int, default=50, help="Maximum errors to store per dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    registry_path = Path(args.registry)
    manifest_root = Path(args.manifest_root)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    results: List[Dict[str, Any]] = []

    for entry in registry.get("datasets", []):
        dataset_id = entry["dataset_id"]
        manifest_path = manifest_root / f"{dataset_id}_common_manifest.jsonl"

        if not manifest_path.exists():
            results.append(
                {
                    "dataset_id": dataset_id,
                    "manifest_path": str(manifest_path),
                    "valid": False,
                    "error": "manifest_file_not_found",
                }
            )
            continue

        report = validate_manifest(manifest_path, max_errors=args.max_errors)
        report["dataset_id"] = dataset_id
        results.append(report)

    final_report = {
        "registry": str(registry_path),
        "manifest_root": str(manifest_root),
        "dataset_count": len(results),
        "valid_dataset_count": sum(1 for item in results if item.get("valid")),
        "invalid_dataset_count": sum(1 for item in results if not item.get("valid")),
        "results": results,
        "valid": all(item.get("valid") for item in results),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")

    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
