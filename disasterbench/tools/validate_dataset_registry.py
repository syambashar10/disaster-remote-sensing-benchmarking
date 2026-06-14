from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from disasterbench.loaders.config_loader_factory import create_loader_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all dataset configs listed in a dataset registry."
    )
    parser.add_argument("--registry", required=True, help="Path to dataset registry JSON.")
    parser.add_argument("--max-samples", type=int, default=5, help="Max samples per dataset for fast validation.")
    parser.add_argument("--output", default=None, help="Optional output JSON report path.")
    return parser.parse_args()


def validate_one_dataset(entry: Dict[str, Any], max_samples: int) -> Dict[str, Any]:
    dataset_id = entry["dataset_id"]
    config_path = entry["config_path"]

    try:
        loader = create_loader_from_config(config_path, max_samples=max_samples)
        stats = loader.get_statistics()
        pairing = loader.validate_pairing()

        first_sample_error = None
        first_sample_id = None

        try:
            if len(loader) > 0:
                first_sample = loader.load_sample(0)
                first_sample_id = first_sample.get("sample_id")
        except Exception as exc:
            first_sample_error = str(exc)

        return {
            "dataset_id": dataset_id,
            "config_path": config_path,
            "loader_class": loader.__class__.__name__,
            "sample_count_checked": len(loader),
            "total_samples_reported": stats.get("total_samples"),
            "pairing_valid": bool(pairing.get("valid", False)),
            "first_sample_id": first_sample_id,
            "first_sample_error": first_sample_error,
            "valid": bool(pairing.get("valid", False)) and first_sample_error is None,
        }

    except Exception as exc:
        return {
            "dataset_id": dataset_id,
            "config_path": config_path,
            "valid": False,
            "error": str(exc),
        }


def main() -> None:
    args = parse_args()

    registry_path = Path(args.registry)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    results: List[Dict[str, Any]] = []

    for entry in registry.get("datasets", []):
        results.append(validate_one_dataset(entry, max_samples=args.max_samples))

    report = {
        "registry": str(registry_path),
        "registry_name": registry.get("registry_name"),
        "version": registry.get("version"),
        "dataset_count": len(results),
        "valid_dataset_count": sum(1 for item in results if item.get("valid")),
        "invalid_dataset_count": sum(1 for item in results if not item.get("valid")),
        "results": results,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
