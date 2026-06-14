from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from disasterbench.loaders.config_loader_factory import create_loader_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export common JSONL manifests for every dataset in a registry."
    )
    parser.add_argument("--registry", required=True, help="Path to dataset registry JSON.")
    parser.add_argument("--output-root", required=True, help="Output directory for JSONL manifests and summaries.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum samples per dataset.")
    return parser.parse_args()


def export_one_dataset(
    dataset_id: str,
    config_path: str,
    output_root: Path,
    max_samples: int | None,
) -> Dict[str, Any]:
    loader_kwargs = {}

    if max_samples is not None:
        loader_kwargs["max_samples"] = max_samples

    loader = create_loader_from_config(config_path, **loader_kwargs)

    output_jsonl = output_root / f"{dataset_id}_common_manifest.jsonl"
    summary_json = output_root / f"{dataset_id}_common_manifest_summary.json"

    output_root.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    errors = []

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for index in range(len(loader)):
            try:
                sample = loader.load_sample(index)
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
                exported_count += 1
            except Exception as exc:
                errors.append(
                    {
                        "index": index,
                        "error": str(exc),
                    }
                )

    summary = {
        "dataset_id": dataset_id,
        "config_path": config_path,
        "loader_class": loader.__class__.__name__,
        "requested_sample_count": len(loader),
        "exported_count": exported_count,
        "error_count": len(errors),
        "errors": errors[:50],
        "output_jsonl": str(output_jsonl),
        "summary_json": str(summary_json),
        "valid": len(errors) == 0,
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    args = parse_args()

    registry_path = Path(args.registry)
    output_root = Path(args.output_root)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    results: List[Dict[str, Any]] = []

    for entry in registry.get("datasets", []):
        dataset_id = entry["dataset_id"]
        config_path = entry["config_path"]

        results.append(
            export_one_dataset(
                dataset_id=dataset_id,
                config_path=config_path,
                output_root=output_root,
                max_samples=args.max_samples,
            )
        )

    report = {
        "registry": str(registry_path),
        "output_root": str(output_root),
        "dataset_count": len(results),
        "valid_dataset_count": sum(1 for item in results if item["valid"]),
        "invalid_dataset_count": sum(1 for item in results if not item["valid"]),
        "results": results,
    }

    report_path = output_root / "registry_common_manifest_export_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
