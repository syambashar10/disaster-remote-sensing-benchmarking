from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from disasterbench.loaders.config_loader_factory import create_loader_from_config


def _compact_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_id": sample.get("dataset_id"),
        "sample_id": sample.get("sample_id"),
        "split": sample.get("split"),
        "task_type": sample.get("task_type"),
        "image": sample.get("image"),
        "annotation_count": len(sample.get("annotations", [])),
        "first_annotation": sample.get("annotations", [None])[0],
        "metadata_keys": list(sample.get("metadata", {}).keys()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a dataset config by creating its loader, checking pairing, and loading one sample."
    )
    parser.add_argument("--config", required=True, help="Path to dataset config JSON.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples for fast validation.")
    parser.add_argument("--output", default=None, help="Optional output JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader_kwargs = {}
    if args.max_samples is not None:
        loader_kwargs["max_samples"] = args.max_samples

    loader = create_loader_from_config(args.config, **loader_kwargs)

    statistics = loader.get_statistics()
    pairing = loader.validate_pairing()

    first_sample = None
    first_sample_error = None

    try:
        if len(loader) > 0:
            first_sample = _compact_sample(loader.load_sample(0))
    except Exception as exc:
        first_sample_error = str(exc)

    report = {
        "config_path": args.config,
        "loader_class": loader.__class__.__name__,
        "sample_count": len(loader),
        "statistics": statistics,
        "pairing": pairing,
        "first_sample": first_sample,
        "first_sample_error": first_sample_error,
        "valid": bool(pairing.get("valid", False)) and first_sample_error is None,
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
