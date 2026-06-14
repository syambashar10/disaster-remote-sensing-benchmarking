from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from disasterbench.loaders.config_loader_factory import create_loader_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export loaded dataset samples into DisasterBench common JSONL manifest."
    )
    parser.add_argument("--config", required=True, help="Path to dataset config JSON.")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--summary-output", default=None, help="Optional output summary JSON path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum samples to export.")
    return parser.parse_args()


def _safe_json_dump_line(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def main() -> None:
    args = parse_args()

    loader_kwargs = {}
    if args.max_samples is not None:
        loader_kwargs["max_samples"] = args.max_samples

    loader = create_loader_from_config(args.config, **loader_kwargs)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    errors = []

    with output_path.open("w", encoding="utf-8") as handle:
        for index in range(len(loader)):
            try:
                sample = loader.load_sample(index)
                handle.write(_safe_json_dump_line(sample) + "\n")
                exported_count += 1
            except Exception as exc:
                errors.append(
                    {
                        "index": index,
                        "error": str(exc),
                    }
                )

    summary = {
        "config_path": args.config,
        "loader_class": loader.__class__.__name__,
        "requested_sample_count": len(loader),
        "exported_count": exported_count,
        "error_count": len(errors),
        "errors": errors[:50],
        "output_jsonl": str(output_path),
        "valid": len(errors) == 0,
    }

    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
