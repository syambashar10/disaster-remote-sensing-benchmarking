from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from disasterbench.schemas.common_sample import validate_common_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a DisasterBench common JSONL manifest."
    )
    parser.add_argument("--manifest", required=True, help="Input common manifest JSONL path.")
    parser.add_argument("--output", required=True, help="Output validation summary JSON path.")
    parser.add_argument("--max-errors", type=int, default=50, help="Maximum errors to store in report.")
    return parser.parse_args()


def validate_manifest(manifest_path: str | Path, max_errors: int = 50) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)

    total_lines = 0
    valid_samples = 0
    errors = []

    dataset_ids = Counter()
    task_types = Counter()
    annotation_types = Counter()
    geometry_types = Counter()

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()

            if not line:
                continue

            total_lines += 1

            try:
                sample = json.loads(line)
                validate_common_sample(sample)

                valid_samples += 1

                dataset_ids[str(sample.get("dataset_id"))] += 1
                task_types[str(sample.get("task_type"))] += 1

                for annotation in sample.get("annotations", []):
                    annotation_types[str(annotation.get("type"))] += 1
                    geometry_types[str(annotation.get("geometry_type"))] += 1

            except Exception as exc:
                if len(errors) < max_errors:
                    errors.append(
                        {
                            "line_number": line_number,
                            "error": str(exc),
                        }
                    )

    return {
        "manifest_path": str(manifest_path),
        "total_lines": total_lines,
        "valid_samples": valid_samples,
        "invalid_samples": total_lines - valid_samples,
        "dataset_ids": dict(dataset_ids),
        "task_types": dict(task_types),
        "annotation_types": dict(annotation_types),
        "geometry_types": dict(geometry_types),
        "errors": errors,
        "valid": total_lines == valid_samples,
    }


def main() -> None:
    args = parse_args()

    report = validate_manifest(args.manifest, max_errors=args.max_errors)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
