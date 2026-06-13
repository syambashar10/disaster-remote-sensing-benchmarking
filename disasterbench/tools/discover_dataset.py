from __future__ import annotations

import argparse
import json
from pathlib import Path

from disasterbench.inspection.dataset_discovery import discover_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically discover dataset structure and suggest a config."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to raw or extracted dataset root.")
    parser.add_argument("--dataset-id", required=True, help="Short dataset id for output naming.")
    parser.add_argument("--output-root", required=True, help="Directory where discovery outputs are written.")
    parser.add_argument("--max-samples", type=int, default=5, help="Number of JSON/NPZ/raster samples to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = discover_dataset(
        dataset_root=Path(args.dataset_root),
        dataset_id=args.dataset_id,
        output_root=Path(args.output_root),
        max_samples=args.max_samples,
    )

    print(
        json.dumps(
            {
                "dataset_id": report["dataset_id"],
                "output_dir": report["output_dir"],
                "total_files": report["file_summary"]["total_files"],
                "extension_counts": report["file_summary"]["extension_counts"],
                "task_hints": report["task_hints"],
                "pairing_suggestion_count": len(report["pairing_suggestions"]),
                "human_review_required": report["human_review_required"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
