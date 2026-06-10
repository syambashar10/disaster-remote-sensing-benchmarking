"""
CLI for generic dataset inspection.

Example:
    python -m disasterbench.tools.inspect_dataset \
      --config configs/xbd_config.json \
      --output-root outputs/runs \
      --run-id xbd_generic_inspection_test
"""

from __future__ import annotations

import argparse
import json

from disasterbench.inspection import run_dataset_inspection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generic structural inspection for a dataset config."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to dataset config JSON.",
    )

    parser.add_argument(
        "--output-root",
        default="outputs/runs",
        help="Root folder where the run package will be created.",
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID. If omitted, a timestamped run ID is generated.",
    )

    args = parser.parse_args()

    result = run_dataset_inspection(
        config_path=args.config,
        output_root=args.output_root,
        run_id=args.run_id,
    )

    package = result["package"]
    report = result["report"]
    inspection_outputs = result["inspection_outputs"]

    summary = {
        "dataset_id": report.dataset_id,
        "run_id": package.run_id,
        "root_dir": package.root_dir,
        "blocking_issues": report.has_blocking_issues(),
        "human_review_required": report.requires_human_review(),
        "inspection_report_json": str(
            inspection_outputs["inspection_report_json"]
        ),
        "inspection_summary_md": str(
            inspection_outputs["inspection_summary_md"]
        ),
        "config_snapshot_path": package.config_snapshot_path,
        "run_manifest_path": package.run_manifest_path,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
