"""
CLI for reusable JSON annotation inspection.

Example:
    python -m disasterbench.tools.inspect_json_annotations \
      --json-root datasets/xbd_raw/train/labels \
      --output outputs/runs/xbd_json_inspection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from disasterbench.inspection.json_annotation_inspector import inspect_json_annotations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect JSON annotation files without dataset-specific assumptions."
    )

    parser.add_argument(
        "--json-root",
        required=True,
        help="Folder containing JSON files.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of JSON files to scan.",
    )

    parser.add_argument(
        "--max-list-items-per-list",
        type=int,
        default=None,
        help=(
            "Optional limit for nested JSON list traversal. "
            "If omitted, all list items are inspected exactly."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path.",
    )

    args = parser.parse_args()

    result = inspect_json_annotations(
        json_root=args.json_root,
        max_files=args.max_files,
        max_list_items_per_list=args.max_list_items_per_list,
    )

    result_dict = result.to_dict()
    output_text = json.dumps(result_dict, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")

    print(output_text)


if __name__ == "__main__":
    main()
