from __future__ import annotations

import argparse
import json
from pathlib import Path

from disasterbench.inspection.npz_array_inspector import inspect_npz_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect NPZ array datasets and summarize internal array schemas."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing .npz files.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of NPZ files to inspect.")
    parser.add_argument(
        "--label-keys",
        nargs="*",
        default=["label", "mask", "target", "labels", "y"],
        help="Array keys treated as label/mask arrays for unique-value counting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = inspect_npz_dataset(
        dataset_root=Path(args.dataset_root),
        output_path=Path(args.output),
        max_files=args.max_files,
        label_keys=set(args.label_keys),
    )

    print(
        json.dumps(
            {
                "dataset_root": report["dataset_root"],
                "total_npz_files": report["total_npz_files"],
                "inspected_npz_files": report["inspected_npz_files"],
                "folder_counts": report["folder_counts"],
                "key_combinations": report["key_combinations"],
                "array_shapes_by_key": report["array_shapes_by_key"],
                "array_dtypes_by_key": report["array_dtypes_by_key"],
                "label_value_counts_by_key": report["label_value_counts_by_key"],
                "files_with_positive_label_by_key": report["files_with_positive_label_by_key"],
                "files_with_only_zero_label_by_key": report["files_with_only_zero_label_by_key"],
                "read_error_count": report["read_error_count"],
                "output": args.output,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
