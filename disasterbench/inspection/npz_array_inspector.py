from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _counter_to_json(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def _safe_numeric_min_max(array: np.ndarray) -> dict[str, Any]:
    if array.size == 0:
        return {"min": None, "max": None}

    if not np.issubdtype(array.dtype, np.number):
        return {"min": None, "max": None}

    return {
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
    }


def inspect_npz_file(path: Path, label_keys: set[str] | None = None) -> dict[str, Any]:
    label_keys = label_keys or {"label", "mask", "target", "labels", "y"}

    result: dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "keys": [],
        "arrays": {},
        "label_value_counts": {},
        "error": None,
    }

    try:
        with np.load(path, allow_pickle=False) as data:
            result["valid"] = True
            result["keys"] = list(data.files)

            for key in data.files:
                array = data[key]
                stats = _safe_numeric_min_max(array)

                result["arrays"][key] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "min": stats["min"],
                    "max": stats["max"],
                }

                if key.lower() in label_keys and np.issubdtype(array.dtype, np.integer):
                    values, counts = np.unique(array, return_counts=True)
                    result["label_value_counts"][key] = {
                        str(int(value)): int(count)
                        for value, count in zip(values, counts)
                    }

    except Exception as exc:
        result["error"] = str(exc)

    return result


def inspect_npz_dataset(
    dataset_root: Path,
    output_path: Path | None = None,
    max_files: int | None = None,
    label_keys: set[str] | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    files = sorted(dataset_root.rglob("*.npz"))

    if max_files is not None:
        files_to_read = files[:max_files]
    else:
        files_to_read = files

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "total_npz_files": len(files),
        "inspected_npz_files": len(files_to_read),
        "folder_counts": Counter(),
        "key_combinations": Counter(),
        "array_shapes_by_key": defaultdict(Counter),
        "array_dtypes_by_key": defaultdict(Counter),
        "array_min_max_by_key": {},
        "label_value_counts_by_key": defaultdict(Counter),
        "files_with_positive_label_by_key": Counter(),
        "files_with_only_zero_label_by_key": Counter(),
        "read_error_count": 0,
        "read_errors": [],
        "sample_files": [],
    }

    for path in files_to_read:
        relative = path.relative_to(dataset_root)
        top_folder = relative.parts[0] if len(relative.parts) > 1 else "."
        summary["folder_counts"][top_folder] += 1

        file_report = inspect_npz_file(path, label_keys=label_keys)

        if len(summary["sample_files"]) < 5:
            summary["sample_files"].append(file_report)

        if not file_report["valid"]:
            summary["read_error_count"] += 1
            summary["read_errors"].append(
                {
                    "path": str(path),
                    "error": file_report["error"],
                }
            )
            continue

        keys = tuple(sorted(file_report["keys"]))
        summary["key_combinations"][keys] += 1

        for key, array_info in file_report["arrays"].items():
            shape = tuple(array_info["shape"])
            dtype = array_info["dtype"]

            summary["array_shapes_by_key"][key][shape] += 1
            summary["array_dtypes_by_key"][key][dtype] += 1

            current_min_max = summary["array_min_max_by_key"].setdefault(
                key,
                {"min": None, "max": None},
            )

            arr_min = array_info["min"]
            arr_max = array_info["max"]

            if arr_min is not None:
                if current_min_max["min"] is None or arr_min < current_min_max["min"]:
                    current_min_max["min"] = arr_min

            if arr_max is not None:
                if current_min_max["max"] is None or arr_max > current_min_max["max"]:
                    current_min_max["max"] = arr_max

        for label_key, value_counts in file_report["label_value_counts"].items():
            values_present = set(value_counts.keys())

            for value, count in value_counts.items():
                summary["label_value_counts_by_key"][label_key][value] += count

            positive_values = values_present - {"0"}
            if positive_values:
                summary["files_with_positive_label_by_key"][label_key] += 1

            if values_present == {"0"}:
                summary["files_with_only_zero_label_by_key"][label_key] += 1

    json_ready = {
        "dataset_root": summary["dataset_root"],
        "total_npz_files": summary["total_npz_files"],
        "inspected_npz_files": summary["inspected_npz_files"],
        "folder_counts": _counter_to_json(summary["folder_counts"]),
        "key_combinations": _counter_to_json(summary["key_combinations"]),
        "array_shapes_by_key": {
            key: _counter_to_json(counter)
            for key, counter in summary["array_shapes_by_key"].items()
        },
        "array_dtypes_by_key": {
            key: _counter_to_json(counter)
            for key, counter in summary["array_dtypes_by_key"].items()
        },
        "array_min_max_by_key": summary["array_min_max_by_key"],
        "label_value_counts_by_key": {
            key: _counter_to_json(counter)
            for key, counter in summary["label_value_counts_by_key"].items()
        },
        "files_with_positive_label_by_key": _counter_to_json(
            summary["files_with_positive_label_by_key"]
        ),
        "files_with_only_zero_label_by_key": _counter_to_json(
            summary["files_with_only_zero_label_by_key"]
        ),
        "read_error_count": summary["read_error_count"],
        "read_errors": summary["read_errors"],
        "sample_files": summary["sample_files"],
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(json_ready, indent=2), encoding="utf-8")

    return json_ready
