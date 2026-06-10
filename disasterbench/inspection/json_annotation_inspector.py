"""
Reusable JSON annotation inspector.

This module inspects JSON annotation files without assuming a specific dataset.
It detects structure, common annotation signals, invalid JSON files, top-level
keys, nested key paths, empty lists, and possible label/geometry fields.

It does not approve semantic meanings.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


GEOMETRY_LIKE_KEYS = {
    "bbox",
    "boxes",
    "polygon",
    "polygons",
    "segmentation",
    "coordinates",
    "wkt",
    "geometry",
}

CLASS_LIKE_KEYS = {
    "class",
    "category",
    "category_id",
    "label",
    "labels",
    "subtype",
    "damage",
    "damage_class",
    "feature_type",
}


@dataclass
class JsonReadError:
    path: str
    error: str


@dataclass
class JsonAnnotationInspectionResult:
    json_root: str
    total_json_files: int
    scanned_json_files: int
    valid_json_files: int
    invalid_json_files: int
    empty_json_content_files: int
    scan_limited: bool
    max_files: Optional[int]
    max_list_items_per_list: Optional[int]
    top_level_key_counts: Dict[str, int] = field(default_factory=dict)
    list_truncation_counts: Dict[str, int] = field(default_factory=dict)
    nested_key_path_counts: Dict[str, int] = field(default_factory=dict)
    empty_list_path_counts: Dict[str, int] = field(default_factory=dict)
    annotation_signal_counts: Dict[str, int] = field(default_factory=dict)
    geometry_like_key_counts: Dict[str, int] = field(default_factory=dict)
    class_like_key_counts: Dict[str, int] = field(default_factory=dict)
    example_files: Dict[str, List[str]] = field(default_factory=dict)
    read_errors: List[JsonReadError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["read_errors"] = [asdict(error) for error in self.read_errors]
        return data


def is_empty_json_content(value: Any) -> bool:
    """Return True for structurally empty JSON content."""

    return value in ({}, [], None)


def add_example(
    examples: Dict[str, List[str]],
    signal: str,
    path: Path,
    limit: int = 5,
) -> None:
    """Store a few example files for each detected signal."""

    examples.setdefault(signal, [])

    if len(examples[signal]) < limit:
        examples[signal].append(str(path))


def walk_json_structure(
    value: Any,
    path_prefix: str,
    nested_key_path_counts: Counter[str],
    empty_list_path_counts: Counter[str],
    geometry_like_key_counts: Counter[str],
    class_like_key_counts: Counter[str],
    list_truncation_counts: Counter[str],
    max_list_items_per_list: Optional[int],
) -> None:
    """Recursively count key paths and common annotation-related keys."""

    if isinstance(value, dict):
        for key, nested_value in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            full_path = f"{path_prefix}.{key_text}" if path_prefix else key_text

            nested_key_path_counts[full_path] += 1

            if key_lower in GEOMETRY_LIKE_KEYS:
                geometry_like_key_counts[key_text] += 1

            if key_lower in CLASS_LIKE_KEYS:
                class_like_key_counts[key_text] += 1

            walk_json_structure(
                nested_value,
                full_path,
                nested_key_path_counts,
                empty_list_path_counts,
                geometry_like_key_counts,
                class_like_key_counts,
                list_truncation_counts,
                max_list_items_per_list,
            )

    elif isinstance(value, list):
        list_path = path_prefix or "[root]"

        if len(value) == 0:
            empty_list_path_counts[list_path] += 1

        if max_list_items_per_list is not None and len(value) > max_list_items_per_list:
            list_truncation_counts[list_path] += 1
            items_to_walk = value[:max_list_items_per_list]
        else:
            items_to_walk = value

        for item in items_to_walk:
            walk_json_structure(
                item,
                path_prefix,
                nested_key_path_counts,
                empty_list_path_counts,
                geometry_like_key_counts,
                class_like_key_counts,
                list_truncation_counts,
                max_list_items_per_list,
            )


def detect_annotation_signals(data: Any) -> List[str]:
    """
    Detect common JSON annotation structures.

    These are structural hints only. They are not final semantic approval.
    """

    signals = []

    if isinstance(data, dict):
        if isinstance(data.get("annotations"), list):
            signals.append("coco_like_annotations_list")

        if isinstance(data.get("images"), list) and isinstance(data.get("categories"), list):
            signals.append("coco_like_dataset_file")

        features = data.get("features")

        if isinstance(features, list):
            signals.append("geojson_like_features_list")

        if isinstance(features, dict):
            if isinstance(features.get("xy"), list):
                signals.append("xbd_like_features_xy_list")

            if isinstance(features.get("lng_lat"), list):
                signals.append("xbd_like_features_lng_lat_list")

        if data.get("type") == "FeatureCollection":
            signals.append("geojson_feature_collection")

    return signals


def inspect_json_annotations(
    json_root: str | Path,
    max_files: Optional[int] = None,
    max_list_items_per_list: Optional[int] = None,
) -> JsonAnnotationInspectionResult:
    """
    Inspect JSON files below a root folder.

    Args:
        json_root: Folder containing JSON annotation files.
        max_files: Optional limit for faster discovery scans.
        max_list_items_per_list: Optional limit for large nested JSON lists.
            If None, all list items are inspected exactly.

    Returns:
        JsonAnnotationInspectionResult
    """

    root = Path(json_root)

    if not root.exists():
        return JsonAnnotationInspectionResult(
            json_root=str(root),
            total_json_files=0,
            scanned_json_files=0,
            valid_json_files=0,
            invalid_json_files=0,
            empty_json_content_files=0,
            scan_limited=False,
            max_files=max_files,
            max_list_items_per_list=max_list_items_per_list,
            read_errors=[
                JsonReadError(
                    path=str(root),
                    error="JSON root does not exist.",
                )
            ],
        )

    json_files = sorted(path for path in root.rglob("*.json") if path.is_file())
    total_json_files = len(json_files)

    if max_files is not None:
        json_files_to_scan = json_files[:max_files]
    else:
        json_files_to_scan = json_files

    top_level_key_counts: Counter[str] = Counter()
    nested_key_path_counts: Counter[str] = Counter()
    empty_list_path_counts: Counter[str] = Counter()
    annotation_signal_counts: Counter[str] = Counter()
    geometry_like_key_counts: Counter[str] = Counter()
    class_like_key_counts: Counter[str] = Counter()
    list_truncation_counts: Counter[str] = Counter()
    examples: Dict[str, List[str]] = {}

    valid_json_files = 0
    invalid_json_files = 0
    empty_json_content_files = 0
    read_errors: List[JsonReadError] = []

    for path in json_files_to_scan:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

        except Exception as error:
            invalid_json_files += 1
            read_errors.append(
                JsonReadError(
                    path=str(path),
                    error=str(error),
                )
            )
            add_example(examples, "invalid_json", path)
            continue

        valid_json_files += 1

        if is_empty_json_content(data):
            empty_json_content_files += 1
            add_example(examples, "empty_json_content", path)

        if isinstance(data, dict):
            for key in data.keys():
                top_level_key_counts[str(key)] += 1

        walk_json_structure(
            value=data,
            path_prefix="",
            nested_key_path_counts=nested_key_path_counts,
            empty_list_path_counts=empty_list_path_counts,
            geometry_like_key_counts=geometry_like_key_counts,
            class_like_key_counts=class_like_key_counts,
            list_truncation_counts=list_truncation_counts,
            max_list_items_per_list=max_list_items_per_list,
        )

        for signal in detect_annotation_signals(data):
            annotation_signal_counts[signal] += 1
            add_example(examples, signal, path)

    return JsonAnnotationInspectionResult(
        json_root=str(root),
        total_json_files=total_json_files,
        scanned_json_files=len(json_files_to_scan),
        valid_json_files=valid_json_files,
        invalid_json_files=invalid_json_files,
        empty_json_content_files=empty_json_content_files,
        scan_limited=max_files is not None and total_json_files > max_files,
        max_files=max_files,
        max_list_items_per_list=max_list_items_per_list,
        top_level_key_counts=dict(sorted(top_level_key_counts.items())),
        list_truncation_counts=dict(sorted(list_truncation_counts.items())),
        nested_key_path_counts=dict(sorted(nested_key_path_counts.items())),
        empty_list_path_counts=dict(sorted(empty_list_path_counts.items())),
        annotation_signal_counts=dict(sorted(annotation_signal_counts.items())),
        geometry_like_key_counts=dict(sorted(geometry_like_key_counts.items())),
        class_like_key_counts=dict(sorted(class_like_key_counts.items())),
        example_files=examples,
        read_errors=read_errors,
    )
