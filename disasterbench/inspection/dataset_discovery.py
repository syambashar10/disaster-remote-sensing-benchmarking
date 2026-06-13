from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
RASTER_EXTENSIONS = {".tif", ".tiff"}
JSON_EXTENSIONS = {".json", ".geojson"}
CSV_EXTENSIONS = {".csv"}
NPZ_EXTENSIONS = {".npz"}
VECTOR_EXTENSIONS = {".shp", ".gpkg", ".geojson"}

LABEL_FOLDER_HINTS = {
    "label",
    "labels",
    "mask",
    "masks",
    "floodmap",
    "floodmaps",
    "target",
    "targets",
    "annotation",
    "annotations",
}

IMAGE_FOLDER_HINTS = {
    "image",
    "images",
    "img",
    "imgs",
    "data",
    "s1",
    "s2",
    "sentinel1",
    "sentinel2",
    "s1_raw",
    "train_data",
    "test_data",
    "val_data",
}


@dataclass
class FileRecord:
    relative_path: str
    parent: str
    top_folder: str
    name: str
    stem: str
    extension: str
    size_bytes: int


def _should_skip_dir(dirname: str) -> bool:
    skip_names = {
        ".git",
        ".cache",
        "__pycache__",
        ".ipynb_checkpoints",
        "logs",
    }
    return dirname in skip_names


def scan_dataset_files(dataset_root: Path) -> list[FileRecord]:
    records: list[FileRecord] = []

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]

        current_dir = Path(dirpath)
        for filename in filenames:
            path = current_dir / filename

            try:
                stat = path.stat()
            except OSError:
                continue

            relative = path.relative_to(dataset_root)
            parent = str(relative.parent) if str(relative.parent) != "." else "."
            parts = relative.parts
            top_folder = parts[0] if len(parts) > 1 else "."

            records.append(
                FileRecord(
                    relative_path=str(relative),
                    parent=parent,
                    top_folder=top_folder,
                    name=path.name,
                    stem=path.stem,
                    extension=path.suffix.lower(),
                    size_bytes=stat.st_size,
                )
            )

    return records


def _normalize_stem(stem: str) -> str:
    suffixes = [
        "_mask",
        "_masks",
        "_label",
        "_labels",
        "_target",
        "_targets",
        "_floodmap",
        "_floodmaps",
        "_image",
        "_img",
    ]

    normalized = stem
    for suffix in suffixes:
        if normalized.lower().endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    return normalized


def _folder_hint_score(folder: str, hints: set[str]) -> int:
    folder_lower = folder.lower().replace("-", "_")
    score = 0
    for part in folder_lower.split("/"):
        for hint in hints:
            if hint == part or hint in part:
                score += 1
    return score


def summarize_files(records: list[FileRecord]) -> dict[str, Any]:
    extension_counts = Counter(record.extension or "<no_extension>" for record in records)
    top_folder_counts = Counter(record.top_folder for record in records)
    parent_counts = Counter(record.parent for record in records)

    total_size = sum(record.size_bytes for record in records)

    return {
        "total_files": len(records),
        "total_size_bytes": total_size,
        "extension_counts": dict(extension_counts.most_common()),
        "top_folder_counts": dict(top_folder_counts.most_common()),
        "largest_parent_folders": dict(parent_counts.most_common(30)),
    }


def find_candidate_directories(records: list[FileRecord]) -> dict[str, Any]:
    by_parent: dict[str, list[FileRecord]] = defaultdict(list)
    for record in records:
        by_parent[record.parent].append(record)

    candidates: dict[str, list[dict[str, Any]]] = {
        "image_like_dirs": [],
        "raster_like_dirs": [],
        "json_annotation_dirs": [],
        "csv_metadata_dirs": [],
        "npz_array_dirs": [],
        "possible_label_or_mask_dirs": [],
    }

    for parent, folder_records in by_parent.items():
        ext_counts = Counter(record.extension for record in folder_records)
        total = len(folder_records)

        folder_summary = {
            "path": parent,
            "file_count": total,
            "extension_counts": dict(ext_counts.most_common()),
        }

        if any(ext in IMAGE_EXTENSIONS for ext in ext_counts):
            item = dict(folder_summary)
            item["hint_score"] = _folder_hint_score(parent, IMAGE_FOLDER_HINTS)
            candidates["image_like_dirs"].append(item)

        if any(ext in RASTER_EXTENSIONS for ext in ext_counts):
            candidates["raster_like_dirs"].append(folder_summary)

        if any(ext in JSON_EXTENSIONS for ext in ext_counts):
            candidates["json_annotation_dirs"].append(folder_summary)

        if any(ext in CSV_EXTENSIONS for ext in ext_counts):
            candidates["csv_metadata_dirs"].append(folder_summary)

        if any(ext in NPZ_EXTENSIONS for ext in ext_counts):
            candidates["npz_array_dirs"].append(folder_summary)

        if _folder_hint_score(parent, LABEL_FOLDER_HINTS) > 0:
            item = dict(folder_summary)
            item["hint_score"] = _folder_hint_score(parent, LABEL_FOLDER_HINTS)
            candidates["possible_label_or_mask_dirs"].append(item)

    for key in candidates:
        candidates[key] = sorted(
            candidates[key],
            key=lambda item: (item.get("hint_score", 0), item["file_count"]),
            reverse=True,
        )[:50]

    return candidates


def suggest_pairings(records: list[FileRecord], max_suggestions: int = 30) -> list[dict[str, Any]]:
    by_parent: dict[str, list[FileRecord]] = defaultdict(list)
    for record in records:
        by_parent[record.parent].append(record)

    image_dirs: dict[str, list[FileRecord]] = {}
    candidate_dirs: dict[str, list[FileRecord]] = {}

    for parent, folder_records in by_parent.items():
        extensions = {record.extension for record in folder_records}
        label_score = _folder_hint_score(parent, LABEL_FOLDER_HINTS)

        if any(ext in IMAGE_EXTENSIONS for ext in extensions) and label_score == 0:
            image_dirs[parent] = [
                record for record in folder_records if record.extension in IMAGE_EXTENSIONS
            ]

        if (
            any(ext in JSON_EXTENSIONS for ext in extensions)
            or any(ext in RASTER_EXTENSIONS for ext in extensions)
            or label_score > 0
        ):
            candidate_dirs[parent] = [
                record
                for record in folder_records
                if record.extension in IMAGE_EXTENSIONS
                or record.extension in JSON_EXTENSIONS
                or record.extension in RASTER_EXTENSIONS
            ]

    suggestions: list[dict[str, Any]] = []

    for image_parent, image_records in image_dirs.items():
        image_stems = {_normalize_stem(record.stem) for record in image_records}

        for candidate_parent, candidate_records in candidate_dirs.items():
            if image_parent == candidate_parent:
                continue

            candidate_stems = {_normalize_stem(record.stem) for record in candidate_records}

            if not image_stems or not candidate_stems:
                continue

            matched = len(image_stems & candidate_stems)
            missing = len(image_stems - candidate_stems)
            extra = len(candidate_stems - image_stems)

            if matched == 0:
                continue

            score = matched / max(len(image_stems), 1)

            suggestions.append(
                {
                    "reference_root": image_parent,
                    "candidate_root": candidate_parent,
                    "reference_count": len(image_stems),
                    "candidate_count": len(candidate_stems),
                    "matched_count": matched,
                    "missing_count": missing,
                    "extra_count": extra,
                    "match_score": round(score, 4),
                    "reference_extensions": sorted(
                        {record.extension for record in image_records}
                    ),
                    "candidate_extensions": sorted(
                        {record.extension for record in candidate_records}
                    ),
                }
            )

    return sorted(
        suggestions,
        key=lambda item: (item["match_score"], item["matched_count"]),
        reverse=True,
    )[:max_suggestions]


def inspect_json_samples(dataset_root: Path, records: list[FileRecord], max_samples: int) -> list[dict[str, Any]]:
    samples = []
    json_records = [record for record in records if record.extension in JSON_EXTENSIONS]

    for record in json_records[:max_samples]:
        path = dataset_root / record.relative_path
        item: dict[str, Any] = {"path": record.relative_path}

        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            item["valid_json"] = True
            if isinstance(data, dict):
                item["top_level_type"] = "dict"
                item["top_level_keys"] = list(data.keys())[:30]
            elif isinstance(data, list):
                item["top_level_type"] = "list"
                item["list_length"] = len(data)
            else:
                item["top_level_type"] = type(data).__name__
        except Exception as exc:
            item["valid_json"] = False
            item["error"] = str(exc)

        samples.append(item)

    return samples


def inspect_npz_samples(dataset_root: Path, records: list[FileRecord], max_samples: int) -> list[dict[str, Any]]:
    samples = []
    npz_records = [record for record in records if record.extension in NPZ_EXTENSIONS]

    if not npz_records:
        return samples

    try:
        import numpy as np
    except Exception as exc:
        return [{"error": f"numpy unavailable: {exc}"}]

    for record in npz_records[:max_samples]:
        path = dataset_root / record.relative_path
        item: dict[str, Any] = {"path": record.relative_path}

        try:
            with np.load(path, allow_pickle=False) as data:
                item["keys"] = list(data.keys())
                item["arrays"] = {
                    key: {
                        "shape": list(data[key].shape),
                        "dtype": str(data[key].dtype),
                    }
                    for key in data.keys()
                }
        except Exception as exc:
            item["error"] = str(exc)

        samples.append(item)

    return samples


def inspect_raster_samples(dataset_root: Path, records: list[FileRecord], max_samples: int) -> list[dict[str, Any]]:
    samples = []
    raster_records = [record for record in records if record.extension in RASTER_EXTENSIONS]

    if not raster_records:
        return samples

    try:
        import rasterio
    except Exception as exc:
        return [{"error": f"rasterio unavailable: {exc}"}]

    for record in raster_records[:max_samples]:
        path = dataset_root / record.relative_path
        item: dict[str, Any] = {"path": record.relative_path}

        try:
            with rasterio.open(path) as src:
                item.update(
                    {
                        "width": src.width,
                        "height": src.height,
                        "band_count": src.count,
                        "dtypes": list(src.dtypes),
                        "crs": str(src.crs) if src.crs else None,
                        "nodata": src.nodata,
                    }
                )
        except Exception as exc:
            item["error"] = str(exc)

        samples.append(item)

    return samples


def infer_task_hints(records: list[FileRecord], candidates: dict[str, Any]) -> list[str]:
    extensions = {record.extension for record in records}
    task_hints = []

    if candidates["json_annotation_dirs"]:
        task_hints.extend(["object_detection_or_instance_segmentation", "metadata_or_polygon_annotations"])

    if candidates["possible_label_or_mask_dirs"] and any(ext in RASTER_EXTENSIONS for ext in extensions):
        task_hints.append("semantic_segmentation_or_raster_mask_mapping")

    if candidates["npz_array_dirs"]:
        task_hints.append("array_package_requires_npz_key_inspection")

    if any("instruction" in record.relative_path.lower() for record in records):
        task_hints.append("vision_language_instruction_dataset")

    if not task_hints:
        task_hints.append("manual_review_required")

    return sorted(set(task_hints))


def build_suggested_config(
    dataset_id: str,
    dataset_root: Path,
    candidates: dict[str, Any],
    pairing_suggestions: list[dict[str, Any]],
    task_hints: list[str],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset_root),
        "status": "auto_discovered_needs_human_review",
        "task_hints": task_hints,
        "candidate_directories": candidates,
        "pairing_suggestions": pairing_suggestions,
        "human_review_required": True,
        "manual_review_items": [
            "Confirm official dataset source and license.",
            "Confirm which folders represent images, masks, labels, metadata, and splits.",
            "Confirm label meanings and mapping to standard taxonomy.",
            "Confirm which export formats are valid, derived, lossy, or unsupported.",
        ],
    }


def write_inventory_csv(output_path: Path, records: list[FileRecord]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "parent",
                "top_folder",
                "name",
                "stem",
                "extension",
                "size_bytes",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def write_review_notes(output_path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Dataset Discovery Review Notes: {report['dataset_id']}",
        "",
        f"Dataset root: `{report['dataset_root']}`",
        "",
        "## Summary",
        "",
        f"- Total files: {report['file_summary']['total_files']}",
        f"- Total size bytes: {report['file_summary']['total_size_bytes']}",
        f"- Human review required: {report['human_review_required']}",
        "",
        "## Extension Counts",
        "",
    ]

    for ext, count in report["file_summary"]["extension_counts"].items():
        lines.append(f"- `{ext}`: {count}")

    lines.extend(
        [
            "",
            "## Task Hints",
            "",
        ]
    )

    for hint in report["task_hints"]:
        lines.append(f"- {hint}")

    lines.extend(
        [
            "",
            "## Top Pairing Suggestions",
            "",
        ]
    )

    for item in report["pairing_suggestions"][:10]:
        lines.append(
            f"- `{item['reference_root']}` -> `{item['candidate_root']}` "
            f"(matched={item['matched_count']}, score={item['match_score']})"
        )

    lines.extend(
        [
            "",
            "## Required Human Review",
            "",
            "- Confirm folder meanings.",
            "- Confirm label meanings.",
            "- Confirm task type.",
            "- Confirm supported export formats.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover_dataset(
    dataset_root: Path,
    dataset_id: str,
    output_root: Path,
    max_samples: int = 5,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_dir = output_root / f"{dataset_id}_discovery"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    records = scan_dataset_files(dataset_root)
    file_summary = summarize_files(records)
    candidates = find_candidate_directories(records)
    pairing_suggestions = suggest_pairings(records)
    task_hints = infer_task_hints(records, candidates)

    report: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "file_summary": file_summary,
        "candidate_directories": candidates,
        "pairing_suggestions": pairing_suggestions,
        "task_hints": task_hints,
        "sample_inspection": {
            "json_samples": inspect_json_samples(dataset_root, records, max_samples),
            "npz_samples": inspect_npz_samples(dataset_root, records, max_samples),
            "raster_samples": inspect_raster_samples(dataset_root, records, max_samples),
        },
        "human_review_required": True,
    }

    suggested_config = build_suggested_config(
        dataset_id=dataset_id,
        dataset_root=dataset_root,
        candidates=candidates,
        pairing_suggestions=pairing_suggestions,
        task_hints=task_hints,
    )

    (output_dir / "discovery_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "suggested_config.json").write_text(
        json.dumps(suggested_config, indent=2),
        encoding="utf-8",
    )
    write_inventory_csv(output_dir / "file_inventory.csv", records)
    write_review_notes(output_dir / "review_notes.md", report)

    return report
