import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def normalize_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_target"):
        stem = stem[:-7]
    return stem


def safe_read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inspect_image(path: Path):
    with Image.open(path) as img:
        arr = np.array(img)

    return {
        "path": str(path),
        "width": int(img.width),
        "height": int(img.height),
        "mode": img.mode,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min_value": int(arr.min()) if arr.size else None,
        "max_value": int(arr.max()) if arr.size else None
    }


def inspect_target(path: Path):
    with Image.open(path) as img:
        arr = np.array(img)

    unique_values = sorted(int(v) for v in np.unique(arr))

    return {
        "path": str(path),
        "width": int(img.width),
        "height": int(img.height),
        "mode": img.mode,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "unique_values": unique_values[:50],
        "unique_value_count": len(unique_values)
    }


def verify_xbd(dataset_root: Path, sample_limit: int | None):
    train_root = dataset_root / "train"
    images_dir = train_root / "images"
    labels_dir = train_root / "labels"
    targets_dir = train_root / "targets"

    image_files = sorted(images_dir.glob("*.png"))
    label_files = sorted(labels_dir.glob("*.json"))
    target_files = sorted(targets_dir.glob("*.png"))

    image_stems = {normalize_stem(p): p for p in image_files}
    label_stems = {normalize_stem(p): p for p in label_files}
    target_stems = {normalize_stem(p): p for p in target_files}

    common_stems = sorted(set(image_stems) & set(label_stems))

    if sample_limit is not None:
        stems_to_scan = common_stems[:sample_limit]
    else:
        stems_to_scan = common_stems

    report = {
        "dataset_id": "xbd",
        "verification_rule": "real files inspected before loader implementation",
        "dataset_root": str(dataset_root),
        "paths": {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "targets_dir": str(targets_dir)
        },
        "exists": {
            "images_dir": images_dir.exists(),
            "labels_dir": labels_dir.exists(),
            "targets_dir": targets_dir.exists()
        },
        "counts": {
            "image_files": len(image_files),
            "label_files": len(label_files),
            "target_files": len(target_files)
        },
        "pairing": {
            "images_without_labels": sorted(set(image_stems) - set(label_stems))[:100],
            "labels_without_images": sorted(set(label_stems) - set(image_stems))[:100],
            "images_without_targets": sorted(set(image_stems) - set(target_stems))[:100],
            "targets_without_images": sorted(set(target_stems) - set(image_stems))[:100],
            "images_without_labels_count": len(set(image_stems) - set(label_stems)),
            "labels_without_images_count": len(set(label_stems) - set(image_stems)),
            "images_without_targets_count": len(set(image_stems) - set(target_stems)),
            "targets_without_images_count": len(set(target_stems) - set(image_stems))
        },
        "json_scan": {
            "scanned_files": len(stems_to_scan),
            "read_errors": [],
            "files_with_xy": 0,
            "files_with_empty_xy": 0,
            "files_with_lng_lat": 0,
            "total_xy_objects": 0,
            "feature_type_counts": {},
            "damage_class_counts": {},
            "metadata_key_counts": {},
            "object_count_min": None,
            "object_count_max": None,
            "object_count_average": None
        },
        "image_sample": None,
        "target_sample": None,
        "example_files": {
            "first_5_images": [p.name for p in image_files[:5]],
            "first_5_labels": [p.name for p in label_files[:5]],
            "first_5_targets": [p.name for p in target_files[:5]]
        }
    }

    feature_type_counts = Counter()
    damage_class_counts = Counter()
    metadata_key_counts = Counter()
    object_counts = []

    for stem in stems_to_scan:
        label_path = label_stems[stem]

        try:
            data = safe_read_json(label_path)
        except Exception as exc:
            report["json_scan"]["read_errors"].append(
                {"file": str(label_path), "error": str(exc)}
            )
            continue

        metadata = data.get("metadata", {})
        for key in metadata.keys():
            metadata_key_counts[key] += 1

        features = data.get("features", {})
        xy_objects = features.get("xy", []) or []
        lng_lat_objects = features.get("lng_lat", []) or []

        if xy_objects:
            report["json_scan"]["files_with_xy"] += 1
        else:
            report["json_scan"]["files_with_empty_xy"] += 1

        if lng_lat_objects:
            report["json_scan"]["files_with_lng_lat"] += 1

        report["json_scan"]["total_xy_objects"] += len(xy_objects)
        object_counts.append(len(xy_objects))

        for obj in xy_objects:
            properties = obj.get("properties", {})
            feature_type_counts[properties.get("feature_type", "missing_feature_type")] += 1
            damage_class_counts[properties.get("subtype", "no_damage_label")] += 1

    report["json_scan"]["feature_type_counts"] = dict(feature_type_counts)
    report["json_scan"]["damage_class_counts"] = dict(damage_class_counts)
    report["json_scan"]["metadata_key_counts"] = dict(metadata_key_counts)

    if object_counts:
        report["json_scan"]["object_count_min"] = int(min(object_counts))
        report["json_scan"]["object_count_max"] = int(max(object_counts))
        report["json_scan"]["object_count_average"] = float(round(sum(object_counts) / len(object_counts), 4))

    if image_files:
        report["image_sample"] = inspect_image(image_files[0])

    if target_files:
        report["target_sample"] = inspect_target(target_files[0])

    return report


def main():
    parser = argparse.ArgumentParser(description="Verify xBD dataset before loader implementation.")
    parser.add_argument("--dataset-root", default="datasets/xbd_raw")
    parser.add_argument("--sample-limit", default=None, type=int)
    parser.add_argument("--output", default="outputs/verification/xbd_verification_report.json")
    args = parser.parse_args()

    report = verify_xbd(
        dataset_root=Path(args.dataset_root),
        sample_limit=args.sample_limit
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("xBD verification completed.")
    print(f"Report saved to: {output_path}")

    print("\nCounts:")
    print(report["counts"])

    print("\nPairing:")
    print({
        "images_without_labels": report["pairing"]["images_without_labels_count"],
        "labels_without_images": report["pairing"]["labels_without_images_count"],
        "images_without_targets": report["pairing"]["images_without_targets_count"],
        "targets_without_images": report["pairing"]["targets_without_images_count"]
    })

    print("\nJSON scan:")
    print({
        "scanned_files": report["json_scan"]["scanned_files"],
        "read_errors": len(report["json_scan"]["read_errors"]),
        "files_with_xy": report["json_scan"]["files_with_xy"],
        "files_with_empty_xy": report["json_scan"]["files_with_empty_xy"],
        "total_xy_objects": report["json_scan"]["total_xy_objects"],
        "object_count_average": report["json_scan"]["object_count_average"]
    })

    print("\nDamage classes:")
    print(report["json_scan"]["damage_class_counts"])


if __name__ == "__main__":
    main()
