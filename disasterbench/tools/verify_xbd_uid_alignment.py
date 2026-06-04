import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_uid_alignment(dataset_root: Path, sample_limit=None):
    labels_dir = dataset_root / "train" / "labels"
    label_files = sorted(labels_dir.glob("*.json"))

    if sample_limit is not None:
        label_files = label_files[:sample_limit]

    report = {
        "dataset_id": "xbd",
        "check": "xy_lng_lat_uid_property_alignment",
        "labels_checked": len(label_files),
        "objects_checked": 0,
        "count_mismatch_files": [],
        "uid_mismatches": [],
        "feature_type_mismatches": [],
        "subtype_mismatches": [],
        "status": "passed",
    }

    for label_path in label_files:
        data = read_json(label_path)
        features = data.get("features", {})

        xy_objects = features.get("xy", []) or []
        lng_lat_objects = features.get("lng_lat", []) or []

        if len(xy_objects) != len(lng_lat_objects):
            report["count_mismatch_files"].append(
                {
                    "file": str(label_path),
                    "xy_count": len(xy_objects),
                    "lng_lat_count": len(lng_lat_objects),
                }
            )
            continue

        for index, (xy_obj, geo_obj) in enumerate(zip(xy_objects, lng_lat_objects)):
            report["objects_checked"] += 1

            xy_props = xy_obj.get("properties", {})
            geo_props = geo_obj.get("properties", {})

            if xy_props.get("uid") != geo_props.get("uid"):
                report["uid_mismatches"].append(
                    {
                        "file": str(label_path),
                        "object_index": index,
                        "xy_uid": xy_props.get("uid"),
                        "lng_lat_uid": geo_props.get("uid"),
                    }
                )

            if xy_props.get("feature_type") != geo_props.get("feature_type"):
                report["feature_type_mismatches"].append(
                    {
                        "file": str(label_path),
                        "object_index": index,
                        "xy_feature_type": xy_props.get("feature_type"),
                        "lng_lat_feature_type": geo_props.get("feature_type"),
                    }
                )

            if xy_props.get("subtype") != geo_props.get("subtype"):
                report["subtype_mismatches"].append(
                    {
                        "file": str(label_path),
                        "object_index": index,
                        "xy_subtype": xy_props.get("subtype"),
                        "lng_lat_subtype": geo_props.get("subtype"),
                    }
                )

    if (
        report["count_mismatch_files"]
        or report["uid_mismatches"]
        or report["feature_type_mismatches"]
        or report["subtype_mismatches"]
    ):
        report["status"] = "issues_found"

    return report


def main():
    parser = argparse.ArgumentParser(description="Verify xBD xy/lng_lat object UID/property alignment.")
    parser.add_argument("--dataset-root", default="datasets/xbd_raw")
    parser.add_argument("--sample-limit", default=None, type=int)
    parser.add_argument("--output", default="outputs/verification/xbd_uid_alignment_report.json")
    args = parser.parse_args()

    report = verify_uid_alignment(
        dataset_root=Path(args.dataset_root),
        sample_limit=args.sample_limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("xBD UID/property alignment verification completed.")
    print(f"Report saved to: {output_path}")
    print("Status:", report["status"])
    print("Labels checked:", report["labels_checked"])
    print("Objects checked:", report["objects_checked"])
    print("Count mismatch files:", len(report["count_mismatch_files"]))
    print("UID mismatches:", len(report["uid_mismatches"]))
    print("Feature type mismatches:", len(report["feature_type_mismatches"]))
    print("Subtype mismatches:", len(report["subtype_mismatches"]))


if __name__ == "__main__":
    main()
