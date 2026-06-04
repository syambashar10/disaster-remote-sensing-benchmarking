import argparse
import json
from pathlib import Path

from shapely import wkt


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def verify_alignment(dataset_root: Path, sample_limit=None):
    labels_dir = dataset_root / "train" / "labels"
    label_files = sorted(labels_dir.glob("*.json"))

    if sample_limit is not None:
        label_files = label_files[:sample_limit]

    report = {
        "dataset_id": "xbd",
        "check": "xy_lng_lat_geometry_alignment",
        "labels_checked": len(label_files),
        "files_with_xy_lng_lat_count_mismatch": [],
        "xy_wkt_parse_errors": [],
        "lng_lat_wkt_parse_errors": [],
        "total_xy_objects": 0,
        "total_lng_lat_objects": 0,
        "status": "passed"
    }

    for label_path in label_files:
        data = read_json(label_path)
        features = data.get("features", {})

        xy_objects = features.get("xy", []) or []
        lng_lat_objects = features.get("lng_lat", []) or []

        report["total_xy_objects"] += len(xy_objects)
        report["total_lng_lat_objects"] += len(lng_lat_objects)

        if len(xy_objects) != len(lng_lat_objects):
            report["files_with_xy_lng_lat_count_mismatch"].append(
                {
                    "file": str(label_path),
                    "xy_count": len(xy_objects),
                    "lng_lat_count": len(lng_lat_objects)
                }
            )

        for index, obj in enumerate(xy_objects):
            source_wkt = obj.get("wkt")
            if not source_wkt:
                continue

            try:
                wkt.loads(source_wkt)
            except Exception as exc:
                report["xy_wkt_parse_errors"].append(
                    {
                        "file": str(label_path),
                        "object_index": index,
                        "error": str(exc)
                    }
                )

        for index, obj in enumerate(lng_lat_objects):
            source_wkt = obj.get("wkt")
            if not source_wkt:
                continue

            try:
                wkt.loads(source_wkt)
            except Exception as exc:
                report["lng_lat_wkt_parse_errors"].append(
                    {
                        "file": str(label_path),
                        "object_index": index,
                        "error": str(exc)
                    }
                )

    if (
        report["files_with_xy_lng_lat_count_mismatch"]
        or report["xy_wkt_parse_errors"]
        or report["lng_lat_wkt_parse_errors"]
    ):
        report["status"] = "issues_found"

    return report


def main():
    parser = argparse.ArgumentParser(description="Verify xBD xy/lng_lat geometry alignment.")
    parser.add_argument("--dataset-root", default="datasets/xbd_raw")
    parser.add_argument("--sample-limit", default=None, type=int)
    parser.add_argument("--output", default="outputs/verification/xbd_geometry_alignment_report.json")
    args = parser.parse_args()

    report = verify_alignment(
        dataset_root=Path(args.dataset_root),
        sample_limit=args.sample_limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("xBD geometry alignment verification completed.")
    print(f"Report saved to: {output_path}")
    print("Status:", report["status"])
    print("Labels checked:", report["labels_checked"])
    print("Total xy objects:", report["total_xy_objects"])
    print("Total lng_lat objects:", report["total_lng_lat_objects"])
    print("Count mismatch files:", len(report["files_with_xy_lng_lat_count_mismatch"]))
    print("XY WKT parse errors:", len(report["xy_wkt_parse_errors"]))
    print("lng_lat WKT parse errors:", len(report["lng_lat_wkt_parse_errors"]))


if __name__ == "__main__":
    main()
