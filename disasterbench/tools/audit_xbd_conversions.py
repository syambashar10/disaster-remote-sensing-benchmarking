import argparse
import csv
import json
from pathlib import Path

from disasterbench.loaders.xbd_loader import XBDLoader
from disasterbench.exporters.xbd_coco_exporter import DAMAGE_CLASSES


def bbox_from_polygon(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def close_enough(a, b, tolerance=1e-6):
    return abs(float(a) - float(b)) <= tolerance


def add_anomaly(anomalies, sample_id, annotation_id, issue, details):
    anomalies.append(
        {
            "sample_id": sample_id,
            "annotation_id": annotation_id,
            "issue": issue,
            "details": details,
        }
    )


def audit_xbd(max_samples=None, output_json=None, output_csv=None):
    loader = XBDLoader(split="train")

    total_samples = len(loader)
    sample_limit = total_samples if max_samples is None else min(max_samples, total_samples)

    anomalies = []

    summary = {
        "dataset_id": "xbd",
        "audit_name": "xbd_full_conversion_audit",
        "samples_checked": sample_limit,
        "annotations_checked": 0,
        "parse_errors_found": 0,
        "empty_samples": 0,
        "bbox_math_errors": 0,
        "bbox_enclosure_errors": 0,
        "bbox_bounds_warnings": 0,
        "invalid_polygon_warnings": 0,
        "invalid_geo_coordinate_errors": 0,
        "unknown_damage_class_warnings": 0,
        "status": "passed",
    }

    for index in range(sample_limit):
        if index > 0 and index % 500 == 0:
            print(f"[xbd audit] checked {index}/{sample_limit} samples")

        sample = loader.load_sample(index)
        sample_id = sample["sample_id"]

        image_width = sample["image"]["width"]
        image_height = sample["image"]["height"]

        if sample["annotation_count"] == 0:
            summary["empty_samples"] += 1

        if sample["parse_errors"]:
            summary["parse_errors_found"] += len(sample["parse_errors"])
            add_anomaly(
                anomalies,
                sample_id,
                "sample_level",
                "parse_errors",
                json.dumps(sample["parse_errors"]),
            )

        for annotation in sample["annotations"]:
            summary["annotations_checked"] += 1

            annotation_id = annotation["annotation_id"]
            polygon = annotation["polygon"]
            bbox = annotation["bbox"]
            geo_polygon = annotation.get("geo_polygon", [])
            damage_class = annotation.get("damage_class")

            if len(polygon) < 4:
                summary["invalid_polygon_warnings"] += 1
                add_anomaly(
                    anomalies,
                    sample_id,
                    annotation_id,
                    "polygon_too_short",
                    f"point_count={len(polygon)}",
                )
                continue

            x, y, w, h = bbox

            if w <= 0 or h <= 0:
                summary["bbox_math_errors"] += 1
                add_anomaly(
                    anomalies,
                    sample_id,
                    annotation_id,
                    "non_positive_bbox",
                    f"bbox={bbox}",
                )

            expected_bbox = bbox_from_polygon(polygon)

            bbox_matches = all(
                close_enough(actual, expected)
                for actual, expected in zip(bbox, expected_bbox)
            )

            if not bbox_matches:
                summary["bbox_math_errors"] += 1
                add_anomaly(
                    anomalies,
                    sample_id,
                    annotation_id,
                    "bbox_does_not_match_polygon_minmax",
                    f"actual={bbox}; expected={expected_bbox}",
                )

            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h

            for point in polygon:
                px, py = point

                if not (x_min - 1e-6 <= px <= x_max + 1e-6 and y_min - 1e-6 <= py <= y_max + 1e-6):
                    summary["bbox_enclosure_errors"] += 1
                    add_anomaly(
                        anomalies,
                        sample_id,
                        annotation_id,
                        "bbox_does_not_enclose_polygon_point",
                        f"point={point}; bbox={bbox}",
                    )
                    break

            # Slight tolerance is allowed because some polygons can lie exactly on the image edge.
            if x_min < -1 or y_min < -1 or x_max > image_width + 1 or y_max > image_height + 1:
                summary["bbox_bounds_warnings"] += 1
                add_anomaly(
                    anomalies,
                    sample_id,
                    annotation_id,
                    "bbox_outside_image_bounds",
                    f"bbox={bbox}; image_width={image_width}; image_height={image_height}",
                )

            if damage_class not in DAMAGE_CLASSES:
                summary["unknown_damage_class_warnings"] += 1
                add_anomaly(
                    anomalies,
                    sample_id,
                    annotation_id,
                    "unknown_damage_class",
                    f"damage_class={damage_class}",
                )

            if len(geo_polygon) < 4:
                summary["invalid_geo_coordinate_errors"] += 1
                add_anomaly(
                    anomalies,
                    sample_id,
                    annotation_id,
                    "geo_polygon_too_short",
                    f"point_count={len(geo_polygon)}",
                )
            else:
                for lon, lat in geo_polygon:
                    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                        summary["invalid_geo_coordinate_errors"] += 1
                        add_anomaly(
                            anomalies,
                            sample_id,
                            annotation_id,
                            "invalid_lng_lat_coordinate",
                            f"lon={lon}; lat={lat}",
                        )
                        break

    summary["total_anomalies"] = len(anomalies)

    if anomalies:
        summary["status"] = "issues_found"

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)

        with output_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": summary,
                    "first_100_anomalies": anomalies[:100],
                },
                f,
                indent=2,
            )

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "annotation_id", "issue", "details"],
            )
            writer.writeheader()
            writer.writerows(anomalies)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit xBD loader polygon/bbox/geo conversions.")
    parser.add_argument("--max-samples", default=None)
    parser.add_argument(
        "--output-json",
        default="outputs/verification/xbd_full_conversion_audit_summary.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/verification/xbd_full_conversion_audit_anomalies.csv",
    )
    args = parser.parse_args()

    max_samples = None

    if args.max_samples not in [None, "all", "none", "full"]:
        max_samples = int(args.max_samples)

    summary = audit_xbd(
        max_samples=max_samples,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )

    print("\nxBD conversion audit completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
