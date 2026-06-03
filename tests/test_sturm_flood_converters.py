import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.converters.mask_to_binary import mask_file_to_binary, summarize_binary_mask
from disasterbench.converters.mask_to_polygon import mask_file_to_polygons
from disasterbench.converters.mask_to_bbox import mask_file_to_bboxes
from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


def validate_bbox(bbox, width=128, height=128):
    x, y, w, h = bbox

    if w < 0 or h < 0:
        return False

    if x < 0 or y < 0:
        return False

    if x > width or y > height:
        return False

    if x + w > width + 1:
        return False

    if y + h > height + 1:
        return False

    return True


def test_sensor(sensor, max_samples):
    loader = STURMFloodLoader(sensor=sensor)
    sample_count = len(loader.samples)

    if max_samples is None:
        indices = range(sample_count)
    else:
        indices = range(min(max_samples, sample_count))

    sensor_summary = {
        "sensor": sensor,
        "total_loader_samples": sample_count,
        "tested_samples": 0,
        "binary_unique_values_seen": set(),
        "total_polygons": 0,
        "total_bboxes": 0,
        "errors": [],
    }

    for index in indices:
        sample = loader.load_sample(index)
        mask_path = sample["annotations"][0]["mask_path"]

        try:
            binary = mask_file_to_binary(mask_path)
            unique_values = set(int(value) for value in np.unique(binary))
            sensor_summary["binary_unique_values_seen"].update(unique_values)

            invalid_binary_values = unique_values - {0, 1, 255}
            if invalid_binary_values:
                sensor_summary["errors"].append(
                    {
                        "sample_id": sample["sample_id"],
                        "issue": "invalid_binary_values",
                        "details": sorted(list(invalid_binary_values)),
                    }
                )

            polygons = mask_file_to_polygons(
                mask_path=mask_path,
                target_values=[1, 2, 3, 4, 5],
                min_area_pixels=2.0,
                coordinate_space="pixel",
            )

            bboxes = mask_file_to_bboxes(
                mask_path=mask_path,
                target_values=[1, 2, 3, 4, 5],
                min_area_pixels=2.0,
            )

            if len(polygons) != len(bboxes):
                sensor_summary["errors"].append(
                    {
                        "sample_id": sample["sample_id"],
                        "issue": "polygon_bbox_count_mismatch",
                        "details": f"{len(polygons)} polygons vs {len(bboxes)} bboxes",
                    }
                )

            for bbox_record in bboxes:
                if not validate_bbox(bbox_record["bbox"]):
                    sensor_summary["errors"].append(
                        {
                            "sample_id": sample["sample_id"],
                            "issue": "invalid_bbox",
                            "details": bbox_record["bbox"],
                        }
                    )

            sensor_summary["total_polygons"] += len(polygons)
            sensor_summary["total_bboxes"] += len(bboxes)
            sensor_summary["tested_samples"] += 1

        except Exception as error:
            sensor_summary["errors"].append(
                {
                    "sample_id": sample.get("sample_id", str(index)),
                    "issue": "converter_exception",
                    "details": str(error),
                }
            )

        if sensor_summary["tested_samples"] % 500 == 0:
            print(f"[{sensor}] tested {sensor_summary['tested_samples']} samples")

    sensor_summary["binary_unique_values_seen"] = sorted(
        int(value) for value in sensor_summary["binary_unique_values_seen"]
    )
    sensor_summary["error_count"] = len(sensor_summary["errors"])

    return sensor_summary


def main():
    parser = argparse.ArgumentParser(description="Test STURM-Flood mask converters.")
    parser.add_argument("--max-samples", type=int, default=20, help="Number of samples per sensor to test.")
    parser.add_argument("--all", action="store_true", help="Test all samples from both sensors.")
    args = parser.parse_args()

    max_samples = None if args.all else args.max_samples

    output = {
        "test_name": "sturm_flood_converter_tests",
        "max_samples_per_sensor": "all" if args.all else max_samples,
        "sentinel1": test_sensor("sentinel1", max_samples),
        "sentinel2": test_sensor("sentinel2", max_samples),
    }

    output_path = Path("outputs/verification/sturm_flood_converter_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSTURM-Flood converter test finished.")
    print(f"Saved output to: {output_path}")

    for sensor in ["sentinel1", "sentinel2"]:
        result = output[sensor]
        print(f"\n{sensor}")
        print(f"  Tested samples: {result['tested_samples']}")
        print(f"  Binary values seen: {result['binary_unique_values_seen']}")
        print(f"  Total polygons: {result['total_polygons']}")
        print(f"  Total bboxes: {result['total_bboxes']}")
        print(f"  Errors: {result['error_count']}")

    total_errors = output["sentinel1"]["error_count"] + output["sentinel2"]["error_count"]

    if total_errors > 0:
        raise SystemExit(f"Converter test found {total_errors} errors.")

    print("\nAll converter tests passed.")


if __name__ == "__main__":
    main()
