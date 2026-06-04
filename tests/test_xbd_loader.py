import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.loaders.xbd_loader import XBDLoader


def main():
    loader = XBDLoader()

    pairing = loader.validate_pairing()
    stats = loader.get_statistics()

    assert pairing["image_files"] == 5598
    assert pairing["label_files"] == 5598
    assert pairing["target_files"] == 5598
    assert pairing["images_without_labels"] == 0
    assert pairing["labels_without_images"] == 0
    assert pairing["images_without_targets"] == 0
    assert pairing["targets_without_images"] == 0
    assert pairing["indexed_samples"] == 5598

    sample_0 = loader.load_sample(0)

    assert sample_0["dataset_id"] == "xbd"
    assert sample_0["image"]["width"] == 1024
    assert sample_0["image"]["height"] == 1024
    assert sample_0["target"]["available"] is True
    assert "metadata" in sample_0
    assert "annotations" in sample_0

    non_empty_sample = None

    for index in range(min(len(loader), 500)):
        sample = loader.load_sample(index)

        if sample["annotation_count"] > 0:
            non_empty_sample = sample
            break

    if non_empty_sample is None:
        raise AssertionError("No non-empty xBD sample found in first 500 samples.")

    first_annotation = non_empty_sample["annotations"][0]

    assert first_annotation["annotation_type"] == "polygon"
    assert first_annotation["category"] == "building"
    assert len(first_annotation["polygon"]) >= 4
    assert len(first_annotation["bbox"]) == 4
    assert first_annotation["area_pixels"] > 0

    assert "geo_polygon" in first_annotation
    assert len(first_annotation["geo_polygon"]) >= 4

    lon, lat = first_annotation["geo_polygon"][0]
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90

    x, y, w, h = first_annotation["bbox"]

    assert x >= 0
    assert y >= 0
    assert w > 0
    assert h > 0
    assert x <= 1024
    assert y <= 1024
    assert x + w <= 1025
    assert y + h <= 1025

    output = {
        "test_name": "xbd_loader_test",
        "status": "passed",
        "statistics": stats,
        "sample_0": {
            "sample_id": sample_0["sample_id"],
            "annotation_count": sample_0["annotation_count"],
            "phase": sample_0["disaster_phase"],
        },
        "non_empty_sample": {
            "sample_id": non_empty_sample["sample_id"],
            "annotation_count": non_empty_sample["annotation_count"],
            "phase": non_empty_sample["disaster_phase"],
            "first_annotation_damage_class": first_annotation["damage_class"],
        },
    }

    output_path = Path("outputs/verification/xbd_loader_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("xBD loader test passed.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
