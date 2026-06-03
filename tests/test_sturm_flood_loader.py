import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path

from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


def test_sensor(sensor: str):
    loader = STURMFloodLoader(sensor=sensor)

    pairing = loader.validate_pairing()
    statistics = loader.get_statistics()
    sample_0 = loader.load_sample(0)

    assert pairing["images_without_masks"] == 0
    assert pairing["masks_without_images"] == 0
    assert pairing["images_without_metadata"] == 0
    assert pairing["metadata_without_images"] == 0

    if sensor == "sentinel1":
        assert pairing["indexed_samples"] == 21602
        assert sample_0["image"]["band_count"] == 2

    if sensor == "sentinel2":
        assert pairing["indexed_samples"] == 2675
        assert sample_0["image"]["band_count"] == 9

    assert sample_0["dataset_id"] == "sturm_flood"
    assert sample_0["task_type"] == "semantic_segmentation"
    assert len(sample_0["annotations"]) == 1
    assert sample_0["annotations"][0]["geometry_type"] == "raster_mask"

    return {
        "pairing": pairing,
        "statistics": statistics,
        "sample_0": sample_0,
    }


def main():
    output = {
        "sentinel1": test_sensor("sentinel1"),
        "sentinel2": test_sensor("sentinel2"),
    }

    out_path = Path("outputs/verification/sturm_flood_loader_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("STURMFloodLoader tests passed.")
    print(f"Saved test output to: {out_path}")


if __name__ == "__main__":
    main()
