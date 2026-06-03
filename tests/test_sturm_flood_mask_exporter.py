import json
import sys
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.exporters.mask_exporter import export_sturm_flood_masks


def validate_exported_masks(result, expected_count, mode):
    masks_dir = Path(result["masks_dir"])
    manifest_path = Path(result["manifest_path"])
    class_mapping_path = Path(result["class_mapping_path"])

    assert masks_dir.exists()
    assert manifest_path.exists()
    assert class_mapping_path.exists()

    mask_files = sorted(masks_dir.glob("*.tif"))
    assert len(mask_files) == expected_count

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == expected_count

    for mask_file in mask_files:
        with rasterio.open(mask_file) as src:
            mask = src.read(1)

            assert src.width == 128
            assert src.height == 128
            assert src.count == 1

            unique_values = set(int(value) for value in np.unique(mask))

            if mode == "binary_water":
                assert unique_values.issubset({0, 1, 255})
            else:
                assert unique_values.issubset({0, 1, 2, 3, 4, 5, 99})


def test_mask_export(sensor: str, mode: str, max_samples: int):
    output_dir = Path(f"outputs/masks/sturm_flood_{sensor}_{mode}_sample")

    result = export_sturm_flood_masks(
        sensor=sensor,
        output_dir=output_dir,
        mode=mode,
        max_samples=max_samples,
    )

    assert result["samples_exported"] == max_samples
    assert result["masks_exported"] == max_samples

    validate_exported_masks(
        result=result,
        expected_count=max_samples,
        mode=mode,
    )

    return result


def main():
    output = {
        "sentinel1_binary": test_mask_export("sentinel1", "binary_water", max_samples=20),
        "sentinel2_binary": test_mask_export("sentinel2", "binary_water", max_samples=20),
        "sentinel1_original": test_mask_export("sentinel1", "original", max_samples=20),
        "sentinel2_original": test_mask_export("sentinel2", "original", max_samples=20),
    }

    output_path = Path("outputs/verification/sturm_flood_mask_exporter_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("STURM-Flood mask exporter test passed.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
