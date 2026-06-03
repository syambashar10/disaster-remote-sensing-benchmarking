import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio

from disasterbench.converters.mask_to_binary import mask_file_to_binary
from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


VALID_MASK_MODES = ["original", "binary_water"]


def export_single_mask(
    mask_path: str,
    output_path: Path,
    mode: str,
) -> Dict:
    """
    Export one STURM-Flood mask.

    mode:
    - original: keep original mask values
    - binary_water: convert to 0 non-water, 1 water, 255 no-data
    """
    if mode not in VALID_MASK_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {VALID_MASK_MODES}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mask_path) as src:
        profile = src.profile.copy()

        if mode == "original":
            output_mask = src.read(1)
            output_profile = profile.copy()

        else:
            output_mask = mask_file_to_binary(mask_path)
            output_profile = profile.copy()
            output_profile.update(
                dtype="uint8",
                count=1,
                nodata=255,
            )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(output_mask, 1)

    unique_values = sorted(int(value) for value in np.unique(output_mask))

    return {
        "output_path": str(output_path),
        "mode": mode,
        "unique_values": unique_values,
        "width": int(output_mask.shape[1]),
        "height": int(output_mask.shape[0]),
    }


def export_sturm_flood_masks(
    sensor: str,
    output_dir: str | Path,
    mode: str = "binary_water",
    max_samples: Optional[int] = 10,
) -> Dict:
    """
    Export STURM-Flood masks into a clean output folder.

    This is the most natural export for semantic segmentation workflows.

    Modes:
    - original: preserve original multiclass STURM mask values
    - binary_water: convert to binary water/non-water/no-data mask
    """
    if mode not in VALID_MASK_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from: {VALID_MASK_MODES}")

    loader = STURMFloodLoader(sensor=sensor)

    total_samples = len(loader.samples)
    sample_limit = total_samples if max_samples is None else min(max_samples, total_samples)

    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for index in range(sample_limit):
        sample = loader.load_sample(index)

        sample_id = sample["sample_id"]
        source_mask_path = sample["annotations"][0]["mask_path"]
        output_mask_path = masks_dir / f"{sample_id}.tif"

        export_info = export_single_mask(
            mask_path=source_mask_path,
            output_path=output_mask_path,
            mode=mode,
        )

        manifest.append(
            {
                "dataset_id": "sturm_flood",
                "sensor": sensor,
                "sample_id": sample_id,
                "image_path": sample["image"]["path"],
                "source_mask_path": source_mask_path,
                "exported_mask_path": str(output_mask_path),
                "mode": mode,
                "unique_values": export_info["unique_values"],
                "metadata": sample["metadata"],
            }
        )

    manifest_path = output_dir / "mask_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    class_mapping_path = output_dir / "class_mapping.json"

    if mode == "binary_water":
        class_mapping = {
            "0": "non_water",
            "1": "water",
            "255": "no_data",
            "note": "Binary mask derived from STURM-Flood values. Water = [1, 2, 3, 4, 5], non-water = [0], no-data = [99].",
        }
    else:
        class_mapping = {
            "0": "coastline_or_area_of_interest",
            "1": "flooded_area",
            "2": "river_related_features",
            "3": "open_water",
            "4": "reservoirs",
            "5": "lakes",
            "99": "no_data",
            "note": "Original STURM-Flood mask class values.",
        }

    with class_mapping_path.open("w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2)

    return {
        "output_dir": str(output_dir),
        "masks_dir": str(masks_dir),
        "manifest_path": str(manifest_path),
        "class_mapping_path": str(class_mapping_path),
        "sensor": sensor,
        "mode": mode,
        "samples_exported": sample_limit,
        "masks_exported": len(manifest),
    }
