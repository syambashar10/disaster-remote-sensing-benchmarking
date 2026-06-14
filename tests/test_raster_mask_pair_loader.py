import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from disasterbench.loaders.raster_mask_pair_loader import RasterMaskPairLoader


def _write_raster(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        count = 1
        height, width = array.shape
    else:
        count, height, width = array.shape

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        transform=from_origin(0, 10, 1, 1),
        crs="EPSG:4326",
    ) as dst:
        if array.ndim == 2:
            dst.write(array, 1)
        else:
            dst.write(array)


def _write_config(config_path: Path, dataset_root: Path) -> None:
    config = {
        "dataset_id": "toy_raster",
        "dataset_name": "Toy Raster Mask Dataset",
        "dataset_path": str(dataset_root),
        "task_type": ["semantic_segmentation"],
        "data_family": "raster_image_mask_pair",
        "raw_annotation_type": "raster_mask",
        "mask_mode": "binary",
        "label_schema": {
            "label_values": {
                "0": "background",
                "1": "positive",
            }
        },
        "pairing_groups": [
            {
                "name": "train",
                "split": "train",
                "image_folder": "train/images",
                "mask_folder": "train/masks",
                "image_extensions": [".tif"],
                "mask_extensions": [".tif"],
            }
        ],
    }

    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_raster_mask_pair_loader_loads_common_sample(tmp_path: Path):
    dataset_root = tmp_path / "dataset"

    image = np.zeros((3, 16, 16), dtype=np.uint16)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0:4, 0:4] = 1

    _write_raster(dataset_root / "train/images/sample_001.tif", image)
    _write_raster(dataset_root / "train/masks/sample_001.tif", mask)

    config_path = tmp_path / "config.json"
    _write_config(config_path, dataset_root)

    loader = RasterMaskPairLoader(config_path=config_path)

    assert len(loader) == 1

    pairing = loader.validate_pairing()
    assert pairing["valid"] is True
    assert pairing["total_matched_count"] == 1

    sample = loader.load_sample(0)

    assert sample["dataset_id"] == "toy_raster"
    assert sample["sample_id"] == "train__sample_001"
    assert sample["task_type"] == "semantic_segmentation"
    assert sample["image"]["width"] == 16
    assert sample["image"]["height"] == 16
    assert sample["image"]["band_count"] == 3

    annotation = sample["annotations"][0]
    assert annotation["type"] == "segmentation_mask"
    assert annotation["geometry_type"] == "raster_mask"
    assert annotation["raw_mask_values_found_in_file"] == [0, 1]


def test_raster_mask_pair_loader_detects_missing_mask(tmp_path: Path):
    dataset_root = tmp_path / "dataset"

    image = np.zeros((3, 16, 16), dtype=np.uint16)

    _write_raster(dataset_root / "train/images/sample_001.tif", image)

    config_path = tmp_path / "config.json"
    _write_config(config_path, dataset_root)

    loader = RasterMaskPairLoader(config_path=config_path)

    pairing = loader.validate_pairing()

    assert pairing["valid"] is False
    assert pairing["total_missing_mask_count"] == 1
    assert pairing["groups"][0]["missing_mask_count"] == 1


def test_raster_mask_pair_loader_expands_pairing_group_template(tmp_path: Path):
    dataset_root = tmp_path / "dataset"

    image = np.zeros((2, 16, 16), dtype=np.uint16)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0:2, 0:2] = 1

    _write_raster(dataset_root / "EVENT001/s1_raw/tile_001.tif", image)
    _write_raster(dataset_root / "EVENT001/mask/tile_001.tif", mask)

    config = {
        "dataset_id": "toy_template_raster",
        "dataset_name": "Toy Template Raster Dataset",
        "dataset_path": str(dataset_root),
        "task_type": ["semantic_segmentation"],
        "data_family": "raster_image_mask_pair",
        "raw_annotation_type": "raster_mask",
        "label_schema": {
            "label_values": {
                "0": "background",
                "1": "positive",
            }
        },
        "pairing_group_templates": [
            {
                "base_folder": ".",
                "event_glob": "EVENT*",
                "image_subfolder": "s1_raw",
                "mask_subfolder": "mask",
                "split": "all",
                "image_extensions": [".tif"],
                "mask_extensions": [".tif"]
            }
        ],
    }

    config_path = tmp_path / "template_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loader = RasterMaskPairLoader(config_path=config_path)

    assert len(loader) == 1

    pairing = loader.validate_pairing()
    assert pairing["valid"] is True
    assert pairing["group_count"] == 1
    assert pairing["total_matched_count"] == 1

    sample = loader.load_sample(0)
    assert sample["dataset_id"] == "toy_template_raster"
    assert sample["sample_id"] == "EVENT001__tile_001"
    assert sample["metadata"]["group"] == "EVENT001"
