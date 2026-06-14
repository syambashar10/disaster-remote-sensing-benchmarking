import json
from pathlib import Path

import pytest

from disasterbench.loaders.config_loader_factory import (
    create_loader_from_config,
    read_dataset_config,
    supported_loader_names,
)
from disasterbench.loaders.npz_segmentation_loader import NPZSegmentationLoader
from disasterbench.loaders.raster_mask_pair_loader import RasterMaskPairLoader


def test_supported_loader_names_contains_generic_loaders():
    names = supported_loader_names()

    assert "NPZSegmentationLoader" in names
    assert "RasterMaskPairLoader" in names


def test_read_dataset_config(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "toy",
                "loader_name": "NPZSegmentationLoader",
            }
        ),
        encoding="utf-8",
    )

    config = read_dataset_config(config_path)

    assert config["dataset_id"] == "toy"
    assert config["loader_name"] == "NPZSegmentationLoader"


def test_create_npz_loader_from_config(tmp_path: Path):
    dataset_root = tmp_path / "npz_dataset"
    scene1 = dataset_root / "scene1"
    scene1.mkdir(parents=True)

    config_path = tmp_path / "npz_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "toy_npz",
                "dataset_name": "Toy NPZ",
                "dataset_path": str(dataset_root),
                "loader_name": "NPZSegmentationLoader",
                "split_structure": {
                    "scene1": "scene1"
                },
                "npz_schema": {
                    "image_key": "image",
                    "label_key": "label",
                    "auxiliary_keys": []
                },
                "label_schema": {
                    "label_values": {
                        "0": "background",
                        "1": "positive"
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loader = create_loader_from_config(config_path)

    assert isinstance(loader, NPZSegmentationLoader)
    assert loader.dataset_id == "toy_npz"


def test_create_raster_loader_from_config(tmp_path: Path):
    dataset_root = tmp_path / "raster_dataset"
    (dataset_root / "train/images").mkdir(parents=True)
    (dataset_root / "train/masks").mkdir(parents=True)

    config_path = tmp_path / "raster_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "toy_raster",
                "dataset_name": "Toy Raster",
                "dataset_path": str(dataset_root),
                "loader_name": "RasterMaskPairLoader",
                "pairing_groups": [
                    {
                        "name": "train",
                        "split": "train",
                        "image_folder": "train/images",
                        "mask_folder": "train/masks",
                        "image_extensions": [".tif"],
                        "mask_extensions": [".tif"]
                    }
                ],
                "label_schema": {
                    "label_values": {
                        "0": "background",
                        "1": "positive"
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loader = create_loader_from_config(config_path)

    assert isinstance(loader, RasterMaskPairLoader)
    assert loader.dataset_id == "toy_raster"


def test_create_loader_from_config_rejects_unknown_loader(tmp_path: Path):
    config_path = tmp_path / "bad_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "bad",
                "loader_name": "UnknownLoader"
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported loader_name"):
        create_loader_from_config(config_path)
