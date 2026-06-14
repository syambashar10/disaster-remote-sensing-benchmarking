import json
from pathlib import Path

import numpy as np

from disasterbench.loaders.npz_segmentation_loader import NPZSegmentationLoader


def _write_config(config_path: Path, dataset_root: Path) -> None:
    config = {
        "dataset_id": "toy_npz",
        "dataset_name": "Toy NPZ Segmentation",
        "dataset_path": str(dataset_root),
        "task_type": ["semantic_segmentation"],
        "data_family": "npz_multiband_semantic_segmentation",
        "raw_annotation_type": "npz_array_mask",
        "split_structure": {
            "scene1": "scene1",
            "scene2": "scene2",
        },
        "npz_schema": {
            "image_key": "image",
            "auxiliary_keys": ["aerosol"],
            "label_key": "label",
            "image_shape": [12, 32, 32],
            "label_shape": [32, 32],
            "image_dtype": "int16",
            "label_dtype": "uint8",
        },
        "label_schema": {
            "label_values": {
                "0": "background",
                "1": "positive",
            }
        },
    }

    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_npz_segmentation_loader_loads_common_sample(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    scene1 = dataset_root / "scene1"
    scene2 = dataset_root / "scene2"
    scene1.mkdir(parents=True)
    scene2.mkdir(parents=True)

    image = np.zeros((12, 32, 32), dtype=np.int16)
    aerosol = np.ones((32, 32), dtype=np.float32)
    label = np.zeros((32, 32), dtype=np.uint8)
    label[0:4, 0:4] = 1

    np.savez(scene1 / "patch_001.npz", image=image, aerosol=aerosol, label=label)
    np.savez(scene2 / "patch_002.npz", image=image, aerosol=aerosol, label=label)

    config_path = tmp_path / "config.json"
    _write_config(config_path, dataset_root)

    loader = NPZSegmentationLoader(config_path=config_path)

    assert len(loader) == 2

    samples = loader.list_samples()
    assert len(samples) == 2
    assert samples[0]["split"] == "scene1"

    pairing = loader.validate_pairing()
    assert pairing["valid"] is True
    assert pairing["total_npz_files"] == 2
    assert pairing["missing_image_key_count"] == 0
    assert pairing["missing_label_key_count"] == 0
    assert pairing["shape_mismatch_count"] == 0
    assert pairing["read_error_count"] == 0

    common_sample = loader.load_sample(0)

    assert common_sample["dataset_id"] == "toy_npz"
    assert common_sample["task_type"] == "semantic_segmentation"
    assert common_sample["image"]["width"] == 32
    assert common_sample["image"]["height"] == 32
    assert common_sample["image"]["band_count"] == 12
    assert common_sample["image"]["container_format"] == "npz"
    assert common_sample["image"]["array_key"] == "image"

    annotation = common_sample["annotations"][0]
    assert annotation["type"] == "segmentation_mask"
    assert annotation["geometry_type"] == "array_mask"
    assert annotation["container_format"] == "npz"
    assert annotation["label_key"] == "label"
    assert annotation["raw_label_values_found_in_file"] == [0, 1]


def test_npz_segmentation_loader_detects_missing_label_key(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    scene1 = dataset_root / "scene1"
    scene1.mkdir(parents=True)

    image = np.zeros((12, 32, 32), dtype=np.int16)

    np.savez(scene1 / "bad_patch.npz", image=image)

    config_path = tmp_path / "config.json"
    _write_config(config_path, dataset_root)

    loader = NPZSegmentationLoader(config_path=config_path)

    pairing = loader.validate_pairing()

    assert pairing["valid"] is False
    assert pairing["missing_label_key_count"] == 1
