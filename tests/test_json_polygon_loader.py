import json
from pathlib import Path

from PIL import Image

from disasterbench.loaders.json_polygon_loader import JsonPolygonLoader


def _write_config(config_path: Path, dataset_root: Path) -> None:
    config = {
        "dataset_id": "toy_json_polygon",
        "dataset_name": "Toy JSON Polygon Dataset",
        "dataset_path": str(dataset_root),
        "loader_name": "JsonPolygonLoader",
        "task_type": ["object_detection", "instance_segmentation"],
        "data_family": "json_polygon_annotations",
        "raw_annotation_type": "polygon",
        "feature_type": "building",
        "split_structure": {
            "train_images": "images",
            "train_labels": "labels",
        },
        "json_polygon_schema": {
            "image_folder": "images",
            "label_folder": "labels",
            "feature_container": "features.xy",
            "wkt_key": "wkt",
            "properties_key": "properties",
            "class_key": "feature_type",
            "damage_key": "subtype",
            "uid_key": "uid",
        },
    }

    config_path.write_text(json.dumps(config), encoding="utf-8")


def test_json_polygon_loader_loads_common_sample(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    Image.new("RGB", (64, 64)).save(images_dir / "sample_001.png")

    label = {
        "metadata": {
            "disaster": "toy_disaster",
            "width": 64,
            "height": 64,
        },
        "features": {
            "xy": [
                {
                    "properties": {
                        "feature_type": "building",
                        "subtype": "no-damage",
                        "uid": "building_001",
                    },
                    "wkt": "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
                }
            ]
        },
    }

    (labels_dir / "sample_001.json").write_text(json.dumps(label), encoding="utf-8")

    config_path = tmp_path / "config.json"
    _write_config(config_path, dataset_root)

    loader = JsonPolygonLoader(config_path=config_path)

    assert len(loader) == 1

    pairing = loader.validate_pairing()
    assert pairing["valid"] is True
    assert pairing["matched_count"] == 1

    sample = loader.load_sample(0)

    assert sample["dataset_id"] == "toy_json_polygon"
    assert sample["sample_id"] == "sample_001"
    assert sample["task_type"] == "instance_segmentation"
    assert sample["image"]["width"] == 64
    assert sample["image"]["height"] == 64

    annotation = sample["annotations"][0]
    assert annotation["type"] == "polygon"
    assert annotation["geometry_type"] == "polygon"
    assert annotation["class_name"] == "building"
    assert annotation["damage_class"] == "no-damage"
    assert annotation["bbox"] == [0.0, 0.0, 10.0, 10.0]


def test_json_polygon_loader_detects_missing_label(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    Image.new("RGB", (64, 64)).save(images_dir / "sample_001.png")

    config_path = tmp_path / "config.json"
    _write_config(config_path, dataset_root)

    loader = JsonPolygonLoader(config_path=config_path)

    pairing = loader.validate_pairing()

    assert pairing["valid"] is False
    assert pairing["missing_label_count"] == 1
