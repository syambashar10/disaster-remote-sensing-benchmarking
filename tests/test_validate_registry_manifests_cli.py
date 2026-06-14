import json
import subprocess
import sys
from pathlib import Path


def test_validate_registry_manifests_cli(tmp_path: Path):
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()

    sample = {
        "dataset_id": "toy",
        "sample_id": "sample_001",
        "task_type": "semantic_segmentation",
        "image": {
            "path": "image.tif",
            "width": 16,
            "height": 16,
            "band_count": 1,
            "dtypes": ["uint8"],
            "crs": None,
            "bounds": None
        },
        "annotations": [
            {
                "type": "segmentation_mask",
                "geometry_type": "raster_mask",
                "mask_path": "mask.tif",
                "mask_values": {
                    "0": "background",
                    "1": "positive"
                }
            }
        ],
        "metadata": {
            "source_format": "test"
        }
    }

    manifest_path = manifest_root / "toy_common_manifest.jsonl"
    manifest_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_name": "Toy Registry",
                "version": "0.1",
                "datasets": [
                    {
                        "dataset_id": "toy",
                        "config_path": "configs/toy.json",
                        "loader_family": "RasterMaskPairLoader"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "validation.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.validate_registry_manifests",
            "--registry",
            str(registry_path),
            "--manifest-root",
            str(manifest_root),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["valid"] is True
    assert report["dataset_count"] == 1
    assert report["valid_dataset_count"] == 1
    assert report["invalid_dataset_count"] == 0
    assert report["results"][0]["valid_samples"] == 1


def test_validate_registry_manifests_cli_detects_missing_manifest(tmp_path: Path):
    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_name": "Toy Registry",
                "version": "0.1",
                "datasets": [
                    {
                        "dataset_id": "missing_dataset",
                        "config_path": "configs/missing.json",
                        "loader_family": "RasterMaskPairLoader"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "validation.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.validate_registry_manifests",
            "--registry",
            str(registry_path),
            "--manifest-root",
            str(manifest_root),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["valid"] is False
    assert report["invalid_dataset_count"] == 1
    assert report["results"][0]["error"] == "manifest_file_not_found"
