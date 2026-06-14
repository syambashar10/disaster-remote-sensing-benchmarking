import json
import subprocess
import sys
from pathlib import Path

from disasterbench.capabilities.capability_matrix import build_capability_matrix


def test_build_capability_matrix(tmp_path: Path):
    config_path = tmp_path / "toy_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "toy",
                "dataset_name": "Toy Dataset",
                "loader_name": "RasterMaskPairLoader",
                "data_family": "raster_image_mask_pair",
                "recommended_formats": ["semantic_mask"],
                "lossy_or_derived_formats": ["bbox_from_mask"],
                "unsupported_formats": ["native_coco_instance_segmentation"]
            }
        ),
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_name": "Toy Registry",
                "version": "0.1",
                "datasets": [
                    {
                        "dataset_id": "toy",
                        "config_path": str(config_path),
                        "loader_family": "RasterMaskPairLoader"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_capability_matrix(registry_path)

    assert report["dataset_count"] == 1
    assert report["matrix"]["toy"]["semantic_mask"] == "supported"
    assert report["matrix"]["toy"]["bbox_from_mask"] == "lossy_or_derived"
    assert report["matrix"]["toy"]["native_coco_instance_segmentation"] == "unsupported"


def test_build_capability_matrix_cli(tmp_path: Path):
    config_path = tmp_path / "toy_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "toy",
                "dataset_name": "Toy Dataset",
                "loader_name": "NPZSegmentationLoader",
                "data_family": "npz_multiband_semantic_segmentation",
                "recommended_formats": ["semantic_mask"],
                "lossy_or_derived_formats": ["polygon_from_mask"],
                "unsupported_formats": ["native_yolo_detection"]
            }
        ),
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_name": "Toy Registry",
                "version": "0.1",
                "datasets": [
                    {
                        "dataset_id": "toy",
                        "config_path": str(config_path),
                        "loader_family": "NPZSegmentationLoader"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    json_output = tmp_path / "capabilities.json"
    markdown_output = tmp_path / "capabilities.md"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.build_capability_matrix",
            "--registry",
            str(registry_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json_output.exists()
    assert markdown_output.exists()

    markdown = markdown_output.read_text(encoding="utf-8")

    assert "Dataset Export Capability Matrix" in markdown
    assert "semantic_mask" in markdown
    assert "lossy_or_derived" in markdown
