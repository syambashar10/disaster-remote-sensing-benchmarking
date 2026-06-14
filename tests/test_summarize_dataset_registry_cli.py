import json
import subprocess
import sys
from pathlib import Path


def test_summarize_dataset_registry_cli(tmp_path: Path):
    config_path = tmp_path / "toy_config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "toy",
                "dataset_name": "Toy Dataset",
                "loader_name": "RasterMaskPairLoader",
                "data_family": "raster_image_mask_pair",
                "raw_annotation_type": "raster_mask",
                "task_type": ["semantic_segmentation"],
                "status": "configured_loadable",
                "human_review_required": True,
                "recommended_formats": ["semantic_mask"],
                "lossy_or_derived_formats": ["bbox_from_mask"],
                "unsupported_formats": ["native_yolo_detection"],
                "label_schema": {
                    "label_values": {"0": "background", "1": "positive"},
                    "notes": "Toy label schema."
                }
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
                        "status": "configured_loadable",
                        "loader_family": "RasterMaskPairLoader"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "summary.md"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.summarize_dataset_registry",
            "--registry",
            str(registry_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    text = output_path.read_text(encoding="utf-8")

    assert "# Toy Registry" in text
    assert "toy" in text
    assert "RasterMaskPairLoader" in text
    assert "semantic_mask" in text
