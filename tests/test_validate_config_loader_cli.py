import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_validate_config_loader_cli_with_npz_dataset(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    scene1 = dataset_root / "scene1"
    scene1.mkdir(parents=True)

    image = np.zeros((3, 16, 16), dtype=np.int16)
    label = np.zeros((16, 16), dtype=np.uint8)
    label[0:2, 0:2] = 1

    np.savez(scene1 / "sample_001.npz", image=image, label=label)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "toy_cli_npz",
                "dataset_name": "Toy CLI NPZ",
                "dataset_path": str(dataset_root),
                "loader_name": "NPZSegmentationLoader",
                "task_type": ["semantic_segmentation"],
                "data_family": "npz_multiband_semantic_segmentation",
                "raw_annotation_type": "npz_array_mask",
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

    output_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.validate_config_loader",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "NPZSegmentationLoader" in result.stdout
    assert output_path.exists()

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["valid"] is True
    assert report["loader_class"] == "NPZSegmentationLoader"
    assert report["sample_count"] == 1
    assert report["first_sample"]["dataset_id"] == "toy_cli_npz"
