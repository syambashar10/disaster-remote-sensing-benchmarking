import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_export_common_manifest_cli_with_npz_dataset(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    scene1 = dataset_root / "scene1"
    scene1.mkdir(parents=True)

    for idx in range(2):
        image = np.zeros((3, 16, 16), dtype=np.int16)
        label = np.zeros((16, 16), dtype=np.uint8)
        label[0:2, 0:2] = 1
        np.savez(scene1 / f"sample_{idx}.npz", image=image, label=label)

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_id": "toy_manifest_npz",
                "dataset_name": "Toy Manifest NPZ",
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

    output_jsonl = tmp_path / "manifest.jsonl"
    summary_output = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.export_common_manifest",
            "--config",
            str(config_path),
            "--output-jsonl",
            str(output_jsonl),
            "--summary-output",
            str(summary_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = output_jsonl.read_text(encoding="utf-8").strip().splitlines()
    summary = json.loads(summary_output.read_text(encoding="utf-8"))

    assert len(lines) == 2
    assert summary["exported_count"] == 2
    assert summary["error_count"] == 0
    assert summary["valid"] is True

    first_sample = json.loads(lines[0])
    assert first_sample["dataset_id"] == "toy_manifest_npz"
    assert first_sample["task_type"] == "semantic_segmentation"
    assert len(first_sample["annotations"]) == 1
