import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_export_registry_manifests_cli_with_npz_dataset(tmp_path: Path):
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
                "dataset_id": "toy_registry_manifest",
                "dataset_name": "Toy Registry Manifest",
                "dataset_path": str(dataset_root),
                "loader_name": "NPZSegmentationLoader",
                "task_type": ["semantic_segmentation"],
                "data_family": "npz_multiband_semantic_segmentation",
                "raw_annotation_type": "npz_array_mask",
                "split_structure": {"scene1": "scene1"},
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

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_name": "Toy Registry",
                "version": "0.1",
                "datasets": [
                    {
                        "dataset_id": "toy_registry_manifest",
                        "config_path": str(config_path),
                        "status": "configured_loadable",
                        "loader_family": "NPZSegmentationLoader"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "exports"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.export_registry_manifests",
            "--registry",
            str(registry_path),
            "--output-root",
            str(output_root),
            "--max-samples",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_path = output_root / "toy_registry_manifest_common_manifest.jsonl"
    summary_path = output_root / "registry_common_manifest_export_summary.json"

    assert manifest_path.exists()
    assert summary_path.exists()

    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    report = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(lines) == 1
    assert report["dataset_count"] == 1
    assert report["valid_dataset_count"] == 1

    sample = json.loads(lines[0])
    assert sample["dataset_id"] == "toy_registry_manifest"
