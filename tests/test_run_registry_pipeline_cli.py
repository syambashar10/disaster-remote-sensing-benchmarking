import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_run_registry_pipeline_cli_with_npz_dataset(tmp_path: Path):
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
                "dataset_id": "toy_pipeline",
                "dataset_name": "Toy Pipeline Dataset",
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
                },
                "recommended_formats": ["semantic_mask"],
                "lossy_or_derived_formats": ["bbox_from_mask"],
                "unsupported_formats": ["native_yolo_detection"]
            }
        ),
        encoding="utf-8",
    )

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "registry_name": "Toy Pipeline Registry",
                "version": "0.1",
                "datasets": [
                    {
                        "dataset_id": "toy_pipeline",
                        "config_path": str(config_path),
                        "status": "configured_loadable",
                        "loader_family": "NPZSegmentationLoader"
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "pipeline_outputs"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.run_registry_pipeline",
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

    final_report_path = output_root / "registry_pipeline_report.json"
    manifest_path = output_root / "common_manifests" / "toy_pipeline_common_manifest.jsonl"
    capability_md_path = output_root / "reports" / "dataset_capability_matrix.md"

    assert final_report_path.exists()
    assert manifest_path.exists()
    assert capability_md_path.exists()

    final_report = json.loads(final_report_path.read_text(encoding="utf-8"))

    assert final_report["valid"] is True
    assert final_report["dataset_count"] == 1
    assert final_report["dataset_config_validation"]["valid"] is True
    assert final_report["common_manifest_export"]["valid"] is True
    assert final_report["common_manifest_validation"]["valid"] is True

    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
