import json
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

from disasterbench.pipelines import run_dataset_verification_pipeline


def write_tif(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as dst:
        dst.write(array, 1)


def test_dataset_verification_pipeline_runs_configured_checks(tmp_path):
    root = tmp_path / "dataset"

    images = root / "images"
    labels = root / "labels"
    masks = root / "masks"

    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    masks.mkdir(parents=True)

    (images / "sample_001.png").write_bytes(b"image")
    (labels / "sample_001.json").write_text("{}", encoding="utf-8")

    write_tif(root / "rasters" / "sample_001.tif", np.ones((2, 2), dtype=np.uint8))
    write_tif(root / "mask_rasters" / "sample_001.tif", np.zeros((2, 2), dtype=np.uint8))

    config = {
        "dataset_id": "fake_dataset",
        "dataset_path": str(root),
        "task_type": "semantic_segmentation",
        "pairing_checks": [
            {
                "name": "image_label_pairing",
                "reference_root": "images",
                "reference_extensions": [".png"],
                "candidate_roots": {
                    "labels": "labels"
                },
                "candidate_extensions": {
                    "labels": [".json"]
                }
            }
        ],
        "paired_raster_checks": [
            {
                "name": "raster_mask_alignment",
                "reference_root": "rasters",
                "candidate_root": "mask_rasters"
            }
        ]
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = run_dataset_verification_pipeline(
        config_path=config_path,
        output_root=tmp_path / "runs",
        run_id="pipeline_test",
    )

    check_statuses = {check.name: check.status for check in result.checks}

    assert result.dataset_id == "fake_dataset"
    assert check_statuses["generic_inspection"] == "passed"
    assert check_statuses["image_label_pairing"] == "passed"
    assert check_statuses["raster_mask_alignment"] == "passed"
    assert (tmp_path / "runs" / "pipeline_test" / "verification_report.json").exists()


def test_verify_dataset_cli_outputs_report(tmp_path):
    root = tmp_path / "dataset"
    images = root / "images"
    labels = root / "labels"

    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (images / "sample_001.png").write_bytes(b"image")
    (labels / "sample_001.json").write_text("{}", encoding="utf-8")

    config = {
        "dataset_id": "fake_dataset",
        "dataset_path": str(root),
        "task_type": "object_detection",
        "pairing_checks": [
            {
                "name": "image_label_pairing",
                "reference_root": "images",
                "reference_extensions": [".png"],
                "candidate_roots": {
                    "labels": "labels"
                },
                "candidate_extensions": {
                    "labels": [".json"]
                }
            }
        ]
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.verify_dataset",
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "cli_pipeline_test",
            "--max-pairs",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    data = json.loads(completed.stdout)

    assert data["dataset_id"] == "fake_dataset"
    assert data["run_id"] == "cli_pipeline_test"
    assert any(check["name"] == "image_label_pairing" for check in data["checks"])
    assert (tmp_path / "runs" / "cli_pipeline_test" / "verification_report.json").exists()
