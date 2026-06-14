import json
import subprocess
import sys
from pathlib import Path


def test_validate_common_manifest_cli(tmp_path: Path):
    manifest_path = tmp_path / "manifest.jsonl"

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
            "bounds": None,
        },
        "annotations": [
            {
                "type": "segmentation_mask",
                "geometry_type": "raster_mask",
                "mask_path": "mask.tif",
                "mask_values": {
                    "0": "background",
                    "1": "positive",
                },
            }
        ],
        "metadata": {
            "source_format": "test",
        },
    }

    manifest_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    output_path = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.validate_common_manifest",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["valid"] is True
    assert report["total_lines"] == 1
    assert report["valid_samples"] == 1
    assert report["invalid_samples"] == 0
    assert report["dataset_ids"]["toy"] == 1
    assert report["task_types"]["semantic_segmentation"] == 1
    assert report["annotation_types"]["segmentation_mask"] == 1


def test_validate_common_manifest_cli_detects_invalid_sample(tmp_path: Path):
    manifest_path = tmp_path / "bad_manifest.jsonl"

    bad_sample = {
        "dataset_id": "toy",
        "sample_id": "sample_001"
    }

    manifest_path.write_text(json.dumps(bad_sample) + "\n", encoding="utf-8")

    output_path = tmp_path / "summary.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.validate_common_manifest",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["valid"] is False
    assert report["total_lines"] == 1
    assert report["valid_samples"] == 0
    assert report["invalid_samples"] == 1
    assert len(report["errors"]) == 1
