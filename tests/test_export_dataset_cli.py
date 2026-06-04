import json
import subprocess
import sys
from pathlib import Path


def run_cli(command):
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return completed.stdout


def main():
    commands = {
        "xbd_geojson": [
            sys.executable, "-m", "disasterbench.tools.export_dataset",
            "--dataset", "xbd",
            "--format", "geojson",
            "--max-samples", "5",
            "--output", "outputs/cli_tests/xbd_cli.geojson",
        ],
        "xbd_coco": [
            sys.executable, "-m", "disasterbench.tools.export_dataset",
            "--dataset", "xbd",
            "--format", "coco_segmentation",
            "--max-samples", "5",
            "--output", "outputs/cli_tests/xbd_cli_coco.json",
        ],
        "xbd_yolo_detection": [
            sys.executable, "-m", "disasterbench.tools.export_dataset",
            "--dataset", "xbd",
            "--format", "yolo_detection_bbox",
            "--max-samples", "5",
            "--output", "outputs/cli_tests/xbd_cli_yolo_detection",
        ],
        "xbd_yolo_segmentation": [
            sys.executable, "-m", "disasterbench.tools.export_dataset",
            "--dataset", "xbd",
            "--format", "yolo_segmentation",
            "--max-samples", "5",
            "--output", "outputs/cli_tests/xbd_cli_yolo_segmentation",
        ],
        "sturm_geojson": [
            sys.executable, "-m", "disasterbench.tools.export_dataset",
            "--dataset", "sturm_flood",
            "--sensor", "sentinel1",
            "--format", "geojson",
            "--max-samples", "5",
            "--output", "outputs/cli_tests/sturm_cli.geojson",
        ],
    }

    results = {}

    for name, command in commands.items():
        stdout = run_cli(command)
        assert "Export completed." in stdout
        results[name] = {
            "status": "passed",
            "command": " ".join(command),
        }

    expected_failure = subprocess.run(
        [
            sys.executable, "-m", "disasterbench.tools.export_dataset",
            "--dataset", "xbd",
            "--format", "mask_segmentation",
            "--max-samples", "5",
        ],
        capture_output=True,
        text=True,
    )

    assert expected_failure.returncode != 0
    assert "xBD does not support mask_segmentation" in (
        expected_failure.stderr + expected_failure.stdout
    )

    results["xbd_mask_segmentation_expected_failure"] = {
        "status": "passed",
        "reason": "xBD correctly rejects mask_segmentation export",
    }

    verification_output = {
        "test_name": "export_dataset_cli_test",
        "status": "passed",
        "results": results,
    }

    verification_path = Path("outputs/verification/export_dataset_cli_test.json")
    verification_path.parent.mkdir(parents=True, exist_ok=True)

    with verification_path.open("w", encoding="utf-8") as f:
        json.dump(verification_output, f, indent=2)

    print("export_dataset CLI test passed.")
    print(json.dumps(verification_output, indent=2))


if __name__ == "__main__":
    main()
