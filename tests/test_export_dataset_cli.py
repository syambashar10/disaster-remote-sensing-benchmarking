import json
import subprocess
import sys
from pathlib import Path


def run_command(command):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise AssertionError(f"Command failed: {' '.join(command)}")

    return result.stdout


def test_command(export_format, output_path, extra_args=None):
    command = [
        sys.executable,
        "-m",
        "disasterbench.tools.export_dataset",
        "--dataset",
        "sturm_flood",
        "--sensor",
        "sentinel1",
        "--format",
        export_format,
        "--output",
        output_path,
        "--max-samples",
        "5",
    ]

    if extra_args:
        command.extend(extra_args)

    stdout = run_command(command)

    if "Export completed." not in stdout:
        raise AssertionError("Export completion message not found.")

    return stdout


def main():
    outputs = {
        "geojson": "outputs/cli_tests/sturm_s1.geojson",
        "coco_segmentation": "outputs/cli_tests/sturm_s1_coco.json",
        "yolo_segmentation": "outputs/cli_tests/yolo_segmentation_s1",
        "mask_segmentation": "outputs/cli_tests/masks_s1",
        "yolo_detection_bbox": "outputs/cli_tests/yolo_detection_s1",
    }

    results = {}

    results["geojson"] = test_command(
        "geojson",
        outputs["geojson"],
    )

    results["coco_segmentation"] = test_command(
        "coco_segmentation",
        outputs["coco_segmentation"],
    )

    results["yolo_segmentation"] = test_command(
        "yolo_segmentation",
        outputs["yolo_segmentation"],
    )

    results["mask_segmentation"] = test_command(
        "mask_segmentation",
        outputs["mask_segmentation"],
        extra_args=["--mask-mode", "binary_water"],
    )

    results["yolo_detection_bbox"] = test_command(
        "yolo_detection_bbox",
        outputs["yolo_detection_bbox"],
    )

    checks = [
        Path(outputs["geojson"]).exists(),
        Path(outputs["coco_segmentation"]).exists(),
        (Path(outputs["yolo_segmentation"]) / "labels").exists(),
        (Path(outputs["mask_segmentation"]) / "masks").exists(),
        (Path(outputs["yolo_detection_bbox"]) / "labels").exists(),
    ]

    if not all(checks):
        raise AssertionError("One or more CLI output files/folders were not created.")

    output_path = Path("outputs/verification/export_dataset_cli_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "test_name": "export_dataset_cli_test",
                "dataset": "sturm_flood",
                "sensor": "sentinel1",
                "formats_tested": list(outputs.keys()),
                "outputs": outputs,
                "status": "passed",
            },
            f,
            indent=2,
        )

    print("export_dataset CLI test passed.")
    print(f"Saved test result to: {output_path}")


if __name__ == "__main__":
    main()
