import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.loaders.xbd_loader import XBDLoader
from disasterbench.exporters.xbd_yolo_detection_exporter import (
    export_xbd_yolo_detection,
    yolo_detection_line,
)


def main():
    max_samples = 50
    output_dir = Path("outputs/yolo_detection/xbd_integrity_sample")

    result = export_xbd_yolo_detection(
        output_dir=output_dir,
        max_samples=max_samples,
        split="train",
    )

    loader = XBDLoader(split="train")
    labels_dir = output_dir / "labels"

    expected_total_lines = 0

    for index in range(max_samples):
        sample = loader.load_sample(index)
        sample_id = sample["sample_id"]

        expected_lines = [
            yolo_detection_line(
                annotation=annotation,
                image_width=sample["image"]["width"],
                image_height=sample["image"]["height"],
            )
            for annotation in sample["annotations"]
        ]

        label_path = labels_dir / f"{sample_id}.txt"

        assert label_path.exists(), f"Missing YOLO label file: {label_path}"

        with label_path.open("r", encoding="utf-8") as f:
            exported_lines = [line.strip() for line in f.readlines() if line.strip()]

        assert exported_lines == expected_lines, f"YOLO lines mismatch for sample: {sample_id}"

        expected_total_lines += len(expected_lines)

    assert expected_total_lines == result["label_lines_exported"]

    verification_output = {
        "test_name": "xbd_yolo_detection_integrity_test",
        "status": "passed",
        "samples_checked": max_samples,
        "label_lines_checked": expected_total_lines,
        "export_output": str(output_dir),
        "result": result,
    }

    verification_path = Path("outputs/verification/xbd_yolo_detection_integrity_test.json")
    verification_path.parent.mkdir(parents=True, exist_ok=True)

    with verification_path.open("w", encoding="utf-8") as f:
        json.dump(verification_output, f, indent=2)

    print("xBD YOLO detection integrity test passed.")
    print(json.dumps(verification_output, indent=2))


if __name__ == "__main__":
    main()
