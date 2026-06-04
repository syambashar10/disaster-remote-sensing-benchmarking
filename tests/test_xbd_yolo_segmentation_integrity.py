import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.loaders.xbd_loader import XBDLoader
from disasterbench.exporters.xbd_yolo_segmentation_exporter import (
    export_xbd_yolo_segmentation,
    yolo_segmentation_line,
)


def main():
    max_samples = 50
    output_dir = Path("outputs/yolo_segmentation/xbd_integrity_sample")

    result = export_xbd_yolo_segmentation(
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

        expected_lines = []

        for annotation in sample["annotations"]:
            line = yolo_segmentation_line(
                annotation=annotation,
                image_width=sample["image"]["width"],
                image_height=sample["image"]["height"],
            )

            parts = line.split()

            if len(parts) >= 7:
                expected_lines.append(line)

        label_path = labels_dir / f"{sample_id}.txt"

        assert label_path.exists(), f"Missing YOLO segmentation label file: {label_path}"

        with label_path.open("r", encoding="utf-8") as f:
            exported_lines = [line.strip() for line in f.readlines() if line.strip()]

        assert exported_lines == expected_lines, f"YOLO segmentation lines mismatch for sample: {sample_id}"

        expected_total_lines += len(expected_lines)

    assert expected_total_lines == result["label_lines_exported"]

    verification_output = {
        "test_name": "xbd_yolo_segmentation_integrity_test",
        "status": "passed",
        "samples_checked": max_samples,
        "label_lines_checked": expected_total_lines,
        "export_output": str(output_dir),
        "result": result,
    }

    verification_path = Path("outputs/verification/xbd_yolo_segmentation_integrity_test.json")
    verification_path.parent.mkdir(parents=True, exist_ok=True)

    with verification_path.open("w", encoding="utf-8") as f:
        json.dump(verification_output, f, indent=2)

    print("xBD YOLO segmentation integrity test passed.")
    print(json.dumps(verification_output, indent=2))


if __name__ == "__main__":
    main()
