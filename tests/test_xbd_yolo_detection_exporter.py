import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.exporters.xbd_yolo_detection_exporter import export_xbd_yolo_detection
from disasterbench.exporters.xbd_coco_exporter import DAMAGE_CLASSES


def validate_yolo_detection_line(line: str):
    parts = line.strip().split()

    assert len(parts) == 5

    class_id = int(parts[0])
    assert 0 <= class_id < len(DAMAGE_CLASSES)

    coords = [float(value) for value in parts[1:]]

    for value in coords:
        assert 0.0 <= value <= 1.0


def main():
    output_dir = Path("outputs/yolo_detection/xbd_sample")

    result = export_xbd_yolo_detection(
        output_dir=output_dir,
        max_samples=20,
        split="train",
    )

    labels_dir = output_dir / "labels"
    manifest_path = output_dir / "image_label_manifest.json"
    yaml_path = output_dir / "data.yaml"

    assert labels_dir.exists()
    assert manifest_path.exists()
    assert yaml_path.exists()

    label_files = sorted(labels_dir.glob("*.txt"))
    assert len(label_files) == 20

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == 20

    total_lines = 0

    for label_file in label_files:
        with label_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        total_lines += len(lines)

        for line in lines:
            validate_yolo_detection_line(line)

    assert total_lines == result["label_lines_exported"]

    verification_output = {
        "test_name": "xbd_yolo_detection_exporter_test",
        "status": "passed",
        "result": result,
        "total_label_lines_checked": total_lines,
    }

    verification_path = Path("outputs/verification/xbd_yolo_detection_exporter_test.json")
    verification_path.parent.mkdir(parents=True, exist_ok=True)

    with verification_path.open("w", encoding="utf-8") as f:
        json.dump(verification_output, f, indent=2)

    print("xBD YOLO detection exporter test passed.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
