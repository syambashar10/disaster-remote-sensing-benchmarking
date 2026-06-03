import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.exporters.yolo_segmentation_exporter import export_sturm_flood_yolo_segmentation


def validate_yolo_line(line: str):
    parts = line.strip().split()

    if len(parts) < 7:
        raise AssertionError("YOLO segmentation line must contain class_id plus at least 3 points.")

    class_id = int(parts[0])
    assert class_id == 0

    coords = [float(value) for value in parts[1:]]

    if len(coords) % 2 != 0:
        raise AssertionError("YOLO segmentation coordinates must be x/y pairs.")

    for value in coords:
        assert 0.0 <= value <= 1.0


def test_yolo(sensor: str, max_samples: int):
    output_dir = Path(f"outputs/yolo_segmentation/sturm_flood_{sensor}_sample")

    result = export_sturm_flood_yolo_segmentation(
        sensor=sensor,
        output_dir=output_dir,
        max_samples=max_samples,
        target_values=[1, 2, 3, 4, 5],
        min_area_pixels=2.0,
    )

    labels_dir = output_dir / "labels"
    manifest_path = output_dir / "image_label_manifest.json"
    yaml_path = output_dir / "data.yaml"

    assert labels_dir.exists()
    assert manifest_path.exists()
    assert yaml_path.exists()

    label_files = sorted(labels_dir.glob("*.txt"))
    assert len(label_files) == max_samples

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == max_samples

    total_lines = 0

    for label_file in label_files:
        with label_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        total_lines += len(lines)

        for line in lines:
            validate_yolo_line(line)

    assert total_lines == result["label_lines_exported"]

    return result


def main():
    output = {
        "sentinel1": test_yolo("sentinel1", max_samples=20),
        "sentinel2": test_yolo("sentinel2", max_samples=20),
    }

    output_path = Path("outputs/verification/sturm_flood_yolo_segmentation_exporter_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("STURM-Flood YOLO segmentation exporter test passed.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
