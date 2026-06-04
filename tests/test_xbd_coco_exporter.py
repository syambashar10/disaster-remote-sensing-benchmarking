import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.exporters.xbd_coco_exporter import export_xbd_coco


def main():
    output_path = Path("outputs/coco/xbd_sample_coco.json")

    result = export_xbd_coco(
        output_path=output_path,
        max_samples=20,
        split="train",
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    assert "info" in coco
    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco

    assert coco["info"]["dataset_id"] == "xbd"
    assert len(coco["images"]) == 20
    assert len(coco["categories"]) >= 1
    assert len(coco["annotations"]) > 0

    category_ids = {category["id"] for category in coco["categories"]}
    image_ids = {image["id"] for image in coco["images"]}

    first_annotation = coco["annotations"][0]

    assert first_annotation["image_id"] in image_ids
    assert first_annotation["category_id"] in category_ids
    assert len(first_annotation["bbox"]) == 4
    assert len(first_annotation["segmentation"]) == 1
    assert len(first_annotation["segmentation"][0]) >= 6
    assert first_annotation["area"] > 0
    assert first_annotation["iscrowd"] == 0

    x, y, w, h = first_annotation["bbox"]

    assert x >= 0
    assert y >= 0
    assert w > 0
    assert h > 0
    assert x <= 1024
    assert y <= 1024
    assert x + w <= 1025
    assert y + h <= 1025

    assert "attributes" in first_annotation
    assert first_annotation["attributes"]["dataset_id"] == "xbd"
    assert "damage_class" in first_annotation["attributes"]

    verification_output = {
        "test_name": "xbd_coco_exporter_test",
        "status": "passed",
        "result": result,
        "first_annotation": first_annotation,
    }

    verification_path = Path("outputs/verification/xbd_coco_exporter_test.json")
    verification_path.parent.mkdir(parents=True, exist_ok=True)

    with verification_path.open("w", encoding="utf-8") as f:
        json.dump(verification_output, f, indent=2)

    print("xBD COCO exporter test passed.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
