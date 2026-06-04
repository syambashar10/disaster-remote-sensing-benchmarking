import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.loaders.xbd_loader import XBDLoader
from disasterbench.exporters.xbd_coco_exporter import (
    export_xbd_coco,
    damage_class_to_category_id,
    flatten_polygon,
)


def round_list(values, digits=6):
    return [round(float(value), digits) for value in values]


def main():
    max_samples = 50
    output_path = Path("outputs/coco/xbd_integrity_sample_coco.json")

    result = export_xbd_coco(
        output_path=output_path,
        max_samples=max_samples,
        split="train",
    )

    loader = XBDLoader(split="train")

    expected_images = []
    expected_annotations = []

    annotation_id = 1

    for image_id, index in enumerate(range(max_samples), start=1):
        sample = loader.load_sample(index)

        expected_images.append(
            {
                "id": image_id,
                "sample_id": sample["sample_id"],
                "width": sample["image"]["width"],
                "height": sample["image"]["height"],
                "disaster_phase": sample["disaster_phase"],
            }
        )

        for annotation in sample["annotations"]:
            segmentation = flatten_polygon(annotation["polygon"])

            if len(segmentation) < 6:
                continue

            expected_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": damage_class_to_category_id(annotation["damage_class"]),
                    "bbox": round_list(annotation["bbox"]),
                    "segmentation": round_list(segmentation),
                    "area": round(float(annotation["area_pixels"]), 6),
                    "sample_id": sample["sample_id"],
                    "annotation_id": annotation["annotation_id"],
                    "uid": annotation["uid"],
                    "damage_class": annotation["damage_class"],
                }
            )

            annotation_id += 1

    with output_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    assert result["images_exported"] == len(coco["images"])
    assert result["annotations_exported"] == len(coco["annotations"])

    assert len(coco["images"]) == len(expected_images)
    assert len(coco["annotations"]) == len(expected_annotations)

    for expected, exported in zip(expected_images, coco["images"]):
        assert exported["id"] == expected["id"]
        assert exported["sample_id"] == expected["sample_id"]
        assert exported["width"] == expected["width"]
        assert exported["height"] == expected["height"]
        assert exported["disaster_phase"] == expected["disaster_phase"]

    for expected, exported in zip(expected_annotations, coco["annotations"]):
        assert exported["id"] == expected["id"]
        assert exported["image_id"] == expected["image_id"]
        assert exported["category_id"] == expected["category_id"]
        assert round_list(exported["bbox"]) == expected["bbox"]
        assert round_list(exported["segmentation"][0]) == expected["segmentation"]
        assert round(float(exported["area"]), 6) == expected["area"]

        attrs = exported["attributes"]

        assert attrs["sample_id"] == expected["sample_id"]
        assert attrs["annotation_id"] == expected["annotation_id"]
        assert attrs["uid"] == expected["uid"]
        assert attrs["damage_class"] == expected["damage_class"]

    verification_output = {
        "test_name": "xbd_coco_integrity_test",
        "status": "passed",
        "samples_checked": max_samples,
        "images_checked": len(expected_images),
        "annotations_checked": len(expected_annotations),
        "export_output": str(output_path),
        "result": result,
    }

    verification_path = Path("outputs/verification/xbd_coco_integrity_test.json")
    verification_path.parent.mkdir(parents=True, exist_ok=True)

    with verification_path.open("w", encoding="utf-8") as f:
        json.dump(verification_output, f, indent=2)

    print("xBD COCO integrity test passed.")
    print(json.dumps(verification_output, indent=2))


if __name__ == "__main__":
    main()
