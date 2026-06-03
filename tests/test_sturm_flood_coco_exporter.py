import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.exporters.coco_exporter import export_sturm_flood_coco_segmentation


def test_coco(sensor: str, max_samples: int):
    output_path = Path(f"outputs/coco/sturm_flood_{sensor}_sample_coco.json")

    result = export_sturm_flood_coco_segmentation(
        sensor=sensor,
        output_path=output_path,
        max_samples=max_samples,
        target_values=[1, 2, 3, 4, 5],
        min_area_pixels=2.0,
    )

    if not output_path.exists():
        raise AssertionError(f"COCO output was not created: {output_path}")

    with output_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    assert "images" in coco
    assert "annotations" in coco
    assert "categories" in coco
    assert len(coco["images"]) == max_samples
    assert len(coco["categories"]) == 1
    assert coco["categories"][0]["name"] == "water_or_flood_region"

    for image in coco["images"]:
        assert "id" in image
        assert "file_name" in image
        assert image["width"] == 128
        assert image["height"] == 128
        assert image["sensor"] == sensor

    for annotation in coco["annotations"]:
        assert "id" in annotation
        assert "image_id" in annotation
        assert "category_id" in annotation
        assert "segmentation" in annotation
        assert "bbox" in annotation
        assert "area" in annotation
        assert annotation["category_id"] == 1

        bbox = annotation["bbox"]
        assert len(bbox) == 4

        x, y, w, h = bbox
        assert x >= 0
        assert y >= 0
        assert w >= 0
        assert h >= 0
        assert x <= 128
        assert y <= 128
        assert x + w <= 129
        assert y + h <= 129

        segmentation = annotation["segmentation"]
        assert isinstance(segmentation, list)
        assert len(segmentation) >= 1
        assert len(segmentation[0]) >= 6

    return result


def main():
    output = {
        "sentinel1": test_coco("sentinel1", max_samples=20),
        "sentinel2": test_coco("sentinel2", max_samples=20),
    }

    output_path = Path("outputs/verification/sturm_flood_coco_exporter_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("STURM-Flood COCO exporter test passed.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
