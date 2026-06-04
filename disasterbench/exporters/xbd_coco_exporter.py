import json
from pathlib import Path
from typing import Dict, List, Optional

from disasterbench.loaders.xbd_loader import XBDLoader


DAMAGE_CLASSES = [
    "no_damage_label",
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
    "un-classified",
]


def build_categories() -> List[Dict]:
    return [
        {
            "id": index + 1,
            "name": damage_class,
            "supercategory": "building_damage",
        }
        for index, damage_class in enumerate(DAMAGE_CLASSES)
    ]


def damage_class_to_category_id(damage_class: str) -> int:
    if damage_class not in DAMAGE_CLASSES:
        damage_class = "un-classified"

    return DAMAGE_CLASSES.index(damage_class) + 1


def flatten_polygon(points: List[List[float]]) -> List[float]:
    """
    Convert [[x, y], ...] polygon points into COCO segmentation format:
    [x1, y1, x2, y2, ...]
    """
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]

    flattened = []

    for x, y in points:
        flattened.extend([float(x), float(y)])

    return flattened


def export_xbd_coco(
    output_path: str | Path,
    max_samples: Optional[int] = 10,
    split: str = "train",
) -> Dict:
    """
    Export xBD annotations to COCO-style JSON.

    Notes:
    - xBD polygons come from features.xy pixel coordinates.
    - COCO bbox uses [x_min, y_min, width, height].
    - COCO segmentation uses polygon coordinates.
    - Categories are damage classes.
    """
    loader = XBDLoader(split=split)

    total_samples = len(loader)
    sample_limit = total_samples if max_samples is None else min(max_samples, total_samples)

    images = []
    annotations = []
    annotation_id = 1

    for image_id, index in enumerate(range(sample_limit), start=1):
        sample = loader.load_sample(index)

        images.append(
            {
                "id": image_id,
                "file_name": sample["image"]["path"],
                "width": sample["image"]["width"],
                "height": sample["image"]["height"],
                "sample_id": sample["sample_id"],
                "disaster_phase": sample["disaster_phase"],
            }
        )

        for annotation in sample["annotations"]:
            segmentation = flatten_polygon(annotation["polygon"])

            if len(segmentation) < 6:
                continue

            coco_annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": damage_class_to_category_id(annotation["damage_class"]),
                "bbox": annotation["bbox"],
                "segmentation": [segmentation],
                "area": annotation["area_pixels"],
                "iscrowd": 0,
                "attributes": {
                    "dataset_id": "xbd",
                    "sample_id": sample["sample_id"],
                    "annotation_id": annotation["annotation_id"],
                    "uid": annotation.get("uid"),
                    "category": annotation["category"],
                    "feature_type": annotation["feature_type"],
                    "damage_class": annotation["damage_class"],
                    "disaster_phase": sample["disaster_phase"],
                    "disaster": sample["metadata"].get("disaster"),
                    "disaster_type": sample["metadata"].get("disaster_type"),
                    "capture_date": sample["metadata"].get("capture_date"),
                    "source_annotation_type": "polygon",
                },
            }

            annotations.append(coco_annotation)
            annotation_id += 1

    coco = {
        "info": {
            "dataset_id": "xbd",
            "dataset_name": "xBD / xView2",
            "split": split,
            "source_annotation_type": "polygon",
            "export_format": "coco_instance_segmentation",
            "note": "COCO annotations are exported from xBD features.xy pixel polygons. Categories represent damage classes.",
        },
        "images": images,
        "annotations": annotations,
        "categories": build_categories(),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    return {
        "output_path": str(output_path),
        "samples_exported": sample_limit,
        "images_exported": len(images),
        "annotations_exported": len(annotations),
        "categories_exported": len(coco["categories"]),
    }
