import json
from pathlib import Path
from typing import Dict, List, Optional

from disasterbench.converters.mask_to_polygon import mask_file_to_polygons
from disasterbench.converters.mask_to_bbox import polygon_to_coco_bbox
from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


def flatten_polygon(polygon: List[List[float]]) -> List[float]:
    flattened = []

    for x, y in polygon:
        flattened.extend([float(x), float(y)])

    return flattened


def build_coco_annotation(
    annotation_id: int,
    image_id: int,
    polygon_record: Dict,
    category_id: int = 1,
) -> Dict:
    polygon = polygon_record["polygon"]

    segmentation = [flatten_polygon(polygon)]
    bbox = polygon_to_coco_bbox(polygon)

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "bbox": bbox,
        "area": float(polygon_record["area_pixels"]),
        "iscrowd": 0,
        "attributes": {
            "source_annotation_type": "raster_mask",
            "derived_geometry": "polygon",
            "source_mask_values": polygon_record["source_mask_values"],
            "coordinate_space": polygon_record["coordinate_space"],
            "hole_count": len(polygon_record.get("holes", [])),
            "note": "COCO polygon segmentation is derived from a raster mask. Interior holes are counted but not fully represented in standard polygon segmentation.",
        },
    }


def export_sturm_flood_coco_segmentation(
    sensor: str,
    output_path: str | Path,
    max_samples: Optional[int] = 10,
    target_values: Optional[List[int]] = None,
    min_area_pixels: float = 2.0,
) -> Dict:
    """
    Export STURM-Flood mask-derived polygons to COCO segmentation format.

    Notes:
    - STURM-Flood is originally a raster-mask semantic segmentation dataset.
    - This COCO export is a derived polygon segmentation representation.
    - Bounding boxes are derived from polygons.
    """
    if target_values is None:
        target_values = [1, 2, 3, 4, 5]

    loader = STURMFloodLoader(sensor=sensor)

    total_samples = len(loader.samples)
    sample_limit = total_samples if max_samples is None else min(max_samples, total_samples)

    images = []
    annotations = []

    annotation_id = 1

    for index in range(sample_limit):
        sample = loader.load_sample(index)

        image_id = index + 1
        image_info = sample["image"]
        mask_path = sample["annotations"][0]["mask_path"]

        images.append(
            {
                "id": image_id,
                "file_name": image_info["path"],
                "width": image_info["width"],
                "height": image_info["height"],
                "dataset_id": sample["dataset_id"],
                "sensor": sample["sensor"],
                "sample_id": sample["sample_id"],
                "metadata": sample["metadata"],
            }
        )

        polygons = mask_file_to_polygons(
            mask_path=mask_path,
            target_values=target_values,
            min_area_pixels=min_area_pixels,
            coordinate_space="pixel",
            preserve_holes=True,
        )

        for polygon_record in polygons:
            if len(polygon_record["polygon"]) < 4:
                continue

            coco_annotation = build_coco_annotation(
                annotation_id=annotation_id,
                image_id=image_id,
                polygon_record=polygon_record,
                category_id=1,
            )

            annotations.append(coco_annotation)
            annotation_id += 1

    coco = {
        "info": {
            "description": "STURM-Flood derived COCO segmentation export",
            "dataset_id": "sturm_flood",
            "sensor": sensor,
            "source_annotation_type": "raster_mask",
            "export_format": "coco_segmentation",
            "note": "This export derives polygon segmentations and bounding boxes from raster flood masks.",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "water_or_flood_region",
                "supercategory": "water",
            }
        ],
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
    }
