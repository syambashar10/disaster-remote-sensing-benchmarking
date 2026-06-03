from pathlib import Path
from typing import Iterable, List

from disasterbench.converters.mask_to_polygon import mask_file_to_polygons


DEFAULT_TARGET_VALUES = [1, 2, 3, 4, 5]


def polygon_to_coco_bbox(polygon: List[List[float]]) -> List[float]:
    """
    Convert polygon points into COCO-style bbox:
    [x_min, y_min, width, height]
    """
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min),
        float(y_max - y_min),
    ]


def mask_file_to_bboxes(
    mask_path: str | Path,
    target_values: Iterable[int] = DEFAULT_TARGET_VALUES,
    min_area_pixels: float = 1.0,
) -> List[dict]:
    """
    Convert selected mask regions into COCO-style bounding boxes.

    Note:
    For flood masks, this is a derived/lossy representation because
    bounding boxes do not preserve the exact flood shape.
    """
    polygons = mask_file_to_polygons(
        mask_path=mask_path,
        target_values=target_values,
        min_area_pixels=min_area_pixels,
        coordinate_space="pixel",
    )

    bboxes = []

    for polygon_record in polygons:
        polygon = polygon_record["polygon"]
        bbox = polygon_to_coco_bbox(polygon)

        bboxes.append(
            {
                "bbox": bbox,
                "bbox_format": "coco_xywh",
                "area_pixels": polygon_record["area_pixels"],
                "source_mask_values": polygon_record["source_mask_values"],
                "lossy": True,
                "lossy_reason": "Bounding boxes approximate flood regions and do not preserve full mask shape.",
            }
        )

    return bboxes
