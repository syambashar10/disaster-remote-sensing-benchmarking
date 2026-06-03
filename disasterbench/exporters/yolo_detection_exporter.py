import json
from pathlib import Path
from typing import Dict, List, Optional

from disasterbench.converters.mask_to_bbox import mask_file_to_bboxes
from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


def coco_bbox_to_yolo_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int,
) -> List[float]:
    """
    Convert COCO bbox [x_min, y_min, width, height]
    to YOLO bbox [x_center, y_center, width, height], normalized 0-1.
    """
    x_min, y_min, width, height = bbox

    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    return [
        max(0.0, min(1.0, x_center / image_width)),
        max(0.0, min(1.0, y_center / image_height)),
        max(0.0, min(1.0, width / image_width)),
        max(0.0, min(1.0, height / image_height)),
    ]


def yolo_detection_line_from_bbox(
    bbox_record: Dict,
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> str:
    yolo_bbox = coco_bbox_to_yolo_bbox(
        bbox=bbox_record["bbox"],
        image_width=image_width,
        image_height=image_height,
    )

    values = [str(class_id)] + [f"{value:.6f}" for value in yolo_bbox]

    return " ".join(values)


def export_sturm_flood_yolo_detection(
    sensor: str,
    output_dir: str | Path,
    max_samples: Optional[int] = 10,
    target_values: Optional[List[int]] = None,
    min_area_pixels: float = 2.0,
) -> Dict:
    """
    Export STURM-Flood mask-derived bounding boxes to YOLO detection format.

    Important:
    This is a lossy export. STURM-Flood is originally a raster-mask segmentation
    dataset. Bounding boxes approximate flood/water regions and do not preserve
    the full shape of the mask.
    """
    if target_values is None:
        target_values = [1, 2, 3, 4, 5]

    loader = STURMFloodLoader(sensor=sensor)

    total_samples = len(loader.samples)
    sample_limit = total_samples if max_samples is None else min(max_samples, total_samples)

    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    total_label_lines = 0

    for index in range(sample_limit):
        sample = loader.load_sample(index)

        image_width = sample["image"]["width"]
        image_height = sample["image"]["height"]
        image_path = sample["image"]["path"]
        mask_path = sample["annotations"][0]["mask_path"]
        sample_id = sample["sample_id"]

        bboxes = mask_file_to_bboxes(
            mask_path=mask_path,
            target_values=target_values,
            min_area_pixels=min_area_pixels,
        )

        label_lines = [
            yolo_detection_line_from_bbox(
                bbox_record=bbox_record,
                image_width=image_width,
                image_height=image_height,
                class_id=0,
            )
            for bbox_record in bboxes
        ]

        label_path = labels_dir / f"{sample_id}.txt"

        with label_path.open("w", encoding="utf-8") as f:
            for line in label_lines:
                f.write(line + "\n")

        total_label_lines += len(label_lines)

        manifest.append(
            {
                "dataset_id": "sturm_flood",
                "sensor": sensor,
                "sample_id": sample_id,
                "image_path": image_path,
                "mask_path": mask_path,
                "label_path": str(label_path),
                "label_count": len(label_lines),
                "lossy": True,
                "lossy_reason": "Bounding boxes approximate raster flood regions and do not preserve exact mask shape.",
            }
        )

    manifest_path = output_dir / "image_label_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    yaml_path = output_dir / "data.yaml"

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("names:\n")
        f.write("  0: water_or_flood_region\n")
        f.write("nc: 1\n")
        f.write("# Lossy export: bounding boxes are derived from STURM-Flood raster masks.\n")
        f.write("# Image conversion/copying is handled separately because STURM images are GeoTIFF rasters.\n")

    return {
        "output_dir": str(output_dir),
        "labels_dir": str(labels_dir),
        "manifest_path": str(manifest_path),
        "yaml_path": str(yaml_path),
        "samples_exported": sample_limit,
        "label_files_exported": sample_limit,
        "label_lines_exported": total_label_lines,
        "lossy": True,
    }
