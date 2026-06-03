import json
from pathlib import Path
from typing import Dict, List, Optional

from disasterbench.converters.mask_to_polygon import mask_file_to_polygons
from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


def normalize_polygon(
    polygon: List[List[float]],
    width: int,
    height: int,
) -> List[float]:
    """
    Convert polygon points from pixel coordinates to YOLO normalized coordinates.

    YOLO segmentation format:
    class_id x1 y1 x2 y2 x3 y3 ...

    Coordinates must be between 0 and 1.
    """
    normalized = []

    # Remove duplicated closing point if present.
    if len(polygon) > 1 and polygon[0] == polygon[-1]:
        polygon = polygon[:-1]

    for x, y in polygon:
        x_norm = max(0.0, min(1.0, float(x) / float(width)))
        y_norm = max(0.0, min(1.0, float(y) / float(height)))

        normalized.extend([x_norm, y_norm])

    return normalized


def yolo_line_from_polygon(
    polygon_record: Dict,
    image_width: int,
    image_height: int,
    class_id: int = 0,
) -> Optional[str]:
    polygon = polygon_record["polygon"]

    if len(polygon) < 3:
        return None

    normalized_points = normalize_polygon(
        polygon=polygon,
        width=image_width,
        height=image_height,
    )

    # Need at least 3 points = 6 numbers.
    if len(normalized_points) < 6:
        return None

    values = [str(class_id)] + [f"{value:.6f}" for value in normalized_points]

    return " ".join(values)


def export_sturm_flood_yolo_segmentation(
    sensor: str,
    output_dir: str | Path,
    max_samples: Optional[int] = 10,
    target_values: Optional[List[int]] = None,
    min_area_pixels: float = 2.0,
) -> Dict:
    """
    Export STURM-Flood mask-derived polygons to YOLO segmentation label format.

    Notes:
    - STURM-Flood is originally a raster-mask semantic segmentation dataset.
    - YOLO segmentation labels are derived from raster mask polygons.
    - Interior holes are not fully represented in standard YOLO segmentation labels.
    - This exporter writes label .txt files and an image manifest, but does not copy/convert images.
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

        polygons = mask_file_to_polygons(
            mask_path=mask_path,
            target_values=target_values,
            min_area_pixels=min_area_pixels,
            coordinate_space="pixel",
            preserve_holes=True,
        )

        label_lines = []

        for polygon_record in polygons:
            line = yolo_line_from_polygon(
                polygon_record=polygon_record,
                image_width=image_width,
                image_height=image_height,
                class_id=0,
            )

            if line is not None:
                label_lines.append(line)

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
                "note": "YOLO segmentation labels are derived from raster mask polygons.",
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
        f.write("# Note: labels are derived from STURM-Flood raster masks.\n")
        f.write("# Image conversion/copying is handled separately because STURM images are GeoTIFF rasters.\n")

    return {
        "output_dir": str(output_dir),
        "labels_dir": str(labels_dir),
        "manifest_path": str(manifest_path),
        "yaml_path": str(yaml_path),
        "samples_exported": sample_limit,
        "label_files_exported": sample_limit,
        "label_lines_exported": total_label_lines,
    }
