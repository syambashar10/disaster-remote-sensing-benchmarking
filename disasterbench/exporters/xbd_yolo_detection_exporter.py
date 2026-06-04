import json
from pathlib import Path
from typing import Dict, List, Optional

from disasterbench.loaders.xbd_loader import XBDLoader
from disasterbench.exporters.xbd_coco_exporter import DAMAGE_CLASSES


def damage_class_to_yolo_id(damage_class: str) -> int:
    if damage_class not in DAMAGE_CLASSES:
        damage_class = "un-classified"

    return DAMAGE_CLASSES.index(damage_class)


def coco_bbox_to_yolo_bbox(
    bbox: List[float],
    image_width: int,
    image_height: int,
) -> List[float]:
    x_min, y_min, width, height = bbox

    x_center = x_min + width / 2.0
    y_center = y_min + height / 2.0

    return [
        max(0.0, min(1.0, x_center / image_width)),
        max(0.0, min(1.0, y_center / image_height)),
        max(0.0, min(1.0, width / image_width)),
        max(0.0, min(1.0, height / image_height)),
    ]


def yolo_detection_line(annotation: Dict, image_width: int, image_height: int) -> str:
    class_id = damage_class_to_yolo_id(annotation["damage_class"])

    yolo_bbox = coco_bbox_to_yolo_bbox(
        bbox=annotation["bbox"],
        image_width=image_width,
        image_height=image_height,
    )

    values = [str(class_id)] + [f"{value:.6f}" for value in yolo_bbox]
    return " ".join(values)


def export_xbd_yolo_detection(
    output_dir: str | Path,
    max_samples: Optional[int] = 10,
    split: str = "train",
) -> Dict:
    """
    Export xBD building damage annotations to YOLO detection format.

    xBD polygon annotations are converted to YOLO bounding boxes.
    This is suitable for detection-style experiments.
    """
    loader = XBDLoader(split=split)

    total_samples = len(loader)
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
        sample_id = sample["sample_id"]

        label_lines = [
            yolo_detection_line(
                annotation=annotation,
                image_width=image_width,
                image_height=image_height,
            )
            for annotation in sample["annotations"]
        ]

        label_path = labels_dir / f"{sample_id}.txt"

        with label_path.open("w", encoding="utf-8") as f:
            for line in label_lines:
                f.write(line + "\n")

        total_label_lines += len(label_lines)

        manifest.append(
            {
                "dataset_id": "xbd",
                "sample_id": sample_id,
                "image_path": sample["image"]["path"],
                "label_path": str(label_path),
                "source_label_path": sample["label_path"],
                "disaster_phase": sample["disaster_phase"],
                "annotation_count": sample["annotation_count"],
            }
        )

    manifest_path = output_dir / "image_label_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    yaml_path = output_dir / "data.yaml"

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("names:\n")
        for index, damage_class in enumerate(DAMAGE_CLASSES):
            f.write(f"  {index}: {damage_class}\n")
        f.write(f"nc: {len(DAMAGE_CLASSES)}\n")
        f.write("# YOLO detection labels exported from xBD polygon-derived bounding boxes.\n")

    return {
        "output_dir": str(output_dir),
        "labels_dir": str(labels_dir),
        "manifest_path": str(manifest_path),
        "yaml_path": str(yaml_path),
        "samples_exported": sample_limit,
        "label_files_exported": sample_limit,
        "label_lines_exported": total_label_lines,
        "classes_exported": len(DAMAGE_CLASSES),
    }
