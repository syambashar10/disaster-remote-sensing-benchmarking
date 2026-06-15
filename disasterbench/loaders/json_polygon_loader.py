from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from shapely import wkt
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

from disasterbench.loaders.base_loader import BaseDatasetLoader
from disasterbench.schemas.common_sample import build_image_info, validate_common_sample


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _geometry_to_polygons(geometry: Any) -> List[Polygon]:
    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        polygons: List[Polygon] = []
        for geom in geometry.geoms:
            polygons.extend(_geometry_to_polygons(geom))
        return polygons

    return []


def _polygon_to_points(polygon: Polygon) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in polygon.exterior.coords]


def _polygon_to_holes(polygon: Polygon) -> List[List[List[float]]]:
    return [
        [[float(x), float(y)] for x, y in interior.coords]
        for interior in polygon.interiors
    ]


def _bbox_from_points(points: List[List[float]]) -> List[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

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


def _collect_files(folder: Path, extensions: List[str]) -> List[Path]:
    files: List[Path] = []

    for ext in extensions:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(set(files))


class JsonPolygonLoader(BaseDatasetLoader):
    """
    Generic loader for image + JSON polygon annotation datasets.

    This supports xBD-style datasets where:
    - image files are stored in one folder
    - JSON label files are stored in another folder
    - annotations contain WKT polygons, usually under features.xy
    """

    def __init__(
        self,
        config_path: str | Path,
        max_samples: int | None = None,
    ):
        self.config_path = Path(config_path)
        self.config = _load_json(self.config_path)

        self.dataset_id = self.config["dataset_id"]
        self.dataset_root = Path(self.config["dataset_path"])

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        split_structure = self.config.get("split_structure", {})
        schema = self.config.get("json_polygon_schema", {})

        self.image_folder = self.dataset_root / schema.get(
            "image_folder",
            split_structure.get("train_images", "images"),
        )
        self.label_folder = self.dataset_root / schema.get(
            "label_folder",
            split_structure.get("train_labels", "labels"),
        )

        self.image_extensions = schema.get("image_extensions", [".png", ".jpg", ".jpeg"])
        self.label_extensions = schema.get("label_extensions", [".json"])

        self.feature_container = schema.get("feature_container", "features.xy")
        self.wkt_key = schema.get("wkt_key", "wkt")
        self.properties_key = schema.get("properties_key", "properties")
        self.class_key = schema.get("class_key", "feature_type")
        self.damage_key = schema.get("damage_key", "subtype")
        self.uid_key = schema.get("uid_key", "uid")

        self.image_files = _collect_files(self.image_folder, self.image_extensions)
        self.label_files = _collect_files(self.label_folder, self.label_extensions)

        self.image_map = {path.stem: path for path in self.image_files}
        self.label_map = {path.stem: path for path in self.label_files}

        self.sample_ids = sorted(set(self.image_map) & set(self.label_map))

        if max_samples is not None:
            self.sample_ids = self.sample_ids[:max_samples]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _get_nested_value(self, data: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
        current: Any = data

        for part in dotted_key.split("."):
            if not isinstance(current, dict):
                return default
            current = current.get(part)

            if current is None:
                return default

        return current

    def _read_image_info(self, path: Path) -> Dict[str, Any]:
        with Image.open(path) as image:
            return {
                "width": int(image.width),
                "height": int(image.height),
                "mode": image.mode,
            }

    def _parse_wkt_polygons(self, source_wkt: str) -> List[Polygon]:
        geometry = wkt.loads(source_wkt)

        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        return _geometry_to_polygons(geometry)

    def _parse_annotations(self, label_path: Path) -> Dict[str, Any]:
        data = _load_json(label_path)

        metadata = data.get("metadata", {})
        objects = self._get_nested_value(data, self.feature_container, default=[]) or []

        annotations = []
        parse_errors = []

        for object_index, obj in enumerate(objects):
            source_wkt = obj.get(self.wkt_key)

            if not source_wkt:
                parse_errors.append(
                    {
                        "object_index": object_index,
                        "reason": "missing_wkt",
                    }
                )
                continue

            properties = obj.get(self.properties_key, {}) or {}

            try:
                polygons = self._parse_wkt_polygons(source_wkt)
            except Exception as exc:
                parse_errors.append(
                    {
                        "object_index": object_index,
                        "reason": "wkt_parse_error",
                        "details": str(exc),
                    }
                )
                continue

            for part_index, polygon in enumerate(polygons):
                if polygon.is_empty:
                    continue

                points = _polygon_to_points(polygon)

                if len(points) < 4:
                    continue

                annotation_id = properties.get(self.uid_key)
                if annotation_id is None:
                    annotation_id = f"{object_index}_{part_index}"

                annotations.append(
                    {
                        "type": "polygon",
                        "geometry_type": "polygon",
                        "annotation_id": str(annotation_id),
                        "source_object_index": object_index,
                        "source_part_index": part_index,
                        "class_name": properties.get(self.class_key, self.config.get("feature_type", "object")),
                        "damage_class": properties.get(self.damage_key, "unknown"),
                        "polygon": points,
                        "holes": _polygon_to_holes(polygon),
                        "bbox": _bbox_from_points(points),
                        "area": float(polygon.area),
                        "properties": properties,
                    }
                )

        return {
            "metadata": metadata,
            "annotations": annotations,
            "parse_errors": parse_errors,
        }

    def list_samples(self) -> List[Dict[str, str]]:
        return [
            {
                "sample_id": sample_id,
                "image_path": str(self.image_map[sample_id]),
                "label_path": str(self.label_map[sample_id]),
            }
            for sample_id in self.sample_ids
        ]

    def load_sample(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.sample_ids):
            raise IndexError(
                f"Sample index {index} out of range. Dataset has {len(self.sample_ids)} samples."
            )

        sample_id = self.sample_ids[index]
        image_path = self.image_map[sample_id]
        label_path = self.label_map[sample_id]

        image_meta = self._read_image_info(image_path)
        parsed = self._parse_annotations(label_path)

        image_info = build_image_info(
            path=str(image_path),
            width=image_meta["width"],
            height=image_meta["height"],
            band_count=3 if image_meta["mode"] in {"RGB", "RGBA"} else 1,
            dtypes=[image_meta["mode"]],
            crs=None,
            bounds=None,
        )

        image_info["image_mode"] = image_meta["mode"]

        common_sample = {
            "dataset_id": self.dataset_id,
            "sample_id": sample_id,
            "split": self.config.get("split", "train"),
            "task_type": "instance_segmentation",
            "image": image_info,
            "annotations": parsed["annotations"],
            "metadata": {
                "label_path": str(label_path),
                "raw_metadata": parsed["metadata"],
                "parse_errors": parsed["parse_errors"],
                "source_format": "image_json_polygon_pair",
            },
        }

        validate_common_sample(common_sample)

        return common_sample

    def validate_pairing(self) -> Dict[str, Any]:
        image_ids = set(self.image_map)
        label_ids = set(self.label_map)

        missing_labels = sorted(image_ids - label_ids)
        extra_labels = sorted(label_ids - image_ids)

        return {
            "dataset_id": self.dataset_id,
            "image_count": len(self.image_files),
            "label_count": len(self.label_files),
            "matched_count": len(set(image_ids) & set(label_ids)),
            "missing_label_count": len(missing_labels),
            "extra_label_count": len(extra_labels),
            "missing_label_examples": missing_labels[:20],
            "extra_label_examples": extra_labels[:20],
            "valid": len(missing_labels) == 0 and len(extra_labels) == 0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.config.get("dataset_name"),
            "loader_name": "JsonPolygonLoader",
            "data_family": self.config.get("data_family", "json_polygon_annotations"),
            "raw_annotation_type": self.config.get("raw_annotation_type"),
            "task_type": self.config.get("task_type", []),
            "total_samples": len(self.sample_ids),
            "pairing": self.validate_pairing(),
        }
