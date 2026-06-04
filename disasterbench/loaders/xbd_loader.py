import json
from pathlib import Path
from typing import Dict, List

from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely import wkt


def normalize_stem(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_target"):
        stem = stem[:-7]
    return stem


def geometry_to_polygons(geometry):
    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        polygons = []
        for geom in geometry.geoms:
            polygons.extend(geometry_to_polygons(geom))
        return polygons

    return []


def polygon_to_points(polygon: Polygon) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in polygon.exterior.coords]


def polygon_to_holes(polygon: Polygon) -> List[List[List[float]]]:
    return [
        [[float(x), float(y)] for x, y in interior.coords]
        for interior in polygon.interiors
    ]


def polygon_to_coco_bbox(points: List[List[float]]) -> List[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

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


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_image_info(path: Path) -> Dict:
    with Image.open(path) as img:
        return {
            "path": str(path),
            "width": int(img.width),
            "height": int(img.height),
            "mode": img.mode,
        }


class XBDLoader:
    """
    Dataset-specific loader for xBD / xView2.

    xBD annotations are JSON files containing WKT building polygons.
    The loader maps xBD into the DisasterBench common internal schema.

    It keeps:
    - pixel polygon from features.xy for COCO / YOLO / bbox exports
    - geo polygon from features.lng_lat for GeoJSON exports
    """

    def __init__(
        self,
        dataset_root: str | Path = "datasets/xbd_raw",
        split: str = "train",
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split

        self.split_root = self.dataset_root / split
        self.images_dir = self.split_root / "images"
        self.labels_dir = self.split_root / "labels"
        self.targets_dir = self.split_root / "targets"

        self.image_files = sorted(self.images_dir.glob("*.png"))
        self.label_files = sorted(self.labels_dir.glob("*.json"))
        self.target_files = sorted(self.targets_dir.glob("*.png"))

        self.image_map = {normalize_stem(path): path for path in self.image_files}
        self.label_map = {normalize_stem(path): path for path in self.label_files}
        self.target_map = {normalize_stem(path): path for path in self.target_files}

        self.sample_ids = sorted(set(self.image_map) & set(self.label_map))

    def __len__(self):
        return len(self.sample_ids)

    def list_samples(self) -> List[str]:
        return self.sample_ids

    def validate_pairing(self) -> Dict:
        image_ids = set(self.image_map)
        label_ids = set(self.label_map)
        target_ids = set(self.target_map)

        return {
            "image_files": len(self.image_files),
            "label_files": len(self.label_files),
            "target_files": len(self.target_files),
            "images_without_labels": len(image_ids - label_ids),
            "labels_without_images": len(label_ids - image_ids),
            "images_without_targets": len(image_ids - target_ids),
            "targets_without_images": len(target_ids - image_ids),
            "indexed_samples": len(self.sample_ids),
        }

    def get_statistics(self) -> Dict:
        return {
            "dataset_id": "xbd",
            "dataset_name": "xBD / xView2",
            "split": self.split,
            "raw_annotation_type": "polygon",
            "task_type": [
                "building_damage_assessment",
                "object_detection",
                "instance_segmentation",
                "change_analysis",
            ],
            "pairing": self.validate_pairing(),
        }

    def _get_paths(self, sample_id: str) -> Dict:
        return {
            "image_path": self.image_map[sample_id],
            "label_path": self.label_map[sample_id],
            "target_path": self.target_map.get(sample_id),
        }

    def _parse_wkt_polygons(self, source_wkt: str):
        geometry = wkt.loads(source_wkt)

        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        return geometry_to_polygons(geometry)

    def _parse_annotations(self, label_path: Path) -> Dict:
        data = read_json(label_path)

        metadata = data.get("metadata", {})
        features = data.get("features", {})

        xy_objects = features.get("xy", []) or []
        lng_lat_objects = features.get("lng_lat", []) or []

        annotations = []
        parse_errors = []

        if len(xy_objects) != len(lng_lat_objects):
            parse_errors.append(
                {
                    "reason": "xy_lng_lat_count_mismatch",
                    "xy_count": len(xy_objects),
                    "lng_lat_count": len(lng_lat_objects),
                }
            )

        paired_count = min(len(xy_objects), len(lng_lat_objects))

        for object_index in range(paired_count):
            xy_obj = xy_objects[object_index]
            geo_obj = lng_lat_objects[object_index]

            properties = xy_obj.get("properties", {})
            source_xy_wkt = xy_obj.get("wkt")
            source_geo_wkt = geo_obj.get("wkt")

            if not source_xy_wkt:
                parse_errors.append(
                    {
                        "object_index": object_index,
                        "reason": "missing_xy_wkt",
                    }
                )
                continue

            if not source_geo_wkt:
                parse_errors.append(
                    {
                        "object_index": object_index,
                        "reason": "missing_lng_lat_wkt",
                    }
                )
                continue

            try:
                xy_polygons = self._parse_wkt_polygons(source_xy_wkt)
            except Exception as exc:
                parse_errors.append(
                    {
                        "object_index": object_index,
                        "reason": "xy_wkt_parse_error",
                        "details": str(exc),
                    }
                )
                continue

            try:
                geo_polygons = self._parse_wkt_polygons(source_geo_wkt)
            except Exception as exc:
                parse_errors.append(
                    {
                        "object_index": object_index,
                        "reason": "lng_lat_wkt_parse_error",
                        "details": str(exc),
                    }
                )
                continue

            part_count = min(len(xy_polygons), len(geo_polygons))

            for part_index in range(part_count):
                xy_polygon = xy_polygons[part_index]
                geo_polygon = geo_polygons[part_index]

                if xy_polygon.is_empty or geo_polygon.is_empty:
                    continue

                xy_points = polygon_to_points(xy_polygon)
                geo_points = polygon_to_points(geo_polygon)

                if len(xy_points) < 4 or len(geo_points) < 4:
                    continue

                damage_class = properties.get("subtype", "no_damage_label")
                uid = properties.get("uid")

                annotations.append(
                    {
                        "annotation_id": f"{uid or object_index}_{part_index}",
                        "source_object_index": object_index,
                        "geometry_part_index": part_index,
                        "category": "building",
                        "feature_type": properties.get("feature_type", "building"),
                        "damage_class": damage_class,
                        "uid": uid,
                        "annotation_type": "polygon",
                        "coordinate_space": "pixel",
                        "polygon": xy_points,
                        "holes": polygon_to_holes(xy_polygon),
                        "bbox": polygon_to_coco_bbox(xy_points),
                        "area_pixels": float(xy_polygon.area),
                        "geo_coordinate_space": "lng_lat",
                        "geo_polygon": geo_points,
                        "geo_holes": polygon_to_holes(geo_polygon),
                        "source_xy_wkt": source_xy_wkt,
                        "source_lng_lat_wkt": source_geo_wkt,
                    }
                )

        return {
            "metadata": metadata,
            "annotations": annotations,
            "annotation_count": len(annotations),
            "parse_errors": parse_errors,
        }

    def load_sample(self, index_or_sample_id) -> Dict:
        if isinstance(index_or_sample_id, int):
            sample_id = self.sample_ids[index_or_sample_id]
        else:
            sample_id = str(index_or_sample_id)

        if sample_id not in self.image_map:
            raise KeyError(f"Unknown xBD sample_id: {sample_id}")

        paths = self._get_paths(sample_id)

        image_info = read_image_info(paths["image_path"])
        parsed = self._parse_annotations(paths["label_path"])

        phase = "unknown"

        if sample_id.endswith("_pre_disaster"):
            phase = "pre_disaster"
        elif sample_id.endswith("_post_disaster"):
            phase = "post_disaster"

        return {
            "dataset_id": "xbd",
            "dataset_name": "xBD / xView2",
            "split": self.split,
            "sample_id": sample_id,
            "disaster_phase": phase,
            "image": image_info,
            "label_path": str(paths["label_path"]),
            "target": {
                "path": str(paths["target_path"]) if paths["target_path"] else None,
                "available": paths["target_path"] is not None,
            },
            "metadata": parsed["metadata"],
            "annotations": parsed["annotations"],
            "annotation_count": parsed["annotation_count"],
            "parse_errors": parsed["parse_errors"],
            "raw_annotation_type": "polygon",
        }
