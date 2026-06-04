import json
from pathlib import Path
from typing import Dict, List, Optional

from disasterbench.loaders.xbd_loader import XBDLoader


def build_geojson_polygon_coordinates(
    outer_ring: List[List[float]],
    holes: Optional[List[List[List[float]]]] = None,
) -> List:
    if holes is None:
        holes = []

    return [outer_ring] + holes


def annotation_to_geojson_feature(sample: Dict, annotation: Dict) -> Dict:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": build_geojson_polygon_coordinates(
                outer_ring=annotation["geo_polygon"],
                holes=annotation.get("geo_holes", []),
            ),
        },
        "properties": {
            "dataset_id": "xbd",
            "dataset_name": "xBD / xView2",
            "sample_id": sample["sample_id"],
            "image_path": sample["image"]["path"],
            "label_path": sample["label_path"],
            "disaster_phase": sample["disaster_phase"],
            "disaster": sample["metadata"].get("disaster"),
            "disaster_type": sample["metadata"].get("disaster_type"),
            "capture_date": sample["metadata"].get("capture_date"),
            "sensor": sample["metadata"].get("sensor"),
            "annotation_id": annotation["annotation_id"],
            "uid": annotation.get("uid"),
            "category": annotation["category"],
            "feature_type": annotation["feature_type"],
            "damage_class": annotation["damage_class"],
            "source_annotation_type": "polygon",
            "coordinate_space": "lng_lat",
            "pixel_bbox": annotation["bbox"],
            "area_pixels": annotation["area_pixels"],
        },
    }


def export_xbd_geojson(
    output_path: str | Path,
    max_samples: Optional[int] = 10,
    split: str = "train",
    include_empty_samples: bool = False,
) -> Dict:
    """
    Export xBD building polygons to GeoJSON using lng_lat coordinates.

    Notes:
    - xBD raw labels contain both features.xy and features.lng_lat.
    - GeoJSON uses the lng_lat polygons verified by xBD geometry alignment checks.
    - Pixel bboxes are preserved in properties for traceability.
    """
    loader = XBDLoader(split=split)

    total_samples = len(loader)
    sample_limit = total_samples if max_samples is None else min(max_samples, total_samples)

    features = []
    samples_exported = 0
    empty_samples = 0

    for index in range(sample_limit):
        sample = loader.load_sample(index)
        samples_exported += 1

        if sample["annotation_count"] == 0:
            empty_samples += 1
            if not include_empty_samples:
                continue

        for annotation in sample["annotations"]:
            if "geo_polygon" not in annotation:
                continue

            feature = annotation_to_geojson_feature(sample, annotation)
            features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "dataset_id": "xbd",
            "dataset_name": "xBD / xView2",
            "split": split,
            "source_annotation_type": "polygon",
            "coordinate_space": "lng_lat",
            "samples_exported": samples_exported,
            "empty_samples": empty_samples,
            "features_exported": len(features),
            "note": "GeoJSON polygons are exported from xBD features.lng_lat. Pixel bboxes are stored in feature properties.",
        },
        "features": features,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    return {
        "output_path": str(output_path),
        "samples_exported": samples_exported,
        "empty_samples": empty_samples,
        "features_exported": len(features),
    }
