import json
from pathlib import Path
from typing import Dict, List, Optional

from disasterbench.converters.mask_to_polygon import mask_file_to_polygons
from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


def _close_ring(ring: List[List[float]]) -> List[List[float]]:
    if not ring:
        return ring

    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]

    return ring


def build_geojson_feature(
    polygon_record: Dict,
    sample: Dict,
    feature_id: int,
) -> Dict:
    exterior = _close_ring(polygon_record["polygon"])

    holes = [
        _close_ring(hole)
        for hole in polygon_record.get("holes", [])
        if len(hole) >= 4
    ]

    coordinates = [exterior] + holes

    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates,
        },
        "properties": {
            "dataset_id": sample["dataset_id"],
            "sensor": sample["sensor"],
            "sample_id": sample["sample_id"],
            "source_annotation_type": "raster_mask",
            "derived_geometry": "polygon",
            "coordinate_space": polygon_record["coordinate_space"],
            "area_pixels": polygon_record["area_pixels"],
            "source_mask_values": polygon_record["source_mask_values"],
            "hole_count": len(holes),
            "event_type": sample["metadata"].get("event_type"),
            "country": sample["metadata"].get("country"),
            "ems_code": sample["metadata"].get("ems_code"),
            "aoi_code": sample["metadata"].get("aoi_code"),
            "floodmap_id": sample["metadata"].get("floodmap_id"),
            "tile_id": sample["metadata"].get("tile_id"),
            "epsg_code": sample["metadata"].get("epsg_code"),
        },
    }


def export_sturm_flood_geojson(
    sensor: str,
    output_path: str | Path,
    max_samples: Optional[int] = 10,
    target_values: Optional[List[int]] = None,
    min_area_pixels: float = 2.0,
    coordinate_space: str = "geo",
) -> Dict:
    """
    Export STURM-Flood mask-derived polygons to GeoJSON.

    This converts raster flood mask regions into polygon features.
    It preserves polygon holes when they exist.
    """
    if target_values is None:
        target_values = [1, 2, 3, 4, 5]

    loader = STURMFloodLoader(sensor=sensor)

    total_samples = len(loader.samples)
    sample_limit = total_samples if max_samples is None else min(max_samples, total_samples)

    features = []
    feature_id = 1

    for index in range(sample_limit):
        sample = loader.load_sample(index)
        mask_path = sample["annotations"][0]["mask_path"]

        polygons = mask_file_to_polygons(
            mask_path=mask_path,
            target_values=target_values,
            min_area_pixels=min_area_pixels,
            coordinate_space=coordinate_space,
            preserve_holes=True,
        )

        for polygon_record in polygons:
            feature = build_geojson_feature(
                polygon_record=polygon_record,
                sample=sample,
                feature_id=feature_id,
            )
            features.append(feature)
            feature_id += 1

    geojson = {
        "type": "FeatureCollection",
        "name": f"sturm_flood_{sensor}_polygons",
        "metadata": {
            "dataset_id": "sturm_flood",
            "sensor": sensor,
            "source_annotation_type": "raster_mask",
            "export_format": "geojson",
            "coordinate_space": coordinate_space,
            "target_values": target_values,
            "min_area_pixels": min_area_pixels,
            "samples_exported": sample_limit,
            "features_exported": len(features),
            "note": "Polygons are derived from raster flood masks.",
        },
        "features": features,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    return {
        "output_path": str(output_path),
        "samples_exported": sample_limit,
        "features_exported": len(features),
    }
