from pathlib import Path
from typing import Iterable, List, Literal, Optional

import cv2
import numpy as np
import rasterio
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection


DEFAULT_TARGET_VALUES = [1, 2, 3, 4, 5]


def _contour_to_ring(contour: np.ndarray) -> List[List[float]]:
    points = contour.squeeze(axis=1)

    ring = [
        [float(x), float(y)]
        for x, y in points
    ]

    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])

    return ring


def _pixel_ring_to_geo_ring(pixel_ring: List[List[float]], transform) -> List[List[float]]:
    xs = [point[0] for point in pixel_ring]
    ys = [point[1] for point in pixel_ring]

    geo_xs, geo_ys = rasterio.transform.xy(transform, ys, xs, offset="center")

    return [
        [float(x), float(y)]
        for x, y in zip(geo_xs, geo_ys)
    ]


def _extract_polygon_geometries(geometry):
    """
    Convert Polygon / MultiPolygon / GeometryCollection into a list of Polygon objects.
    """
    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)

    if isinstance(geometry, GeometryCollection):
        polygons = []

        for geom in geometry.geoms:
            polygons.extend(_extract_polygon_geometries(geom))

        return polygons

    return []


def _polygon_geometry_to_record(
    polygon_geometry: Polygon,
    target_values: Iterable[int],
    coordinate_space: Literal["pixel", "geo"],
    transform=None,
) -> dict:
    exterior_pixel = [
        [float(x), float(y)]
        for x, y in polygon_geometry.exterior.coords
    ]

    holes_pixel = [
        [
            [float(x), float(y)]
            for x, y in interior.coords
        ]
        for interior in polygon_geometry.interiors
    ]

    if coordinate_space == "geo":
        if transform is None:
            raise ValueError("transform is required when coordinate_space='geo'")

        exterior = _pixel_ring_to_geo_ring(exterior_pixel, transform)
        holes = [
            _pixel_ring_to_geo_ring(hole, transform)
            for hole in holes_pixel
        ]
    else:
        exterior = exterior_pixel
        holes = holes_pixel

    return {
        "polygon": exterior,
        "holes": holes,
        "area_pixels": float(polygon_geometry.area),
        "coordinate_space": coordinate_space,
        "source_mask_values": list(target_values),
    }


def mask_array_to_polygons(
    mask: np.ndarray,
    target_values: Iterable[int] = DEFAULT_TARGET_VALUES,
    min_area_pixels: float = 1.0,
    simplify_tolerance: float = 0.0,
    coordinate_space: Literal["pixel", "geo"] = "pixel",
    transform=None,
    max_polygons: Optional[int] = None,
    preserve_holes: bool = True,
) -> List[dict]:
    """
    Convert selected class regions in a mask array into polygons.

    coordinate_space:
    - "pixel": coordinates are image pixel coordinates
    - "geo": coordinates are projected map coordinates using raster transform

    Output:
    - polygon: exterior ring
    - holes: interior rings, if any

    Notes:
    - preserve_holes=True uses contour hierarchy to keep interior non-water gaps.
    - Some repaired raster contours can become MultiPolygons.
    - MultiPolygons are split into individual Polygon records.
    """
    binary = np.isin(mask, list(target_values)).astype(np.uint8)

    retrieval_mode = cv2.RETR_CCOMP if preserve_holes else cv2.RETR_EXTERNAL

    contours, hierarchy = cv2.findContours(
        binary,
        retrieval_mode,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    polygons = []

    for contour_index, contour in enumerate(contours):
        parent_index = hierarchy[contour_index][3]

        # Only start from external contours. Holes are attached as children.
        if parent_index != -1:
            continue

        if len(contour) < 3:
            continue

        exterior = _contour_to_ring(contour)

        if len(exterior) < 4:
            continue

        holes = []

        if preserve_holes:
            child_index = hierarchy[contour_index][2]

            while child_index != -1:
                child_contour = contours[child_index]

                if len(child_contour) >= 3:
                    hole = _contour_to_ring(child_contour)

                    if len(hole) >= 4:
                        holes.append(hole)

                child_index = hierarchy[child_index][0]

        try:
            geometry = Polygon(exterior, holes)
        except Exception:
            geometry = Polygon(exterior)

        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        polygon_geometries = _extract_polygon_geometries(geometry)

        for polygon_geometry in polygon_geometries:
            if polygon_geometry.is_empty:
                continue

            if polygon_geometry.area < min_area_pixels:
                continue

            if simplify_tolerance > 0:
                polygon_geometry = polygon_geometry.simplify(
                    simplify_tolerance,
                    preserve_topology=True,
                )

            if polygon_geometry.is_empty:
                continue

            record = _polygon_geometry_to_record(
                polygon_geometry=polygon_geometry,
                target_values=target_values,
                coordinate_space=coordinate_space,
                transform=transform,
            )

            if len(record["polygon"]) < 4:
                continue

            polygons.append(record)

            if max_polygons is not None and len(polygons) >= max_polygons:
                return polygons

    return polygons


def mask_file_to_polygons(
    mask_path: str | Path,
    target_values: Iterable[int] = DEFAULT_TARGET_VALUES,
    min_area_pixels: float = 1.0,
    simplify_tolerance: float = 0.0,
    coordinate_space: Literal["pixel", "geo"] = "pixel",
    max_polygons: Optional[int] = None,
    preserve_holes: bool = True,
) -> List[dict]:
    """
    Read a raster mask file and convert selected mask values into polygons.
    """
    mask_path = Path(mask_path)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform

    return mask_array_to_polygons(
        mask=mask,
        target_values=target_values,
        min_area_pixels=min_area_pixels,
        simplify_tolerance=simplify_tolerance,
        coordinate_space=coordinate_space,
        transform=transform,
        max_polygons=max_polygons,
        preserve_holes=preserve_holes,
    )
