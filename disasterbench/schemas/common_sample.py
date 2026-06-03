from typing import Any, Dict, List


REQUIRED_COMMON_SAMPLE_KEYS = [
    "dataset_id",
    "sample_id",
    "task_type",
    "image",
    "annotations",
    "metadata",
]


def validate_common_sample(sample: Dict[str, Any]) -> bool:
    missing_keys = []

    for key in REQUIRED_COMMON_SAMPLE_KEYS:
        if key not in sample:
            missing_keys.append(key)

    if missing_keys:
        raise ValueError(f"Common sample missing required keys: {missing_keys}")

    if not isinstance(sample["image"], dict):
        raise TypeError("sample['image'] must be a dictionary")

    if not isinstance(sample["annotations"], list):
        raise TypeError("sample['annotations'] must be a list")

    if not isinstance(sample["metadata"], dict):
        raise TypeError("sample['metadata'] must be a dictionary")

    return True


def build_image_info(
    path: str,
    width: int,
    height: int,
    band_count: int,
    dtypes: List[str],
    crs: str | None = None,
    bounds: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    return {
        "path": path,
        "width": width,
        "height": height,
        "band_count": band_count,
        "dtypes": dtypes,
        "crs": crs,
        "bounds": bounds,
    }


def build_mask_annotation(
    mask_path: str,
    mask_values: Dict[int, str],
    geometry_type: str = "raster_mask",
    mask_mode: str = "original",
) -> Dict[str, Any]:
    return {
        "type": "segmentation_mask",
        "geometry_type": geometry_type,
        "mask_path": mask_path,
        "mask_mode": mask_mode,
        "mask_values": mask_values,
    }
