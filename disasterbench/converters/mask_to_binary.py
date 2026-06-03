from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio


DEFAULT_WATER_VALUES = [1, 2, 3, 4, 5]
DEFAULT_NON_WATER_VALUES = [0]
DEFAULT_NO_DATA_VALUES = [99]


def mask_array_to_binary(
    mask: np.ndarray,
    water_values: Iterable[int] = DEFAULT_WATER_VALUES,
    non_water_values: Iterable[int] = DEFAULT_NON_WATER_VALUES,
    no_data_values: Iterable[int] = DEFAULT_NO_DATA_VALUES,
    no_data_output_value: int = 255,
) -> np.ndarray:
    """
    Convert a multiclass STURM-Flood mask array into a binary water mask.

    Output values:
    - 0 = non-water
    - 1 = water
    - 255 = no-data
    """
    binary = np.zeros(mask.shape, dtype=np.uint8)

    water_pixels = np.isin(mask, list(water_values))
    no_data_pixels = np.isin(mask, list(no_data_values))

    binary[water_pixels] = 1
    binary[no_data_pixels] = no_data_output_value

    return binary


def mask_file_to_binary(
    mask_path: str | Path,
    output_path: Optional[str | Path] = None,
    water_values: Iterable[int] = DEFAULT_WATER_VALUES,
    non_water_values: Iterable[int] = DEFAULT_NON_WATER_VALUES,
    no_data_values: Iterable[int] = DEFAULT_NO_DATA_VALUES,
    no_data_output_value: int = 255,
) -> np.ndarray:
    """
    Read a raster mask file and convert it into a binary water mask.

    If output_path is provided, the binary mask is saved as a GeoTIFF.
    """
    mask_path = Path(mask_path)

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        profile = src.profile.copy()

    binary = mask_array_to_binary(
        mask=mask,
        water_values=water_values,
        non_water_values=non_water_values,
        no_data_values=no_data_values,
        no_data_output_value=no_data_output_value,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile.update(
            dtype="uint8",
            count=1,
            nodata=no_data_output_value,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(binary, 1)

    return binary


def summarize_binary_mask(binary_mask: np.ndarray) -> dict:
    """
    Return counts for each value in a binary mask.
    """
    unique_values, counts = np.unique(binary_mask, return_counts=True)

    return {
        int(value): int(count)
        for value, count in zip(unique_values, counts)
    }
