from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from disasterbench.loaders.json_polygon_loader import JsonPolygonLoader
from disasterbench.loaders.npz_segmentation_loader import NPZSegmentationLoader
from disasterbench.loaders.raster_mask_pair_loader import RasterMaskPairLoader
from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader
from disasterbench.loaders.xbd_loader import XBDLoader


SUPPORTED_CONFIG_LOADERS = {
    "JsonPolygonLoader": JsonPolygonLoader,
    "NPZSegmentationLoader": NPZSegmentationLoader,
    "RasterMaskPairLoader": RasterMaskPairLoader,
    "STURMFloodLoader": STURMFloodLoader,
    "XBDLoader": XBDLoader,
}


def read_dataset_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def supported_loader_names() -> List[str]:
    return sorted(SUPPORTED_CONFIG_LOADERS.keys())


def create_loader_from_config(
    config_path: str | Path,
    **loader_kwargs: Any,
):
    """
    Create the correct dataset loader from a dataset config file.

    The config must contain:

    {
      "loader_name": "RasterMaskPairLoader"
    }

    Generic loaders receive config_path directly.
    Legacy dataset-specific loaders are handled with light compatibility logic.
    """
    config_path = Path(config_path)
    config = read_dataset_config(config_path)

    loader_name = config.get("loader_name")

    if not loader_name:
        raise ValueError(f"Config does not contain 'loader_name': {config_path}")

    if loader_name not in SUPPORTED_CONFIG_LOADERS:
        raise ValueError(
            f"Unsupported loader_name '{loader_name}'. "
            f"Supported loaders: {supported_loader_names()}"
        )

    if loader_name in {"JsonPolygonLoader", "NPZSegmentationLoader", "RasterMaskPairLoader", "STURMFloodLoader"}:
        loader_class = SUPPORTED_CONFIG_LOADERS[loader_name]
        return loader_class(config_path=config_path, **loader_kwargs)

    if loader_name == "XBDLoader":
        dataset_root = config.get("dataset_path", "datasets/xbd_raw")
        loader_class = SUPPORTED_CONFIG_LOADERS[loader_name]
        return loader_class(dataset_root=dataset_root, **loader_kwargs)

    raise ValueError(f"Unhandled loader_name: {loader_name}")
