import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import rasterio

from disasterbench.loaders.base_loader import BaseDatasetLoader
from disasterbench.schemas.common_sample import (
    build_image_info,
    build_mask_annotation,
    validate_common_sample,
)


STURM_MASK_VALUES = {
    0: "coastline_or_area_of_interest",
    1: "flooded_area",
    2: "river_related_features",
    3: "open_water",
    4: "reservoirs",
    5: "lakes",
    99: "no_data",
}

STURM_BINARY_WATER_MAPPING = {
    "non_water": [0],
    "water": [1, 2, 3, 4, 5],
    "no_data": [99],
}


class STURMFloodLoader(BaseDatasetLoader):
    """
    Dataset-specific loader for STURM-Flood.

    This loader reads:
    - Sentinel-1 or Sentinel-2 GeoTIFF image tiles
    - Corresponding GeoTIFF floodmap masks
    - Metadata CSV rows

    It returns samples using the DisasterBench common internal schema.
    """

    VALID_SENSORS = ["sentinel1", "sentinel2"]
    VALID_MASK_MODES = ["original", "binary_water"]

    def __init__(
        self,
        config_path: str = "configs/sturm_flood_config.json",
        sensor: str = "sentinel1",
        mask_mode: str = "original",
    ):
        self.config_path = Path(config_path)
        self.sensor = sensor.lower()
        self.mask_mode = mask_mode

        if self.sensor not in self.VALID_SENSORS:
            raise ValueError(f"Invalid sensor '{sensor}'. Choose from: {self.VALID_SENSORS}")

        if self.mask_mode not in self.VALID_MASK_MODES:
            raise ValueError(f"Invalid mask_mode '{mask_mode}'. Choose from: {self.VALID_MASK_MODES}")

        self.config = self._load_config(self.config_path)
        self.dataset_root = Path(self.config.get("dataset_path", "datasets/sturm_flood_raw"))

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        sensor_paths = self._get_sensor_paths(self.sensor)

        self.image_dir = self.dataset_root / sensor_paths["image_folder"]
        self.mask_dir = self.dataset_root / sensor_paths["mask_folder"]
        self.metadata_path = self.dataset_root / sensor_paths["metadata_file"]

        self._validate_required_paths()

        self.image_files = self._count_raster_files(self.image_dir)
        self.mask_files = self._count_raster_files(self.mask_dir)
        self.metadata_df = pd.read_csv(self.metadata_path)

        self.samples = self._build_sample_index()

    def _load_config(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _get_sensor_paths(self, sensor: str) -> Dict[str, str]:
        """
        Folder layout verified from the real STURM-Flood dataset.

        Sentinel1:
        - Sentinel1/S1
        - Sentinel1/Floodmaps
        - sentinel1_metadata.csv

        Sentinel2:
        - Sentinel2/S2
        - Sentinel2/Floodmaps
        - sentinel2_metadata.csv
        """
        if sensor == "sentinel1":
            return {
                "image_folder": "Sentinel1/S1",
                "mask_folder": "Sentinel1/Floodmaps",
                "metadata_file": "sentinel1_metadata.csv",
            }

        if sensor == "sentinel2":
            return {
                "image_folder": "Sentinel2/S2",
                "mask_folder": "Sentinel2/Floodmaps",
                "metadata_file": "sentinel2_metadata.csv",
            }

        raise ValueError(f"Unknown sensor: {sensor}")

    def _validate_required_paths(self) -> None:
        required_paths = [self.image_dir, self.mask_dir, self.metadata_path]

        missing = [str(path) for path in required_paths if not path.exists()]

        if missing:
            raise FileNotFoundError(f"Missing required STURM-Flood paths: {missing}")

    def _count_raster_files(self, folder: Path) -> List[Path]:
        files = []

        for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            files.extend(folder.glob(ext))

        return sorted(files)

    def _build_sample_index(self) -> List[Dict[str, Any]]:
        image_map = {path.stem: path for path in self.image_files}
        mask_map = {path.stem: path for path in self.mask_files}

        if "tile_id" not in self.metadata_df.columns:
            raise ValueError("Metadata CSV does not contain required column: tile_id")

        metadata_df = self.metadata_df.copy()
        metadata_df["tile_stem"] = metadata_df["tile_id"].astype(str).apply(lambda value: Path(value).stem)

        samples = []

        for _, row in metadata_df.iterrows():
            tile_stem = row["tile_stem"]

            image_path = image_map.get(tile_stem)
            mask_path = mask_map.get(tile_stem)

            if image_path is None or mask_path is None:
                continue

            samples.append(
                {
                    "sample_id": tile_stem,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "metadata": row.drop(labels=["tile_stem"]).to_dict(),
                }
            )

        return samples

    def _inspect_raster_metadata(self, path: Path) -> Dict[str, Any]:
        with rasterio.open(path) as src:
            return {
                "width": int(src.width),
                "height": int(src.height),
                "band_count": int(src.count),
                "dtypes": list(src.dtypes),
                "crs": str(src.crs) if src.crs else None,
                "bounds": {
                    "left": float(src.bounds.left),
                    "bottom": float(src.bounds.bottom),
                    "right": float(src.bounds.right),
                    "top": float(src.bounds.top),
                },
                "nodata": src.nodata,
            }

    def _get_mask_values_for_file(self, mask_path: Path) -> List[int]:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        return sorted([int(value) for value in np.unique(mask)])

    def _build_binary_mask_values(self) -> Dict[int, str]:
        return {
            0: "non_water",
            1: "water",
            255: "no_data",
        }

    def list_samples(self) -> List[Dict[str, str]]:
        return [
            {
                "sample_id": sample["sample_id"],
                "image_path": str(sample["image_path"]),
                "mask_path": str(sample["mask_path"]),
            }
            for sample in self.samples
        ]

    def load_sample(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.samples):
            raise IndexError(f"Sample index {index} out of range. Dataset has {len(self.samples)} samples.")

        sample_record = self.samples[index]

        image_path = sample_record["image_path"]
        mask_path = sample_record["mask_path"]

        image_meta = self._inspect_raster_metadata(image_path)
        mask_values_found = self._get_mask_values_for_file(mask_path)

        image_info = build_image_info(
            path=str(image_path),
            width=image_meta["width"],
            height=image_meta["height"],
            band_count=image_meta["band_count"],
            dtypes=image_meta["dtypes"],
            crs=image_meta["crs"],
            bounds=image_meta["bounds"],
        )

        if self.mask_mode == "binary_water":
            mask_values = self._build_binary_mask_values()
        else:
            mask_values = STURM_MASK_VALUES

        annotation = build_mask_annotation(
            mask_path=str(mask_path),
            mask_values=mask_values,
            geometry_type="raster_mask",
            mask_mode=self.mask_mode,
        )

        annotation["raw_mask_values_found_in_file"] = mask_values_found
        annotation["binary_water_mapping"] = STURM_BINARY_WATER_MAPPING

        common_sample = {
            "dataset_id": "sturm_flood",
            "sample_id": sample_record["sample_id"],
            "sensor": self.sensor,
            "task_type": "semantic_segmentation",
            "image": image_info,
            "annotations": [annotation],
            "metadata": sample_record["metadata"],
        }

        validate_common_sample(common_sample)

        return common_sample

    def validate_pairing(self) -> Dict[str, int]:
        image_stems = {path.stem for path in self.image_files}
        mask_stems = {path.stem for path in self.mask_files}
        metadata_stems = set(self.metadata_df["tile_id"].astype(str).apply(lambda value: Path(value).stem))

        return {
            "image_files": len(image_stems),
            "mask_files": len(mask_stems),
            "metadata_rows": int(len(self.metadata_df)),
            "images_without_masks": len(image_stems - mask_stems),
            "masks_without_images": len(mask_stems - image_stems),
            "images_without_metadata": len(image_stems - metadata_stems),
            "metadata_without_images": len(metadata_stems - image_stems),
            "indexed_samples": len(self.samples),
        }

    def get_statistics(self) -> Dict[str, Any]:
        pairing = self.validate_pairing()

        mask_values = set()

        for sample in self.samples[:100]:
            values = self._get_mask_values_for_file(sample["mask_path"])
            mask_values.update(values)

        first_sample = self.load_sample(0)

        return {
            "dataset_id": "sturm_flood",
            "sensor": self.sensor,
            "mask_mode": self.mask_mode,
            "counts": pairing,
            "task_type": "semantic_segmentation",
            "raw_annotation_type": "raster_mask",
            "recommended_formats": [
                "mask_segmentation",
                "coco_segmentation",
                "yolo_segmentation",
                "geojson",
                "statistics",
            ],
            "lossy_formats": [
                "yolo_detection_bbox",
                "coco_detection_bbox",
            ],
            "mask_values_found_in_first_100_samples": sorted([int(value) for value in mask_values]),
            "first_sample_summary": {
                "sample_id": first_sample["sample_id"],
                "image_width": first_sample["image"]["width"],
                "image_height": first_sample["image"]["height"],
                "image_band_count": first_sample["image"]["band_count"],
                "image_dtypes": first_sample["image"]["dtypes"],
                "image_crs": first_sample["image"]["crs"],
            },
        }
