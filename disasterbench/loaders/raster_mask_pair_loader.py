from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio

from disasterbench.loaders.base_loader import BaseDatasetLoader
from disasterbench.schemas.common_sample import (
    build_image_info,
    build_mask_annotation,
    validate_common_sample,
)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_int_key_dict(mapping: Dict[str, str]) -> Dict[int, str]:
    converted = {}

    for key, value in mapping.items():
        try:
            converted[int(key)] = value
        except ValueError:
            converted[key] = value

    return converted


def _collect_rasters(folder: Path, extensions: List[str]) -> List[Path]:
    files: List[Path] = []

    for extension in extensions:
        pattern = f"*{extension}"
        files.extend(folder.glob(pattern))

    return sorted(files)


def _normalize_stem(path: Path, suffixes_to_remove: List[str] | None = None) -> str:
    stem = path.stem
    suffixes_to_remove = suffixes_to_remove or []

    for suffix in suffixes_to_remove:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return stem


class RasterMaskPairLoader(BaseDatasetLoader):
    """
    Generic loader for raster image + raster mask datasets.

    Expected pattern:
    - image_folder contains GeoTIFF / raster image files
    - mask_folder contains corresponding GeoTIFF / raster mask files
    - image and mask can be matched by normalized filename stem

    This loader is dataset-family based, not dataset-specific.
    It can support datasets such as GDCLD, MMFlood, STURM-style datasets,
    and future raster image-mask datasets through config files.
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

        self.task_type = "semantic_segmentation"
        self.pairing_groups = self.config.get("pairing_groups", [])

        if not self.pairing_groups:
            raise ValueError("Config must contain at least one pairing group.")

        self.label_values = _safe_int_key_dict(
            self.config.get("label_schema", {}).get("label_values", {})
        )

        self.samples = self._build_sample_index()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _build_sample_index(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        for group in self.pairing_groups:
            group_name = group["name"]
            split = group.get("split", group_name)

            image_folder = self.dataset_root / group["image_folder"]
            mask_folder = self.dataset_root / group["mask_folder"]

            image_extensions = group.get("image_extensions", [".tif", ".tiff"])
            mask_extensions = group.get("mask_extensions", [".tif", ".tiff"])

            image_suffixes = group.get("image_suffixes_to_remove", [])
            mask_suffixes = group.get("mask_suffixes_to_remove", [])

            if not image_folder.exists():
                continue

            if not mask_folder.exists():
                continue

            image_files = _collect_rasters(image_folder, image_extensions)
            mask_files = _collect_rasters(mask_folder, mask_extensions)

            image_map = {
                _normalize_stem(path, image_suffixes): path
                for path in image_files
            }
            mask_map = {
                _normalize_stem(path, mask_suffixes): path
                for path in mask_files
            }

            matched_ids = sorted(set(image_map) & set(mask_map))

            for sample_id in matched_ids:
                samples.append(
                    {
                        "sample_id": f"{group_name}__{sample_id}",
                        "local_id": sample_id,
                        "group": group_name,
                        "split": split,
                        "image_path": image_map[sample_id],
                        "mask_path": mask_map[sample_id],
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

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
                "transform": list(src.transform),
                "nodata": src.nodata,
            }

    def _get_mask_values_for_file(self, mask_path: Path) -> List[int]:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        return sorted([int(value) for value in np.unique(mask)])

    def list_samples(self) -> List[Dict[str, str]]:
        return [
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "group": sample["group"],
                "image_path": str(sample["image_path"]),
                "mask_path": str(sample["mask_path"]),
            }
            for sample in self.samples
        ]

    def load_sample(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Sample index {index} out of range. Dataset has {len(self.samples)} samples."
            )

        sample_record = self.samples[index]

        image_path = sample_record["image_path"]
        mask_path = sample_record["mask_path"]

        image_meta = self._inspect_raster_metadata(image_path)
        mask_meta = self._inspect_raster_metadata(mask_path)
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

        image_info["raster_transform"] = image_meta["transform"]
        image_info["nodata"] = image_meta["nodata"]

        annotation = build_mask_annotation(
            mask_path=str(mask_path),
            mask_values=self.label_values,
            geometry_type="raster_mask",
            mask_mode=self.config.get("mask_mode", "original"),
        )

        annotation["raw_mask_values_found_in_file"] = mask_values_found
        annotation["mask_width"] = mask_meta["width"]
        annotation["mask_height"] = mask_meta["height"]
        annotation["mask_band_count"] = mask_meta["band_count"]
        annotation["mask_dtypes"] = mask_meta["dtypes"]
        annotation["mask_crs"] = mask_meta["crs"]
        annotation["mask_bounds"] = mask_meta["bounds"]
        annotation["mask_transform"] = mask_meta["transform"]
        annotation["mask_nodata"] = mask_meta["nodata"]

        common_sample = {
            "dataset_id": self.dataset_id,
            "sample_id": sample_record["sample_id"],
            "split": sample_record["split"],
            "task_type": self.task_type,
            "image": image_info,
            "annotations": [annotation],
            "metadata": {
                "group": sample_record["group"],
                "local_id": sample_record["local_id"],
                "source_format": "raster_image_mask_pair",
            },
        }

        validate_common_sample(common_sample)

        return common_sample

    def validate_pairing(self) -> Dict[str, Any]:
        group_reports = []
        total_matched = 0
        total_missing_masks = 0
        total_extra_masks = 0

        for group in self.pairing_groups:
            group_name = group["name"]

            image_folder = self.dataset_root / group["image_folder"]
            mask_folder = self.dataset_root / group["mask_folder"]

            image_extensions = group.get("image_extensions", [".tif", ".tiff"])
            mask_extensions = group.get("mask_extensions", [".tif", ".tiff"])

            image_suffixes = group.get("image_suffixes_to_remove", [])
            mask_suffixes = group.get("mask_suffixes_to_remove", [])

            image_files = (
                _collect_rasters(image_folder, image_extensions)
                if image_folder.exists()
                else []
            )
            mask_files = (
                _collect_rasters(mask_folder, mask_extensions)
                if mask_folder.exists()
                else []
            )

            if not image_folder.exists() or not mask_folder.exists():
                image_map = {
                    _normalize_stem(path, image_suffixes): path
                    for path in image_files
                }
                mask_map = {
                    _normalize_stem(path, mask_suffixes): path
                    for path in mask_files
                }

                image_ids = set(image_map)
                mask_ids = set(mask_map)

                missing_masks = sorted(image_ids - mask_ids)
                extra_masks = sorted(mask_ids - image_ids)

                total_missing_masks += len(missing_masks)
                total_extra_masks += len(extra_masks)

                group_reports.append(
                    {
                        "group": group_name,
                        "valid": False,
                        "reason": "missing_folder",
                        "image_folder_exists": image_folder.exists(),
                        "mask_folder_exists": mask_folder.exists(),
                        "image_count": len(image_files),
                        "mask_count": len(mask_files),
                        "matched_count": 0,
                        "missing_mask_count": len(missing_masks),
                        "extra_mask_count": len(extra_masks),
                        "missing_mask_examples": missing_masks[:20],
                        "extra_mask_examples": extra_masks[:20],
                    }
                )
                continue

            image_map = {
                _normalize_stem(path, image_suffixes): path
                for path in image_files
            }
            mask_map = {
                _normalize_stem(path, mask_suffixes): path
                for path in mask_files
            }

            image_ids = set(image_map)
            mask_ids = set(mask_map)

            matched = sorted(image_ids & mask_ids)
            missing_masks = sorted(image_ids - mask_ids)
            extra_masks = sorted(mask_ids - image_ids)

            total_matched += len(matched)
            total_missing_masks += len(missing_masks)
            total_extra_masks += len(extra_masks)

            group_reports.append(
                {
                    "group": group_name,
                    "image_folder": str(image_folder),
                    "mask_folder": str(mask_folder),
                    "image_count": len(image_files),
                    "mask_count": len(mask_files),
                    "matched_count": len(matched),
                    "missing_mask_count": len(missing_masks),
                    "extra_mask_count": len(extra_masks),
                    "missing_mask_examples": missing_masks[:20],
                    "extra_mask_examples": extra_masks[:20],
                    "valid": len(missing_masks) == 0 and len(extra_masks) == 0,
                }
            )

        return {
            "dataset_id": self.dataset_id,
            "group_count": len(self.pairing_groups),
            "total_matched_count": total_matched,
            "total_missing_mask_count": total_missing_masks,
            "total_extra_mask_count": total_extra_masks,
            "groups": group_reports,
            "valid": total_missing_masks == 0 and total_extra_masks == 0,
        }

    def get_statistics(self) -> Dict[str, Any]:
        split_counts: Dict[str, int] = {}
        group_counts: Dict[str, int] = {}

        for sample in self.samples:
            split = sample["split"] or "all"
            group = sample["group"]

            split_counts[split] = split_counts.get(split, 0) + 1
            group_counts[group] = group_counts.get(group, 0) + 1

        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.config.get("dataset_name"),
            "loader_name": "RasterMaskPairLoader",
            "data_family": self.config.get("data_family"),
            "raw_annotation_type": self.config.get("raw_annotation_type"),
            "task_type": self.config.get("task_type", []),
            "total_samples": len(self.samples),
            "split_counts": split_counts,
            "group_counts": group_counts,
            "label_schema": self.config.get("label_schema", {}),
        }
