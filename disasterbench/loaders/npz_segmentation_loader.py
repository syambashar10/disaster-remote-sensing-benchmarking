from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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


def _sample_id_from_path(dataset_root: Path, path: Path) -> str:
    relative = path.relative_to(dataset_root)
    return str(relative.with_suffix("")).replace("/", "__")


class NPZSegmentationLoader(BaseDatasetLoader):
    """
    Generic loader for NPZ-packaged semantic segmentation datasets.

    Expected pattern:
    - Each sample is stored as one .npz file.
    - The .npz file contains an image array and a label/mask array.
    - Optional auxiliary arrays can also be stored, such as aerosol, DEM, etc.

    Example:
    sample.npz
    ├── image   -> shape (bands, height, width)
    ├── aerosol -> shape (height, width)
    └── label   -> shape (height, width)
    """

    def __init__(
        self,
        config_path: str | Path = "configs/sen2fire_config.json",
        max_samples: int | None = None,
    ):
        self.config_path = Path(config_path)
        self.config = _load_json(self.config_path)

        self.dataset_id = self.config["dataset_id"]
        self.dataset_root = Path(self.config["dataset_path"])

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

        self.npz_schema = self.config.get("npz_schema", {})
        self.image_key = self.npz_schema.get("image_key", "image")
        self.label_key = self.npz_schema.get("label_key", "label")
        self.auxiliary_keys = self.npz_schema.get("auxiliary_keys", [])

        self.task_type = "semantic_segmentation"
        self.label_values = _safe_int_key_dict(
            self.config.get("label_schema", {}).get("label_values", {})
        )

        self.samples = self._build_sample_index()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _build_sample_index(self) -> List[Dict[str, Any]]:
        split_structure = self.config.get("split_structure", {})

        samples = []

        if split_structure:
            for split_name, split_folder in split_structure.items():
                split_root = self.dataset_root / split_folder

                if not split_root.exists():
                    continue

                for npz_path in sorted(split_root.rglob("*.npz")):
                    samples.append(
                        {
                            "sample_id": _sample_id_from_path(self.dataset_root, npz_path),
                            "split": split_name,
                            "npz_path": npz_path,
                        }
                    )
        else:
            for npz_path in sorted(self.dataset_root.rglob("*.npz")):
                samples.append(
                    {
                        "sample_id": _sample_id_from_path(self.dataset_root, npz_path),
                        "split": None,
                        "npz_path": npz_path,
                    }
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def list_samples(self) -> List[Dict[str, str]]:
        return [
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "npz_path": str(sample["npz_path"]),
            }
            for sample in self.samples
        ]

    def _inspect_npz(self, npz_path: Path) -> Dict[str, Any]:
        with np.load(npz_path, allow_pickle=False) as data:
            keys = list(data.files)

            if self.image_key not in keys:
                raise ValueError(f"Missing image key '{self.image_key}' in {npz_path}")

            if self.label_key not in keys:
                raise ValueError(f"Missing label key '{self.label_key}' in {npz_path}")

            image = data[self.image_key]
            label = data[self.label_key]

            if image.ndim == 3:
                band_count = int(image.shape[0])
                height = int(image.shape[1])
                width = int(image.shape[2])
            elif image.ndim == 2:
                band_count = 1
                height = int(image.shape[0])
                width = int(image.shape[1])
            else:
                raise ValueError(
                    f"Unsupported image array shape {image.shape} in {npz_path}"
                )

            label_values = sorted([int(value) for value in np.unique(label)])

            arrays = {}

            for key in keys:
                arr = data[key]
                arrays[key] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }

            return {
                "keys": keys,
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "label_shape": list(label.shape),
                "label_dtype": str(label.dtype),
                "label_values_found": label_values,
                "width": width,
                "height": height,
                "band_count": band_count,
                "arrays": arrays,
            }

    def load_sample(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.samples):
            raise IndexError(
                f"Sample index {index} out of range. Dataset has {len(self.samples)} samples."
            )

        sample_record = self.samples[index]
        npz_path = sample_record["npz_path"]
        npz_info = self._inspect_npz(npz_path)

        image_info = build_image_info(
            path=str(npz_path),
            width=npz_info["width"],
            height=npz_info["height"],
            band_count=npz_info["band_count"],
            dtypes=[npz_info["image_dtype"]],
            crs=None,
            bounds=None,
        )

        image_info["container_format"] = "npz"
        image_info["array_key"] = self.image_key
        image_info["auxiliary_keys"] = self.auxiliary_keys

        annotation = build_mask_annotation(
            mask_path=str(npz_path),
            mask_values=self.label_values,
            geometry_type="array_mask",
            mask_mode="binary",
        )

        annotation["container_format"] = "npz"
        annotation["label_key"] = self.label_key
        annotation["raw_label_values_found_in_file"] = npz_info["label_values_found"]
        annotation["label_shape"] = npz_info["label_shape"]
        annotation["label_dtype"] = npz_info["label_dtype"]

        common_sample = {
            "dataset_id": self.dataset_id,
            "sample_id": sample_record["sample_id"],
            "split": sample_record["split"],
            "task_type": self.task_type,
            "image": image_info,
            "annotations": [annotation],
            "metadata": {
                "npz_path": str(npz_path),
                "npz_keys": npz_info["keys"],
                "arrays": npz_info["arrays"],
                "source_format": "npz",
            },
        }

        validate_common_sample(common_sample)

        return common_sample

    def validate_pairing(self) -> Dict[str, Any]:
        """
        For NPZ segmentation datasets, image and label are inside the same file.
        So this checks internal key availability instead of image-mask filename pairing.
        """
        missing_image_key = []
        missing_label_key = []
        read_errors = []
        shape_mismatches = []

        for sample in self.samples:
            npz_path = sample["npz_path"]

            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    keys = set(data.files)

                    if self.image_key not in keys:
                        missing_image_key.append(str(npz_path))
                        continue

                    if self.label_key not in keys:
                        missing_label_key.append(str(npz_path))
                        continue

                    image = data[self.image_key]
                    label = data[self.label_key]

                    if image.ndim == 3:
                        image_hw = tuple(image.shape[-2:])
                    elif image.ndim == 2:
                        image_hw = tuple(image.shape)
                    else:
                        shape_mismatches.append(
                            {
                                "path": str(npz_path),
                                "reason": "unsupported_image_shape",
                                "image_shape": list(image.shape),
                                "label_shape": list(label.shape),
                            }
                        )
                        continue

                    label_hw = tuple(label.shape[-2:])

                    if image_hw != label_hw:
                        shape_mismatches.append(
                            {
                                "path": str(npz_path),
                                "image_hw": list(image_hw),
                                "label_hw": list(label_hw),
                            }
                        )

            except Exception as exc:
                read_errors.append(
                    {
                        "path": str(npz_path),
                        "error": str(exc),
                    }
                )

        return {
            "dataset_id": self.dataset_id,
            "total_npz_files": len(self.samples),
            "missing_image_key_count": len(missing_image_key),
            "missing_label_key_count": len(missing_label_key),
            "shape_mismatch_count": len(shape_mismatches),
            "read_error_count": len(read_errors),
            "missing_image_key_examples": missing_image_key[:20],
            "missing_label_key_examples": missing_label_key[:20],
            "shape_mismatch_examples": shape_mismatches[:20],
            "read_error_examples": read_errors[:20],
            "valid": (
                not missing_image_key
                and not missing_label_key
                and not shape_mismatches
                and not read_errors
            ),
        }

    def get_statistics(self) -> Dict[str, Any]:
        split_counts = {}

        for sample in self.samples:
            split = sample["split"] or "all"
            split_counts[split] = split_counts.get(split, 0) + 1

        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.config.get("dataset_name"),
            "loader_name": "NPZSegmentationLoader",
            "data_family": self.config.get("data_family"),
            "raw_annotation_type": self.config.get("raw_annotation_type"),
            "task_type": self.config.get("task_type", []),
            "total_samples": len(self.samples),
            "split_counts": split_counts,
            "npz_schema": self.npz_schema,
            "label_schema": self.config.get("label_schema", {}),
        }
