"""
Reusable raster / GeoTIFF inspector.

This module inspects raster files without assuming a specific dataset.
It checks readability, dimensions, band counts, dtypes, CRS, transforms, nodata,
and optionally unique values for mask-like rasters.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import rasterio


RASTER_EXTENSIONS = {".tif", ".tiff"}


@dataclass
class RasterReadError:
    path: str
    error: str


@dataclass
class RasterInspectionResult:
    raster_root: str
    total_raster_files: int
    scanned_raster_files: int
    valid_raster_files: int
    invalid_raster_files: int
    scan_limited: bool
    max_files: Optional[int]
    collect_unique_values: bool
    unique_values_max_files: Optional[int]
    extension_counts: Dict[str, int] = field(default_factory=dict)
    shape_counts: Dict[str, int] = field(default_factory=dict)
    band_count_counts: Dict[str, int] = field(default_factory=dict)
    dtype_counts: Dict[str, int] = field(default_factory=dict)
    crs_counts: Dict[str, int] = field(default_factory=dict)
    nodata_counts: Dict[str, int] = field(default_factory=dict)
    transform_present_count: int = 0
    unique_value_counts: Dict[str, int] = field(default_factory=dict)
    unique_values_scanned_files: int = 0
    unique_values_skipped_files: int = 0
    example_files: Dict[str, List[str]] = field(default_factory=dict)
    read_errors: List[RasterReadError] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["read_errors"] = [asdict(error) for error in self.read_errors]
        return data


def add_example(
    examples: Dict[str, List[str]],
    signal: str,
    path: Path,
    limit: int = 5,
) -> None:
    examples.setdefault(signal, [])

    if len(examples[signal]) < limit:
        examples[signal].append(str(path))


def normalize_extensions(extensions: Optional[Iterable[str]]) -> set[str]:
    if extensions is None:
        return set(RASTER_EXTENSIONS)

    normalized = set()

    for extension in extensions:
        extension = str(extension).lower().strip()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.add(extension)

    return normalized


def value_to_text(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()

    return str(value)


def inspect_rasters(
    raster_root: str | Path,
    extensions: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    collect_unique_values: bool = False,
    unique_values_max_files: Optional[int] = None,
) -> RasterInspectionResult:
    """
    Inspect raster files under a root folder.

    Args:
        raster_root: Folder containing raster files.
        extensions: File extensions to scan. Defaults to .tif and .tiff.
        max_files: Optional scan limit for faster discovery.
        collect_unique_values: If True, read band 1 and count pixel values.
        unique_values_max_files: Optional limit for unique-value scans.

    Returns:
        RasterInspectionResult.
    """

    root = Path(raster_root)
    allowed_extensions = normalize_extensions(extensions)

    if not root.exists() or not root.is_dir():
        return RasterInspectionResult(
            raster_root=str(root),
            total_raster_files=0,
            scanned_raster_files=0,
            valid_raster_files=0,
            invalid_raster_files=0,
            scan_limited=False,
            max_files=max_files,
            collect_unique_values=collect_unique_values,
            unique_values_max_files=unique_values_max_files,
            read_errors=[
                RasterReadError(
                    path=str(root),
                    error="Raster root does not exist or is not a directory.",
                )
            ],
        )

    raster_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )

    total_raster_files = len(raster_files)

    if max_files is not None:
        raster_files_to_scan = raster_files[:max_files]
    else:
        raster_files_to_scan = raster_files

    extension_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    band_count_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    crs_counts: Counter[str] = Counter()
    nodata_counts: Counter[str] = Counter()
    unique_value_counts: Counter[str] = Counter()
    examples: Dict[str, List[str]] = {}

    valid_raster_files = 0
    invalid_raster_files = 0
    transform_present_count = 0
    unique_values_scanned_files = 0
    unique_values_skipped_files = 0
    read_errors: List[RasterReadError] = []

    for path in raster_files_to_scan:
        extension_counts[path.suffix.lower()] += 1

        try:
            with rasterio.open(path) as src:
                valid_raster_files += 1

                shape_key = f"{src.width}x{src.height}"
                shape_counts[shape_key] += 1
                add_example(examples, f"shape:{shape_key}", path)

                band_count_counts[str(src.count)] += 1

                for dtype in src.dtypes:
                    dtype_counts[str(dtype)] += 1

                crs_key = src.crs.to_string() if src.crs else "None"
                crs_counts[crs_key] += 1

                for nodata in src.nodatavals:
                    nodata_counts[value_to_text(nodata)] += 1

                if src.transform is not None:
                    transform_present_count += 1

                should_collect_unique = collect_unique_values
                if unique_values_max_files is not None:
                    should_collect_unique = (
                        should_collect_unique
                        and unique_values_scanned_files < unique_values_max_files
                    )

                if should_collect_unique:
                    band = src.read(1)
                    values, counts = np.unique(band, return_counts=True)

                    for value, count in zip(values, counts):
                        unique_value_counts[value_to_text(value)] += int(count)

                    unique_values_scanned_files += 1
                elif collect_unique_values:
                    unique_values_skipped_files += 1

        except Exception as error:
            invalid_raster_files += 1
            read_errors.append(
                RasterReadError(
                    path=str(path),
                    error=str(error),
                )
            )
            add_example(examples, "invalid_raster", path)

    return RasterInspectionResult(
        raster_root=str(root),
        total_raster_files=total_raster_files,
        scanned_raster_files=len(raster_files_to_scan),
        valid_raster_files=valid_raster_files,
        invalid_raster_files=invalid_raster_files,
        scan_limited=max_files is not None and total_raster_files > max_files,
        max_files=max_files,
        collect_unique_values=collect_unique_values,
        unique_values_max_files=unique_values_max_files,
        extension_counts=dict(sorted(extension_counts.items())),
        shape_counts=dict(sorted(shape_counts.items())),
        band_count_counts=dict(sorted(band_count_counts.items())),
        dtype_counts=dict(sorted(dtype_counts.items())),
        crs_counts=dict(sorted(crs_counts.items())),
        nodata_counts=dict(sorted(nodata_counts.items())),
        transform_present_count=transform_present_count,
        unique_value_counts=dict(sorted(unique_value_counts.items())),
        unique_values_scanned_files=unique_values_scanned_files,
        unique_values_skipped_files=unique_values_skipped_files,
        example_files=examples,
        read_errors=read_errors,
    )


def write_raster_inspection_report(
    result: RasterInspectionResult,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return output_path
