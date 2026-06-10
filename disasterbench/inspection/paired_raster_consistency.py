"""
Reusable paired raster consistency checker.

This module verifies that paired raster files share spatial properties such as
width, height, CRS, and affine transform. It is useful for image-mask datasets.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import rasterio

from disasterbench.inspection.file_pairing_inspector import build_stem_index


@dataclass
class PairedRasterReadError:
    stem: str
    reference_path: Optional[str]
    candidate_path: Optional[str]
    error: str


@dataclass
class PairedRasterConsistencyResult:
    reference_root: str
    candidate_root: str
    reference_extensions: List[str]
    candidate_extensions: List[str]
    strip_suffixes: List[str]
    total_reference_files: int
    total_candidate_files: int
    paired_stem_count: int
    checked_pair_count: int
    scan_limited: bool
    max_pairs: Optional[int]
    missing_candidate_stems: List[str]
    extra_candidate_stems: List[str]
    duplicate_reference_stems: Dict[str, List[str]]
    duplicate_candidate_stems: Dict[str, List[str]]
    shape_mismatch_count: int = 0
    crs_mismatch_count: int = 0
    transform_mismatch_count: int = 0
    invalid_pair_count: int = 0
    example_mismatches: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    read_errors: List[PairedRasterReadError] = field(default_factory=list)

    def has_mismatches(self) -> bool:
        return (
            self.shape_mismatch_count > 0
            or self.crs_mismatch_count > 0
            or self.transform_mismatch_count > 0
            or self.invalid_pair_count > 0
            or bool(self.missing_candidate_stems)
            or bool(self.extra_candidate_stems)
            or bool(self.duplicate_reference_stems)
            or bool(self.duplicate_candidate_stems)
        )

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["read_errors"] = [asdict(error) for error in self.read_errors]
        return data


def normalize_extensions(extensions: Iterable[str]) -> List[str]:
    normalized = []

    for extension in extensions:
        extension = str(extension).lower().strip()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.append(extension)

    return sorted(set(normalized))


def find_duplicates(index: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        stem: paths
        for stem, paths in sorted(index.items())
        if len(paths) > 1
    }


def add_example(
    examples: Dict[str, List[Dict[str, str]]],
    mismatch_type: str,
    item: Dict[str, str],
    limit: int = 10,
) -> None:
    examples.setdefault(mismatch_type, [])

    if len(examples[mismatch_type]) < limit:
        examples[mismatch_type].append(item)


def transforms_match(reference_transform, candidate_transform, tolerance: float) -> bool:
    reference_values = tuple(reference_transform)
    candidate_values = tuple(candidate_transform)

    if len(reference_values) != len(candidate_values):
        return False

    return all(
        abs(float(reference_value) - float(candidate_value)) <= tolerance
        for reference_value, candidate_value in zip(reference_values, candidate_values)
    )


def inspect_paired_raster_consistency(
    reference_root: str | Path,
    candidate_root: str | Path,
    reference_extensions: Iterable[str] = (".tif", ".tiff"),
    candidate_extensions: Iterable[str] = (".tif", ".tiff"),
    strip_suffixes: Optional[Iterable[str]] = None,
    max_pairs: Optional[int] = None,
    transform_tolerance: float = 1e-9,
) -> PairedRasterConsistencyResult:
    """
    Check spatial consistency between paired raster files.

    Pairing is done using normalized filename stems.
    """

    reference_root = Path(reference_root)
    candidate_root = Path(candidate_root)
    strip_suffixes_list = list(strip_suffixes or [])

    reference_extensions_list = normalize_extensions(reference_extensions)
    candidate_extensions_list = normalize_extensions(candidate_extensions)

    reference_index = build_stem_index(
        reference_root,
        extensions=reference_extensions_list,
        strip_suffixes=strip_suffixes_list,
    )
    candidate_index = build_stem_index(
        candidate_root,
        extensions=candidate_extensions_list,
        strip_suffixes=strip_suffixes_list,
    )

    reference_stems = set(reference_index.keys())
    candidate_stems = set(candidate_index.keys())
    paired_stems = sorted(reference_stems & candidate_stems)

    duplicate_reference_stems = find_duplicates(reference_index)
    duplicate_candidate_stems = find_duplicates(candidate_index)

    if max_pairs is not None:
        stems_to_check = paired_stems[:max_pairs]
    else:
        stems_to_check = paired_stems

    checked_pair_count = 0
    shape_mismatch_count = 0
    crs_mismatch_count = 0
    transform_mismatch_count = 0
    invalid_pair_count = 0
    example_mismatches: Dict[str, List[Dict[str, str]]] = {}
    read_errors: List[PairedRasterReadError] = []

    for stem in stems_to_check:
        reference_paths = reference_index.get(stem, [])
        candidate_paths = candidate_index.get(stem, [])

        if len(reference_paths) != 1 or len(candidate_paths) != 1:
            invalid_pair_count += 1
            add_example(
                example_mismatches,
                "ambiguous_duplicate_pair",
                {
                    "stem": stem,
                    "reference_paths": json.dumps(reference_paths),
                    "candidate_paths": json.dumps(candidate_paths),
                },
            )
            continue

        reference_path = reference_paths[0]
        candidate_path = candidate_paths[0]

        try:
            with rasterio.open(reference_path) as reference_src, rasterio.open(candidate_path) as candidate_src:
                checked_pair_count += 1

                reference_shape = f"{reference_src.width}x{reference_src.height}"
                candidate_shape = f"{candidate_src.width}x{candidate_src.height}"

                if reference_shape != candidate_shape:
                    shape_mismatch_count += 1
                    add_example(
                        example_mismatches,
                        "shape_mismatch",
                        {
                            "stem": stem,
                            "reference_path": reference_path,
                            "candidate_path": candidate_path,
                            "reference_shape": reference_shape,
                            "candidate_shape": candidate_shape,
                        },
                    )

                reference_crs = reference_src.crs.to_string() if reference_src.crs else "None"
                candidate_crs = candidate_src.crs.to_string() if candidate_src.crs else "None"

                if reference_crs != candidate_crs:
                    crs_mismatch_count += 1
                    add_example(
                        example_mismatches,
                        "crs_mismatch",
                        {
                            "stem": stem,
                            "reference_path": reference_path,
                            "candidate_path": candidate_path,
                            "reference_crs": reference_crs,
                            "candidate_crs": candidate_crs,
                        },
                    )

                if not transforms_match(
                    reference_src.transform,
                    candidate_src.transform,
                    tolerance=transform_tolerance,
                ):
                    transform_mismatch_count += 1
                    add_example(
                        example_mismatches,
                        "transform_mismatch",
                        {
                            "stem": stem,
                            "reference_path": reference_path,
                            "candidate_path": candidate_path,
                            "reference_transform": str(reference_src.transform),
                            "candidate_transform": str(candidate_src.transform),
                        },
                    )

        except Exception as error:
            invalid_pair_count += 1
            read_errors.append(
                PairedRasterReadError(
                    stem=stem,
                    reference_path=reference_path,
                    candidate_path=candidate_path,
                    error=str(error),
                )
            )
            add_example(
                example_mismatches,
                "read_error",
                {
                    "stem": stem,
                    "reference_path": reference_path,
                    "candidate_path": candidate_path,
                    "error": str(error),
                },
            )

    return PairedRasterConsistencyResult(
        reference_root=str(reference_root),
        candidate_root=str(candidate_root),
        reference_extensions=reference_extensions_list,
        candidate_extensions=candidate_extensions_list,
        strip_suffixes=strip_suffixes_list,
        total_reference_files=sum(len(paths) for paths in reference_index.values()),
        total_candidate_files=sum(len(paths) for paths in candidate_index.values()),
        paired_stem_count=len(paired_stems),
        checked_pair_count=checked_pair_count,
        scan_limited=max_pairs is not None and len(paired_stems) > max_pairs,
        max_pairs=max_pairs,
        missing_candidate_stems=sorted(reference_stems - candidate_stems)[:20],
        extra_candidate_stems=sorted(candidate_stems - reference_stems)[:20],
        duplicate_reference_stems=duplicate_reference_stems,
        duplicate_candidate_stems=duplicate_candidate_stems,
        shape_mismatch_count=shape_mismatch_count,
        crs_mismatch_count=crs_mismatch_count,
        transform_mismatch_count=transform_mismatch_count,
        invalid_pair_count=invalid_pair_count,
        example_mismatches=example_mismatches,
        read_errors=read_errors,
    )


def write_paired_raster_consistency_report(
    result: PairedRasterConsistencyResult,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return output_path
