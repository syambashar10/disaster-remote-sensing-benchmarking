import json
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

from disasterbench.inspection.paired_raster_consistency import (
    inspect_paired_raster_consistency,
)


def write_test_tif(path, array, crs="EPSG:4326", transform=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    if transform is None:
        transform = from_origin(0, 0, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(array, 1)


def test_paired_raster_consistency_passes_matching_pairs(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"

    write_test_tif(images / "sample_001.tif", np.ones((2, 2), dtype=np.uint8))
    write_test_tif(masks / "sample_001.tif", np.zeros((2, 2), dtype=np.uint8))

    result = inspect_paired_raster_consistency(images, masks)

    assert result.total_reference_files == 1
    assert result.total_candidate_files == 1
    assert result.paired_stem_count == 1
    assert result.checked_pair_count == 1
    assert result.shape_mismatch_count == 0
    assert result.crs_mismatch_count == 0
    assert result.transform_mismatch_count == 0
    assert result.invalid_pair_count == 0
    assert not result.has_mismatches()


def test_paired_raster_consistency_detects_shape_crs_and_transform_mismatch(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"

    write_test_tif(
        images / "sample_001.tif",
        np.ones((2, 2), dtype=np.uint8),
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    )
    write_test_tif(
        masks / "sample_001.tif",
        np.zeros((3, 2), dtype=np.uint8),
        crs="EPSG:3857",
        transform=from_origin(10, 10, 1, 1),
    )

    result = inspect_paired_raster_consistency(images, masks)

    assert result.shape_mismatch_count == 1
    assert result.crs_mismatch_count == 1
    assert result.transform_mismatch_count == 1
    assert result.has_mismatches()


def test_paired_raster_consistency_reports_missing_and_extra_pairs(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"

    write_test_tif(images / "sample_001.tif", np.ones((2, 2), dtype=np.uint8))
    write_test_tif(images / "sample_002.tif", np.ones((2, 2), dtype=np.uint8))
    write_test_tif(masks / "sample_001.tif", np.zeros((2, 2), dtype=np.uint8))
    write_test_tif(masks / "extra_mask.tif", np.zeros((2, 2), dtype=np.uint8))

    result = inspect_paired_raster_consistency(images, masks)

    assert result.paired_stem_count == 1
    assert result.missing_candidate_stems == ["sample_002"]
    assert result.extra_candidate_stems == ["extra_mask"]
    assert result.has_mismatches()


def test_inspect_paired_rasters_cli_writes_output(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"

    write_test_tif(images / "sample_001.tif", np.ones((2, 2), dtype=np.uint8))
    write_test_tif(masks / "sample_001.tif", np.zeros((2, 2), dtype=np.uint8))

    output_path = tmp_path / "paired_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.inspect_paired_rasters",
            "--reference-root",
            str(images),
            "--candidate-root",
            str(masks),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_data = json.loads(completed.stdout)
    file_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_data["checked_pair_count"] == 1
    assert file_data["checked_pair_count"] == 1
    assert file_data["shape_mismatch_count"] == 0
