import json
import subprocess
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin

from disasterbench.inspection.raster_inspector import inspect_rasters


def write_test_tif(path, array, crs="EPSG:4326", nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=from_origin(0, 0, 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


def test_raster_inspector_reads_basic_raster_properties(tmp_path):
    raster_dir = tmp_path / "rasters"

    write_test_tif(
        raster_dir / "sample_001.tif",
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
        nodata=255,
    )
    write_test_tif(
        raster_dir / "sample_002.tif",
        np.array([[1, 1], [2, 2]], dtype=np.uint8),
        nodata=255,
    )

    result = inspect_rasters(raster_dir)

    assert result.total_raster_files == 2
    assert result.scanned_raster_files == 2
    assert result.valid_raster_files == 2
    assert result.invalid_raster_files == 0
    assert result.shape_counts["2x2"] == 2
    assert result.band_count_counts["1"] == 2
    assert result.dtype_counts["uint8"] == 2
    assert result.crs_counts["EPSG:4326"] == 2
    assert result.nodata_counts["255.0"] == 2


def test_raster_inspector_counts_unique_values_for_masks(tmp_path):
    raster_dir = tmp_path / "masks"

    write_test_tif(
        raster_dir / "mask_001.tif",
        np.array([[0, 1], [1, 2]], dtype=np.uint8),
    )

    result = inspect_rasters(
        raster_dir,
        collect_unique_values=True,
    )

    assert result.unique_values_scanned_files == 1
    assert result.unique_value_counts == {
        "0": 1,
        "1": 2,
        "2": 1,
    }


def test_raster_inspector_supports_unique_value_scan_limit(tmp_path):
    raster_dir = tmp_path / "masks"

    write_test_tif(
        raster_dir / "mask_001.tif",
        np.array([[0, 1]], dtype=np.uint8),
    )
    write_test_tif(
        raster_dir / "mask_002.tif",
        np.array([[2, 3]], dtype=np.uint8),
    )

    result = inspect_rasters(
        raster_dir,
        collect_unique_values=True,
        unique_values_max_files=1,
    )

    assert result.unique_values_scanned_files == 1
    assert result.unique_values_skipped_files == 1
    assert result.unique_value_counts == {
        "0": 1,
        "1": 1,
    }


def test_raster_inspector_reports_invalid_raster(tmp_path):
    raster_dir = tmp_path / "rasters"
    raster_dir.mkdir()

    (raster_dir / "bad.tif").write_text("not a raster", encoding="utf-8")

    result = inspect_rasters(raster_dir)

    assert result.total_raster_files == 1
    assert result.valid_raster_files == 0
    assert result.invalid_raster_files == 1
    assert len(result.read_errors) == 1


def test_inspect_rasters_cli_writes_output(tmp_path):
    raster_dir = tmp_path / "rasters"

    write_test_tif(
        raster_dir / "sample_001.tif",
        np.array([[0, 1], [1, 1]], dtype=np.uint8),
    )

    output_path = tmp_path / "raster_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.inspect_rasters",
            "--raster-root",
            str(raster_dir),
            "--collect-unique-values",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_data = json.loads(completed.stdout)
    file_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_data["valid_raster_files"] == 1
    assert file_data["valid_raster_files"] == 1
    assert file_data["unique_value_counts"] == {
        "0": 1,
        "1": 3,
    }
