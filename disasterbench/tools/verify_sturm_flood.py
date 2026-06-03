import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_raster_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []

    files = []
    for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        files.extend(folder.glob(ext))

    return sorted(files)


def inspect_raster(path):
    path = Path(path)

    with rasterio.open(path) as src:
        data = src.read()

        return {
            "path": str(path),
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
            "min_value": float(np.nanmin(data)),
            "max_value": float(np.nanmax(data)),
        }


def collect_mask_values(mask_files, sample_limit):
    values = set()

    for mask_path in mask_files[:sample_limit]:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            unique_values = np.unique(mask)

        for value in unique_values:
            values.add(int(value))

    return sorted(values)


def check_pairing(image_files, mask_files, metadata_df):
    image_stems = {p.stem for p in image_files}
    mask_stems = {p.stem for p in mask_files}

    if "tile_id" not in metadata_df.columns:
        metadata_tile_ids = set()
        metadata_column_found = False
        metadata_tile_id_format = "missing"
    else:
        raw_tile_ids = metadata_df["tile_id"].astype(str)

        # Metadata tile_id values include filenames such as:
        # EMSR260_02VIADANA_2_1_2_2.tif
        # Image and mask Path.stem values remove the .tif extension.
        # Therefore we normalize metadata tile_id values using Path(...).stem.
        metadata_tile_ids = {Path(tile_id).stem for tile_id in raw_tile_ids}
        metadata_column_found = True

        if raw_tile_ids.iloc[0].lower().endswith((".tif", ".tiff")):
            metadata_tile_id_format = "filename_with_extension"
        else:
            metadata_tile_id_format = "stem_without_extension"

    return {
        "metadata_tile_id_column_found": metadata_column_found,
        "metadata_tile_id_format": metadata_tile_id_format,
        "images_without_masks": sorted(list(image_stems - mask_stems))[:20],
        "masks_without_images": sorted(list(mask_stems - image_stems))[:20],
        "images_without_metadata": sorted(list(image_stems - metadata_tile_ids))[:20],
        "metadata_without_images": sorted(list(metadata_tile_ids - image_stems))[:20],
        "images_without_masks_count": len(image_stems - mask_stems),
        "masks_without_images_count": len(mask_stems - image_stems),
        "images_without_metadata_count": len(image_stems - metadata_tile_ids),
        "metadata_without_images_count": len(metadata_tile_ids - image_stems),
    }


def verify_sensor(dataset_root, sensor_name, image_folder, mask_folder, metadata_file, sample_limit):
    image_dir = dataset_root / image_folder
    mask_dir = dataset_root / mask_folder
    metadata_path = dataset_root / metadata_file

    result = {
        "sensor": sensor_name,
        "paths": {
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "metadata_file": str(metadata_path),
        },
        "exists": {
            "image_dir": image_dir.exists(),
            "mask_dir": mask_dir.exists(),
            "metadata_file": metadata_path.exists(),
        },
    }

    if not image_dir.exists() or not mask_dir.exists() or not metadata_path.exists():
        result["error"] = "One or more required paths do not exist."
        return result

    image_files = count_raster_files(image_dir)
    mask_files = count_raster_files(mask_dir)
    metadata_df = pd.read_csv(metadata_path)

    result["counts"] = {
        "image_files": len(image_files),
        "mask_files": len(mask_files),
        "metadata_rows": int(len(metadata_df)),
    }

    result["metadata"] = {
        "columns": list(metadata_df.columns),
        "shape": [int(metadata_df.shape[0]), int(metadata_df.shape[1])],
        "head": metadata_df.head(3).fillna("").to_dict(orient="records"),
    }

    result["pairing_check"] = check_pairing(image_files, mask_files, metadata_df)

    result["sample_rasters"] = {
        "image_sample": inspect_raster(image_files[0]) if image_files else None,
        "mask_sample": inspect_raster(mask_files[0]) if mask_files else None,
    }

    result["mask_values_in_first_samples"] = collect_mask_values(mask_files, sample_limit)

    result["example_files"] = {
        "first_5_images": [p.name for p in image_files[:5]],
        "first_5_masks": [p.name for p in mask_files[:5]],
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify the real STURM-Flood dataset structure and contents.")
    parser.add_argument("--config", default="configs/sturm_flood_config.json")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--output", default="outputs/verification/sturm_flood_verification_report.json")
    args = parser.parse_args()

    config = load_json(args.config)

    dataset_root = Path(config.get("dataset_path", "datasets/sturm_flood_raw"))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    report = {
        "dataset_id": "sturm_flood",
        "verification_rule": "real files inspected before loader implementation",
        "dataset_root": str(dataset_root),
        "sample_limit_for_mask_value_scan": args.sample_limit,
        "sensors": {},
    }

    report["sensors"]["sentinel1"] = verify_sensor(
        dataset_root=dataset_root,
        sensor_name="sentinel1",
        image_folder="Sentinel1/S1",
        mask_folder="Sentinel1/Floodmaps",
        metadata_file="sentinel1_metadata.csv",
        sample_limit=args.sample_limit,
    )

    report["sensors"]["sentinel2"] = verify_sensor(
        dataset_root=dataset_root,
        sensor_name="sentinel2",
        image_folder="Sentinel2/S2",
        mask_folder="Sentinel2/Floodmaps",
        metadata_file="sentinel2_metadata.csv",
        sample_limit=args.sample_limit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSTURM-Flood verification completed.")
    print(f"Report saved to: {output_path}")

    for sensor_name, sensor_report in report["sensors"].items():
        print(f"\nSensor: {sensor_name}")

        if "error" in sensor_report:
            print(f"  ERROR: {sensor_report['error']}")
            continue

        counts = sensor_report["counts"]
        pairing = sensor_report["pairing_check"]

        print(f"  Image files: {counts['image_files']}")
        print(f"  Mask files: {counts['mask_files']}")
        print(f"  Metadata rows: {counts['metadata_rows']}")

        print("  Pairing issues:")
        print(f"    Images without masks: {pairing['images_without_masks_count']}")
        print(f"    Masks without images: {pairing['masks_without_images_count']}")
        print(f"    Images without metadata: {pairing['images_without_metadata_count']}")
        print(f"    Metadata without images: {pairing['metadata_without_images_count']}")

        image_sample = sensor_report["sample_rasters"]["image_sample"]
        mask_sample = sensor_report["sample_rasters"]["mask_sample"]

        print("  Image raster sample:")
        print(f"    Size: {image_sample['width']} x {image_sample['height']}")
        print(f"    Bands: {image_sample['band_count']}")
        print(f"    Dtypes: {image_sample['dtypes']}")
        print(f"    CRS: {image_sample['crs']}")

        print("  Mask raster sample:")
        print(f"    Size: {mask_sample['width']} x {mask_sample['height']}")
        print(f"    Bands: {mask_sample['band_count']}")
        print(f"    Dtypes: {mask_sample['dtypes']}")
        print(f"    CRS: {mask_sample['crs']}")

        print(f"  Mask values found in first {args.sample_limit} samples:")
        print(f"    {sensor_report['mask_values_in_first_samples']}")

    print("\nNo loader should be implemented until this verification output is reviewed.\n")


if __name__ == "__main__":
    main()
