import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ALLOWED_STURM_MASK_VALUES = {0, 1, 2, 3, 4, 5, 99}


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def raster_files(folder):
    folder = Path(folder)
    files = []
    for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
        files.extend(folder.glob(ext))
    return sorted(files)


def safe_counter_key(value):
    if value is None:
        return "None"
    return str(value)


def inspect_raster_metadata(path):
    with rasterio.open(path) as src:
        return {
            "width": int(src.width),
            "height": int(src.height),
            "band_count": int(src.count),
            "dtypes": tuple(str(dtype) for dtype in src.dtypes),
            "crs": str(src.crs) if src.crs else None,
            "nodata": src.nodata,
            "bounds": tuple(float(v) for v in src.bounds),
            "transform": tuple(float(v) for v in src.transform),
        }


def get_mask_values(path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    return set(int(v) for v in np.unique(mask))


def transforms_match(transform_a, transform_b, tolerance=1e-6):
    return np.allclose(np.array(transform_a), np.array(transform_b), atol=tolerance)


def bounds_match(bounds_a, bounds_b, tolerance=1e-6):
    return np.allclose(np.array(bounds_a), np.array(bounds_b), atol=tolerance)


def build_samples(metadata_df, image_files, mask_files):
    image_map = {p.stem: p for p in image_files}
    mask_map = {p.stem: p for p in mask_files}

    metadata_df = metadata_df.copy()
    metadata_df["tile_stem"] = metadata_df["tile_id"].astype(str).apply(lambda value: Path(value).stem)

    samples = []

    for _, row in metadata_df.iterrows():
        tile_stem = row["tile_stem"]

        samples.append(
            {
                "sample_id": tile_stem,
                "image_path": image_map.get(tile_stem),
                "mask_path": mask_map.get(tile_stem),
                "metadata": row.drop(labels=["tile_stem"]).to_dict(),
            }
        )

    return samples


def verify_sensor(dataset_root, sensor_name, image_folder, mask_folder, metadata_file, expected_image_bands):
    image_dir = dataset_root / image_folder
    mask_dir = dataset_root / mask_folder
    metadata_path = dataset_root / metadata_file

    image_files = raster_files(image_dir)
    mask_files = raster_files(mask_dir)
    metadata_df = pd.read_csv(metadata_path)

    image_stems = {p.stem for p in image_files}
    mask_stems = {p.stem for p in mask_files}
    metadata_stems = set(metadata_df["tile_id"].astype(str).apply(lambda value: Path(value).stem))

    samples = build_samples(metadata_df, image_files, mask_files)

    summary = {
        "sensor": sensor_name,
        "paths": {
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "metadata_file": str(metadata_path),
        },
        "counts": {
            "image_files": len(image_files),
            "mask_files": len(mask_files),
            "metadata_rows": int(len(metadata_df)),
            "indexed_samples": sum(1 for s in samples if s["image_path"] is not None and s["mask_path"] is not None),
        },
        "pairing": {
            "images_without_masks": len(image_stems - mask_stems),
            "masks_without_images": len(mask_stems - image_stems),
            "images_without_metadata": len(image_stems - metadata_stems),
            "metadata_without_images": len(metadata_stems - image_stems),
        },
        "metadata": {
            "columns": list(metadata_df.columns),
            "shape": [int(metadata_df.shape[0]), int(metadata_df.shape[1])],
            "duplicate_tile_id_count": int(metadata_df["tile_id"].duplicated().sum()),
            "event_type_counts": metadata_df["event_type"].value_counts().head(20).to_dict() if "event_type" in metadata_df.columns else {},
            "country_counts": metadata_df["country"].value_counts().head(20).to_dict() if "country" in metadata_df.columns else {},
            "epsg_code_counts": metadata_df["epsg_code"].value_counts().head(20).to_dict() if "epsg_code" in metadata_df.columns else {},
        },
        "raster_distributions": {
            "image_size_counts": Counter(),
            "mask_size_counts": Counter(),
            "image_band_count_counts": Counter(),
            "mask_band_count_counts": Counter(),
            "image_dtype_counts": Counter(),
            "mask_dtype_counts": Counter(),
            "image_crs_counts": Counter(),
            "mask_crs_counts": Counter(),
            "image_nodata_counts": Counter(),
            "mask_nodata_counts": Counter(),
        },
        "mask_values_global": [],
        "anomaly_count": 0,
    }

    anomalies = []
    mask_values_global = set()

    total = len(samples)

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample["sample_id"]
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        if idx % 1000 == 0 or idx == total:
            print(f"[{sensor_name}] checked {idx}/{total} samples")

        if image_path is None:
            anomalies.append([sensor_name, sample_id, "missing_image_file", "No matching image file found"])
            continue

        if mask_path is None:
            anomalies.append([sensor_name, sample_id, "missing_mask_file", "No matching mask file found"])
            continue

        try:
            image_meta = inspect_raster_metadata(image_path)
        except Exception as e:
            anomalies.append([sensor_name, sample_id, "image_read_error", str(e)])
            continue

        try:
            mask_meta = inspect_raster_metadata(mask_path)
        except Exception as e:
            anomalies.append([sensor_name, sample_id, "mask_read_error", str(e)])
            continue

        dist = summary["raster_distributions"]

        dist["image_size_counts"][f"{image_meta['width']}x{image_meta['height']}"] += 1
        dist["mask_size_counts"][f"{mask_meta['width']}x{mask_meta['height']}"] += 1
        dist["image_band_count_counts"][str(image_meta["band_count"])] += 1
        dist["mask_band_count_counts"][str(mask_meta["band_count"])] += 1
        dist["image_dtype_counts"][str(image_meta["dtypes"])] += 1
        dist["mask_dtype_counts"][str(mask_meta["dtypes"])] += 1
        dist["image_crs_counts"][safe_counter_key(image_meta["crs"])] += 1
        dist["mask_crs_counts"][safe_counter_key(mask_meta["crs"])] += 1
        dist["image_nodata_counts"][safe_counter_key(image_meta["nodata"])] += 1
        dist["mask_nodata_counts"][safe_counter_key(mask_meta["nodata"])] += 1

        if image_meta["width"] != 128 or image_meta["height"] != 128:
            anomalies.append([sensor_name, sample_id, "unexpected_image_size", f"{image_meta['width']}x{image_meta['height']}"])

        if mask_meta["width"] != 128 or mask_meta["height"] != 128:
            anomalies.append([sensor_name, sample_id, "unexpected_mask_size", f"{mask_meta['width']}x{mask_meta['height']}"])

        if image_meta["band_count"] != expected_image_bands:
            anomalies.append([sensor_name, sample_id, "unexpected_image_band_count", str(image_meta["band_count"])])

        if mask_meta["band_count"] != 1:
            anomalies.append([sensor_name, sample_id, "unexpected_mask_band_count", str(mask_meta["band_count"])])

        if image_meta["width"] != mask_meta["width"] or image_meta["height"] != mask_meta["height"]:
            anomalies.append([sensor_name, sample_id, "image_mask_size_mismatch", "image and mask size differ"])

        if image_meta["crs"] != mask_meta["crs"]:
            anomalies.append([sensor_name, sample_id, "image_mask_crs_mismatch", f"{image_meta['crs']} vs {mask_meta['crs']}"])

        if not bounds_match(image_meta["bounds"], mask_meta["bounds"]):
            anomalies.append([sensor_name, sample_id, "image_mask_bounds_mismatch", "image and mask bounds differ"])

        if not transforms_match(image_meta["transform"], mask_meta["transform"]):
            anomalies.append([sensor_name, sample_id, "image_mask_transform_mismatch", "image and mask transform differ"])

        try:
            mask_values = get_mask_values(mask_path)
            mask_values_global.update(mask_values)

            unexpected_values = sorted(mask_values - ALLOWED_STURM_MASK_VALUES)
            if unexpected_values:
                anomalies.append([sensor_name, sample_id, "unexpected_mask_values", str(unexpected_values)])

        except Exception as e:
            anomalies.append([sensor_name, sample_id, "mask_value_read_error", str(e)])

    summary["mask_values_global"] = sorted(int(v) for v in mask_values_global)
    summary["anomaly_count"] = len(anomalies)

    for key, counter in summary["raster_distributions"].items():
        summary["raster_distributions"][key] = dict(counter)

    return summary, anomalies


def main():
    parser = argparse.ArgumentParser(description="Full verification scan for every STURM-Flood image and mask.")
    parser.add_argument("--config", default="configs/sturm_flood_config.json")
    parser.add_argument("--output-json", default="outputs/verification/sturm_flood_full_verification_summary.json")
    parser.add_argument("--output-csv", default="outputs/verification/sturm_flood_full_verification_anomalies.csv")
    args = parser.parse_args()

    config = load_json(args.config)
    dataset_root = Path(config.get("dataset_path", "datasets/sturm_flood_raw"))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    report = {
        "dataset_id": "sturm_flood",
        "verification_type": "full_dataset_scan",
        "dataset_root": str(dataset_root),
        "sensors": {},
    }

    all_anomalies = []

    s1_summary, s1_anomalies = verify_sensor(
        dataset_root=dataset_root,
        sensor_name="sentinel1",
        image_folder="Sentinel1/S1",
        mask_folder="Sentinel1/Floodmaps",
        metadata_file="sentinel1_metadata.csv",
        expected_image_bands=2,
    )

    s2_summary, s2_anomalies = verify_sensor(
        dataset_root=dataset_root,
        sensor_name="sentinel2",
        image_folder="Sentinel2/S2",
        mask_folder="Sentinel2/Floodmaps",
        metadata_file="sentinel2_metadata.csv",
        expected_image_bands=9,
    )

    report["sensors"]["sentinel1"] = s1_summary
    report["sensors"]["sentinel2"] = s2_summary

    all_anomalies.extend(s1_anomalies)
    all_anomalies.extend(s2_anomalies)

    report["total_anomaly_count"] = len(all_anomalies)

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sensor", "sample_id", "issue_type", "details"])
        writer.writerows(all_anomalies)

    print("\nFull STURM-Flood verification complete.")
    print(f"Summary JSON: {output_json}")
    print(f"Anomalies CSV: {output_csv}")
    print(f"Total anomaly count: {len(all_anomalies)}")

    for sensor_name, sensor_report in report["sensors"].items():
        print(f"\n{sensor_name}")
        print(f"  Counts: {sensor_report['counts']}")
        print(f"  Pairing: {sensor_report['pairing']}")
        print(f"  Mask values global: {sensor_report['mask_values_global']}")
        print(f"  Anomaly count: {sensor_report['anomaly_count']}")
        print(f"  Image size counts: {sensor_report['raster_distributions']['image_size_counts']}")
        print(f"  Mask size counts: {sensor_report['raster_distributions']['mask_size_counts']}")
        print(f"  Image band counts: {sensor_report['raster_distributions']['image_band_count_counts']}")
        print(f"  Mask band counts: {sensor_report['raster_distributions']['mask_band_count_counts']}")


if __name__ == "__main__":
    main()
