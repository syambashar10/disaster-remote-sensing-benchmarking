import json
from pathlib import Path


def main():
    path = Path("outputs/verification/sturm_flood_full_verification_summary.json")

    if not path.exists():
        raise FileNotFoundError(
            "Full verification summary not found. Run:\n"
            "python -m disasterbench.tools.full_verify_sturm_flood"
        )

    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    print("\nSTURM-Flood Full Verification Summary")
    print("=" * 50)
    print("Dataset ID:", report["dataset_id"])
    print("Verification type:", report["verification_type"])
    print("Total anomaly count:", report["total_anomaly_count"])

    for sensor, info in report["sensors"].items():
        print("\n" + sensor)
        print("-" * 50)
        print("Counts:", info["counts"])
        print("Pairing:", info["pairing"])
        print("Mask values global:", info["mask_values_global"])
        print("Anomaly count:", info["anomaly_count"])
        print("Image size counts:", info["raster_distributions"]["image_size_counts"])
        print("Mask size counts:", info["raster_distributions"]["mask_size_counts"])
        print("Image band counts:", info["raster_distributions"]["image_band_count_counts"])
        print("Mask band counts:", info["raster_distributions"]["mask_band_count_counts"])
        print("Image dtype counts:", info["raster_distributions"]["image_dtype_counts"])
        print("Mask dtype counts:", info["raster_distributions"]["mask_dtype_counts"])


if __name__ == "__main__":
    main()
