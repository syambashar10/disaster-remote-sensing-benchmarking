import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle

from disasterbench.converters.mask_to_binary import mask_file_to_binary
from disasterbench.converters.mask_to_polygon import mask_file_to_polygons
from disasterbench.converters.mask_to_bbox import mask_file_to_bboxes
from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


def read_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)


def add_polygon_overlay(ax, polygons):
    for polygon_record in polygons:
        polygon = polygon_record["polygon"]

        if len(polygon) >= 3:
            patch = MplPolygon(
                polygon,
                fill=False,
                linewidth=1.0,
            )
            ax.add_patch(patch)

        for hole in polygon_record.get("holes", []):
            if len(hole) >= 3:
                hole_patch = MplPolygon(
                    hole,
                    fill=False,
                    linewidth=0.8,
                    linestyle="--",
                )
                ax.add_patch(hole_patch)


def add_bbox_overlay(ax, bboxes):
    for bbox_record in bboxes:
        x, y, w, h = bbox_record["bbox"]

        rect = Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            linewidth=1.0,
        )
        ax.add_patch(rect)


def visualize_sample(sensor, index, output_dir):
    loader = STURMFloodLoader(sensor=sensor)
    sample = loader.load_sample(index)

    mask_path = sample["annotations"][0]["mask_path"]
    mask = read_mask(mask_path)

    binary = mask_file_to_binary(mask_path)

    polygons = mask_file_to_polygons(
        mask_path=mask_path,
        target_values=[1, 2, 3, 4, 5],
        min_area_pixels=2.0,
        coordinate_space="pixel",
    )

    bboxes = mask_file_to_bboxes(
        mask_path=mask_path,
        target_values=[1, 2, 3, 4, 5],
        min_area_pixels=2.0,
    )

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(mask, interpolation="nearest")
    axes[0].set_title("Original mask")

    axes[1].imshow(binary, interpolation="nearest")
    axes[1].set_title("Binary water mask")

    axes[2].imshow(binary, interpolation="nearest")
    add_polygon_overlay(axes[2], polygons)
    axes[2].set_title(f"Polygons: {len(polygons)}")

    axes[3].imshow(binary, interpolation="nearest")
    add_bbox_overlay(axes[3], bboxes)
    axes[3].set_title(f"BBoxes: {len(bboxes)}")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{sensor} | index={index} | sample_id={sample['sample_id']}")
    plt.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{sensor}_sample_{index}_conversion_qa.png"
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

    return {
        "sensor": sensor,
        "index": index,
        "sample_id": sample["sample_id"],
        "mask_path": mask_path,
        "polygon_count": len(polygons),
        "bbox_count": len(bboxes),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Visual QA for STURM-Flood mask conversions.")
    parser.add_argument("--output-dir", default="outputs/visualizations/sturm_conversion_qa")
    args = parser.parse_args()

    checks = []

    sample_plan = {
        "sentinel1": [0, 10, 100, 1000, 5000],
        "sentinel2": [0, 10, 100, 500, 1000],
    }

    for sensor, indices in sample_plan.items():
        loader = STURMFloodLoader(sensor=sensor)

        for index in indices:
            if index >= len(loader.samples):
                continue

            result = visualize_sample(
                sensor=sensor,
                index=index,
                output_dir=args.output_dir,
            )
            checks.append(result)
            print(result)

    print("\nVisual QA images saved to:", args.output_dir)


if __name__ == "__main__":
    main()
