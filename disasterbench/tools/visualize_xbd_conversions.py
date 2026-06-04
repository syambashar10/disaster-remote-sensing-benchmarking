import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Rectangle
from PIL import Image

from disasterbench.loaders.xbd_loader import XBDLoader


DAMAGE_COLORS = {
    "no_damage_label": "white",
    "no-damage": "lime",
    "minor-damage": "yellow",
    "major-damage": "orange",
    "destroyed": "red",
    "un-classified": "cyan",
}


def visualize_sample(loader, index, output_dir: Path, max_annotations=None):
    sample = loader.load_sample(index)

    image = Image.open(sample["image"]["path"]).convert("RGB")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(
        f"xBD | index={index} | {sample['sample_id']} | annotations={sample['annotation_count']}"
    )
    ax.axis("off")

    annotations = sample["annotations"]

    if max_annotations is not None:
        annotations = annotations[:max_annotations]

    for annotation in annotations:
        damage_class = annotation["damage_class"]
        color = DAMAGE_COLORS.get(damage_class, "white")

        polygon = annotation["polygon"]
        bbox = annotation["bbox"]

        poly_patch = MplPolygon(
            polygon,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=1.2,
        )
        ax.add_patch(poly_patch)

        x, y, w, h = bbox
        bbox_patch = Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor=color,
            linewidth=0.8,
            linestyle="--",
        )
        ax.add_patch(bbox_patch)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"xbd_conversion_qa_index_{index}_{sample['sample_id']}.png"

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        "index": index,
        "sample_id": sample["sample_id"],
        "image_path": sample["image"]["path"],
        "output_path": str(output_path),
        "annotation_count": sample["annotation_count"],
        "disaster_phase": sample["disaster_phase"],
    }


def main():
    parser = argparse.ArgumentParser(description="Visual QA for xBD polygon and bbox conversions.")
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[0, 1, 10, 100, 1000, 5000],
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualizations/xbd_conversion_qa",
    )
    parser.add_argument(
        "--max-annotations",
        default=None,
        type=int,
        help="Optional limit for annotations drawn per image.",
    )

    args = parser.parse_args()

    loader = XBDLoader(split="train")
    output_dir = Path(args.output_dir)

    results = []

    for index in args.indices:
        if index >= len(loader):
            print(f"Skipping index {index}; dataset has {len(loader)} samples.")
            continue

        result = visualize_sample(
            loader=loader,
            index=index,
            output_dir=output_dir,
            max_annotations=args.max_annotations,
        )

        results.append(result)
        print(f"Saved: {result['output_path']}")

    summary_path = output_dir / "visual_qa_summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_id": "xbd",
                "qa_type": "polygon_bbox_overlay",
                "samples_visualized": len(results),
                "results": results,
            },
            f,
            indent=2,
        )

    print("\nxBD visual QA completed.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
