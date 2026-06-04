import argparse
import json

from disasterbench.exporters.geojson_exporter import export_sturm_flood_geojson
from disasterbench.exporters.coco_exporter import export_sturm_flood_coco_segmentation
from disasterbench.exporters.yolo_segmentation_exporter import export_sturm_flood_yolo_segmentation
from disasterbench.exporters.yolo_detection_exporter import export_sturm_flood_yolo_detection
from disasterbench.exporters.mask_exporter import export_sturm_flood_masks

from disasterbench.exporters.xbd_geojson_exporter import export_xbd_geojson
from disasterbench.exporters.xbd_coco_exporter import export_xbd_coco
from disasterbench.exporters.xbd_yolo_detection_exporter import export_xbd_yolo_detection
from disasterbench.exporters.xbd_yolo_segmentation_exporter import export_xbd_yolo_segmentation


SUPPORTED_DATASETS = ["sturm_flood", "xbd"]

SUPPORTED_FORMATS = [
    "geojson",
    "coco_segmentation",
    "yolo_segmentation",
    "mask_segmentation",
    "yolo_detection_bbox",
]


def parse_max_samples(value):
    if value is None:
        return None

    if str(value).lower() in ["none", "all", "full"]:
        return None

    return int(value)


def build_default_output(dataset, sensor, export_format, mask_mode):
    if dataset == "xbd":
        if export_format == "geojson":
            return "outputs/geojson/xbd.geojson"

        if export_format == "coco_segmentation":
            return "outputs/coco/xbd_coco.json"

        if export_format == "yolo_segmentation":
            return "outputs/yolo_segmentation/xbd"

        if export_format == "yolo_detection_bbox":
            return "outputs/yolo_detection/xbd"

        if export_format == "mask_segmentation":
            return "outputs/masks/xbd"

    if dataset == "sturm_flood":
        if export_format == "geojson":
            return f"outputs/geojson/{dataset}_{sensor}.geojson"

        if export_format == "coco_segmentation":
            return f"outputs/coco/{dataset}_{sensor}_coco.json"

        if export_format == "yolo_segmentation":
            return f"outputs/yolo_segmentation/{dataset}_{sensor}"

        if export_format == "mask_segmentation":
            return f"outputs/masks/{dataset}_{sensor}_{mask_mode}"

        if export_format == "yolo_detection_bbox":
            return f"outputs/yolo_detection/{dataset}_{sensor}"

    raise ValueError(f"Unsupported output path combination: dataset={dataset}, format={export_format}")


def export_sturm_flood(args):
    output = args.output

    if output is None:
        output = build_default_output(
            dataset=args.dataset,
            sensor=args.sensor,
            export_format=args.format,
            mask_mode=args.mask_mode,
        )

    if args.format == "geojson":
        return export_sturm_flood_geojson(
            sensor=args.sensor,
            output_path=output,
            max_samples=args.max_samples,
            target_values=args.target_values,
            min_area_pixels=args.min_area_pixels,
            coordinate_space="geo",
        )

    if args.format == "coco_segmentation":
        return export_sturm_flood_coco_segmentation(
            sensor=args.sensor,
            output_path=output,
            max_samples=args.max_samples,
            target_values=args.target_values,
            min_area_pixels=args.min_area_pixels,
        )

    if args.format == "yolo_segmentation":
        return export_sturm_flood_yolo_segmentation(
            sensor=args.sensor,
            output_dir=output,
            max_samples=args.max_samples,
            target_values=args.target_values,
            min_area_pixels=args.min_area_pixels,
        )

    if args.format == "mask_segmentation":
        return export_sturm_flood_masks(
            sensor=args.sensor,
            output_dir=output,
            mode=args.mask_mode,
            max_samples=args.max_samples,
        )

    if args.format == "yolo_detection_bbox":
        return export_sturm_flood_yolo_detection(
            sensor=args.sensor,
            output_dir=output,
            max_samples=args.max_samples,
            target_values=args.target_values,
            min_area_pixels=args.min_area_pixels,
        )

    raise ValueError(f"Unsupported format for STURM-Flood: {args.format}")


def export_xbd(args):
    output = args.output

    if output is None:
        output = build_default_output(
            dataset=args.dataset,
            sensor=None,
            export_format=args.format,
            mask_mode=args.mask_mode,
        )

    if args.format == "geojson":
        return export_xbd_geojson(
            output_path=output,
            max_samples=args.max_samples,
            split="train",
        )

    if args.format == "coco_segmentation":
        return export_xbd_coco(
            output_path=output,
            max_samples=args.max_samples,
            split="train",
        )

    if args.format == "yolo_segmentation":
        return export_xbd_yolo_segmentation(
            output_dir=output,
            max_samples=args.max_samples,
            split="train",
        )

    if args.format == "yolo_detection_bbox":
        return export_xbd_yolo_detection(
            output_dir=output,
            max_samples=args.max_samples,
            split="train",
        )

    if args.format == "mask_segmentation":
        raise ValueError(
            "xBD does not support mask_segmentation in this framework. "
            "Use geojson, coco_segmentation, yolo_segmentation, or yolo_detection_bbox."
        )

    raise ValueError(f"Unsupported format for xBD: {args.format}")


def main():
    parser = argparse.ArgumentParser(
        description="Export supported disaster remote sensing datasets into training/GIS formats."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=SUPPORTED_DATASETS,
        help="Dataset ID to export. Supported: sturm_flood, xbd.",
    )

    parser.add_argument(
        "--sensor",
        default="sentinel1",
        choices=["sentinel1", "sentinel2"],
        help="Sensor/source for STURM-Flood. Ignored for xBD.",
    )

    parser.add_argument(
        "--format",
        required=True,
        choices=SUPPORTED_FORMATS,
        help="Export format.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output file or directory. If omitted, a default output path is used.",
    )

    parser.add_argument(
        "--max-samples",
        default=10,
        help="Number of samples to export. Use 'all' or 'none' for full dataset export.",
    )

    parser.add_argument(
        "--target-values",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="STURM-Flood mask values to export as water/flood regions.",
    )

    parser.add_argument(
        "--min-area-pixels",
        type=float,
        default=2.0,
        help="Minimum polygon area in pixels for raster-mask polygonization.",
    )

    parser.add_argument(
        "--mask-mode",
        default="binary_water",
        choices=["binary_water", "original"],
        help="Mask export mode used only for STURM-Flood mask_segmentation format.",
    )

    args = parser.parse_args()
    args.max_samples = parse_max_samples(args.max_samples)

    if args.dataset == "sturm_flood":
        result = export_sturm_flood(args)
    elif args.dataset == "xbd":
        result = export_xbd(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print("\nExport completed.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
