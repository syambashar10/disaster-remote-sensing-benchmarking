# DisasterBench: Disaster Remote Sensing Dataset Framework

DisasterBench is a reusable framework for inspecting, verifying, loading, converting, and exporting disaster remote sensing datasets with different annotation formats.

The framework follows a verification-first workflow. Each dataset is inspected from its real files before loaders, converters, or exporters are implemented. Dataset-specific loaders read the original dataset format and return samples using a common internal schema. Reusable converters and exporters then transform the data into training-ready or GIS-ready formats.

## Framework Pipeline

Raw dataset
→ Dataset-specific loader
→ Common internal schema
→ Reusable converters
→ Exporters
→ Training-ready / GIS-ready outputs

## Current Dataset Integration

The current implementation supports two dataset types: STURM-Flood, a raster-mask flood extent mapping dataset, and xBD / xView2, a polygon-based building damage assessment dataset.

The STURM-Flood pipeline supports dataset verification, image-mask-metadata pairing checks, common schema loading, raster mask conversion, visual conversion quality checks, GeoJSON export, COCO segmentation export, YOLO segmentation export, YOLO detection bounding box export, original and binary mask export, and a user-facing export command.


## Supported Datasets

- STURM-Flood: raster-mask flood extent mapping with Sentinel-1 and Sentinel-2 imagery
- xBD / xView2: polygon-based building damage assessment with pre/post-disaster imagery

## Supported STURM-Flood Sensors

- sentinel1
- sentinel2

## Supported Export Formats

- geojson
- coco_segmentation
- yolo_segmentation
- mask_segmentation
- yolo_detection_bbox

## Format Suitability

STURM-Flood is originally a raster-mask semantic segmentation dataset. The most natural formats are mask-based and segmentation-based outputs.

Recommended formats include mask_segmentation, geojson, coco_segmentation, and yolo_segmentation.

Bounding box export is also supported as a derived representation. Since flood regions are irregular areas, bounding boxes approximate the original mask shape and should be used only when detection-style experiments are required.

## User-Facing Export Command

Export STURM-Flood Sentinel-1 samples to GeoJSON:

    python -m disasterbench.tools.export_dataset \
      --dataset sturm_flood \
      --sensor sentinel1 \
      --format geojson \
      --max-samples 100

Export all Sentinel-2 samples to COCO segmentation format:

    python -m disasterbench.tools.export_dataset \
      --dataset sturm_flood \
      --sensor sentinel2 \
      --format coco_segmentation \
      --max-samples all

Export binary water masks:

    python -m disasterbench.tools.export_dataset \
      --dataset sturm_flood \
      --sensor sentinel1 \
      --format mask_segmentation \
      --mask-mode binary_water \
      --max-samples 100


## xBD Export Examples

Export xBD to COCO segmentation:

    python -m disasterbench.tools.export_dataset \
      --dataset xbd \
      --format coco_segmentation \
      --max-samples 100

Export xBD to YOLO detection:

    python -m disasterbench.tools.export_dataset \
      --dataset xbd \
      --format yolo_detection_bbox \
      --max-samples 100

## Verification

The STURM-Flood integration was verified using real dataset files. The verification workflow checks file counts, metadata structure, image-mask-metadata pairing, raster dimensions, band counts, data types, CRS information, bounds and transform alignment, mask values, and conversion validity.

Verification outputs are saved under:

    outputs/verification/

## Repository Structure

    disasterbench/
    ├── converters/
    ├── exporters/
    ├── loaders/
    ├── schemas/
    ├── tools/
    └── utils/

    configs/
    datasets/
    docs/
    experiments/
    outputs/
    tests/

## Documentation

Detailed dataset pipeline documentation is available at:

    docs/sturm_flood_pipeline_status.md
    docs/xbd_pipeline_status.md

## Extension Pattern

The framework is designed to support additional disaster remote sensing datasets through the same pattern: dataset configuration, dataset verification, dataset-specific loading, common schema mapping, reusable conversion, reusable export, testing, and documentation.

This allows polygon-based, bounding-box-based, raster-mask-based, and metadata-rich datasets to be integrated into one consistent benchmarking and export framework.
