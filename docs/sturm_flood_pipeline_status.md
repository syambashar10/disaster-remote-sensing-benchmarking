# STURM-Flood Pipeline Status

## Overview

STURM-Flood is integrated into DisasterBench as a raster-mask semantic segmentation dataset for flood extent and water-region mapping. The pipeline supports Sentinel-1 and Sentinel-2 data through a verification-first workflow, a dataset-specific loader, reusable mask converters, multiple export formats, and a user-facing export command.

## Dataset Role in the Framework

STURM-Flood represents the raster-mask dataset category in DisasterBench. It validates the framework's ability to handle GeoTIFF imagery, pixel-level semantic masks, metadata tables, geospatial information, and derived vector outputs.

## Loader

The STURM-Flood loader is implemented in:

    disasterbench/loaders/sturm_flood_loader.py

The loader supports Sentinel-1 and Sentinel-2 sample loading, image-mask-metadata pairing, common schema output, dataset statistics, and pairing validation.

Main methods:

- list_samples()
- load_sample()
- validate_pairing()
- get_statistics()

## Verification Workflow

Verification scripts are stored in:

    disasterbench/tools/verify_sturm_flood.py
    disasterbench/tools/full_verify_sturm_flood.py
    disasterbench/tools/summarize_sturm_verification.py

Saved verification outputs are stored in:

    outputs/verification/sturm_flood_verification_report.json
    outputs/verification/sturm_flood_full_verification_summary.json
    outputs/verification/sturm_flood_full_verification_anomalies.csv

The verification workflow checks dataset paths, image counts, mask counts, metadata rows, image-mask-metadata pairing, raster dimensions, band counts, data types, CRS information, bounds and transform alignment, mask values, and anomaly records.

## Converter Modules

Converter modules are stored in:

    disasterbench/converters/mask_to_binary.py
    disasterbench/converters/mask_to_polygon.py
    disasterbench/converters/mask_to_bbox.py

Supported conversions include multiclass raster mask to binary water mask, raster mask to polygon features, and raster mask to bounding boxes.

The polygon converter supports Polygon, MultiPolygon, GeometryCollection, and polygon holes/interior rings.

## Export Modules

Exporter modules are stored in:

    disasterbench/exporters/geojson_exporter.py
    disasterbench/exporters/coco_exporter.py
    disasterbench/exporters/yolo_segmentation_exporter.py
    disasterbench/exporters/yolo_detection_exporter.py
    disasterbench/exporters/mask_exporter.py

Supported exports include GeoJSON, COCO segmentation, YOLO segmentation, YOLO detection bounding boxes, original raster masks, and binary water masks.

## Export Format Interpretation

STURM-Flood is naturally suited for segmentation and mask-based workflows. Mask export preserves the closest representation to the original dataset. GeoJSON, COCO segmentation, and YOLO segmentation are derived from raster mask polygons. YOLO detection bounding boxes are supported as an approximate representation for detection-style experiments.

## User-Facing Export Command

The user-facing export command is implemented in:

    disasterbench/tools/export_dataset.py

Example usage:

    python -m disasterbench.tools.export_dataset \
      --dataset sturm_flood \
      --sensor sentinel1 \
      --format geojson \
      --max-samples 100

Full export can be requested using:

    --max-samples all

## Test Coverage

Test files are stored in:

    tests/test_sturm_flood_loader.py
    tests/test_sturm_flood_converters.py
    tests/test_sturm_flood_geojson_exporter.py
    tests/test_sturm_flood_coco_exporter.py
    tests/test_sturm_flood_yolo_segmentation_exporter.py
    tests/test_sturm_flood_yolo_detection_exporter.py
    tests/test_sturm_flood_mask_exporter.py
    tests/test_export_dataset_cli.py

The tests cover loader behavior, conversion validity, exporter structure, mask output validity, YOLO label formatting, COCO JSON structure, GeoJSON structure, and the user-facing export CLI.

## Integration Summary

The STURM-Flood pipeline provides an end-to-end example of how raster-mask disaster datasets can be integrated into DisasterBench. It includes dataset verification, common-schema loading, mask conversion, export generation, visual quality checks, and reusable user-facing commands.
