# xBD Pipeline Status

## Overview

xBD / xView2 is integrated into DisasterBench as a polygon-based building damage assessment dataset. The pipeline supports dataset verification, WKT polygon parsing, common-schema loading, bounding box derivation, GeoJSON export, COCO export, YOLO detection export, YOLO segmentation export, and user-facing CLI export.

## Dataset Role in the Framework

xBD represents the polygon-based disaster damage assessment dataset category in DisasterBench. It validates the framework’s ability to handle object-level building annotations, damage labels, pre/post-disaster imagery, pixel polygons, geographic polygons, and derived detection/segmentation formats.

## Verification Workflow

The xBD integration follows a verification-first workflow before loader or exporter implementation.

Verification includes:

- image, label, and target file counts
- image-label-target pairing
- JSON readability
- annotation object counts
- damage class distribution
- metadata key inspection
- `xy` and `lng_lat` geometry count alignment
- UID/property alignment between `xy` and `lng_lat`
- full conversion audit for polygons, bounding boxes, geographic coordinates, and damage labels

Saved verification outputs are stored under:

    outputs/verification/

## Loader

The xBD loader is implemented in:

    disasterbench/loaders/xbd_loader.py

The loader supports:

- xBD image loading
- label JSON loading
- target mask path tracking
- WKT polygon parsing
- pixel polygon extraction from `features.xy`
- geographic polygon extraction from `features.lng_lat`
- COCO-style bounding box derivation
- damage class preservation
- common internal sample output

Main methods:

- list_samples()
- load_sample()
- validate_pairing()
- get_statistics()

## Converter Logic

xBD annotations are converted from raw WKT polygons into internal framework annotations containing:

- pixel polygon
- pixel bounding box
- geographic polygon
- damage class
- feature type
- UID
- area in pixels

Bounding boxes are mathematically derived from polygon min/max coordinates.

## Export Modules

xBD export support is implemented through:

    disasterbench/exporters/xbd_geojson_exporter.py
    disasterbench/exporters/xbd_coco_exporter.py
    disasterbench/exporters/xbd_yolo_detection_exporter.py
    disasterbench/exporters/xbd_yolo_segmentation_exporter.py

Supported xBD exports:

- GeoJSON
- COCO instance segmentation
- YOLO detection
- YOLO segmentation

## Export Format Interpretation

xBD is naturally suited for polygon, bounding box, and object-level damage assessment formats.

Recommended formats:

- GeoJSON
- COCO segmentation
- YOLO detection
- YOLO segmentation

Mask segmentation is not currently exposed for xBD in the user-facing CLI because the current xBD integration is polygon/object focused.

## Integrity Testing

Exporter integrity tests verify that exported files preserve loader output correctly.

Implemented checks include:

- GeoJSON features match loader `geo_polygon`
- COCO segmentation matches loader pixel polygons
- COCO bbox matches loader bbox
- YOLO detection labels match loader bbox and damage class mapping
- YOLO segmentation labels match loader polygon coordinates and damage class mapping
- user-facing CLI supports xBD export formats

## Visual QA

Visual QA overlays were generated to inspect polygon and bounding box alignment on xBD images. The overlays show building polygons and their derived bounding boxes on top of the original imagery.

## User-Facing Export Command

Example: export xBD to COCO segmentation.

    python -m disasterbench.tools.export_dataset \
      --dataset xbd \
      --format coco_segmentation \
      --max-samples 100

Example: export xBD to GeoJSON.

    python -m disasterbench.tools.export_dataset \
      --dataset xbd \
      --format geojson \
      --max-samples 100

Example: export xBD to YOLO detection.

    python -m disasterbench.tools.export_dataset \
      --dataset xbd \
      --format yolo_detection_bbox \
      --max-samples 100

Example: export xBD to YOLO segmentation.

    python -m disasterbench.tools.export_dataset \
      --dataset xbd \
      --format yolo_segmentation \
      --max-samples 100

## Integration Summary

The xBD pipeline provides an end-to-end example of how polygon-based disaster damage datasets can be integrated into DisasterBench. It includes verification, common-schema loading, polygon-to-bbox conversion, geographic polygon support, multiple export formats, integrity tests, visual QA, and public CLI access.
