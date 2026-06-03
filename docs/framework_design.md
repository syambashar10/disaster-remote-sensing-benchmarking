# DisasterBench Framework Design

## Goal

DisasterBench is a capability-aware framework for disaster remote sensing datasets.

The framework is designed to:

1. Load different disaster remote sensing datasets using dataset-specific loaders.
2. Inspect and verify each dataset before implementation.
3. Convert raw dataset samples into a common internal schema.
4. Recommend appropriate export formats based on each dataset's annotation type.
5. Export datasets into training-ready formats such as COCO, YOLO detection, YOLO segmentation, raster masks, GeoJSON, and statistics.

## Core Rule: Verification First

No loader, converter, or exporter should be written based on assumptions.

Before implementing support for any dataset, the following must be verified from the real dataset files:

- folder structure
- file counts
- image/raster formats
- image dimensions
- number of bands
- data types
- CRS/geospatial metadata if available
- annotation format
- class values
- metadata fields
- image-label pairing consistency
- empty or missing annotations
- sample visualization
- edge cases

If something is unknown, it must be marked as unknown instead of guessed.

## Main Pipeline

Raw dataset
→ Dataset-specific loader
→ Common internal sample schema
→ Capability registry
→ Converter
→ Exporter
→ COCO / YOLO / masks / GeoJSON / statistics

## Dataset Support Strategy

Each dataset must have:

- config file
- dataset notes
- inspection notebook or script
- loader
- common schema mapping
- supported format list
- recommended format list
- lossy format list
- statistics output
- visualization output

## Initial Datasets

### Dataset 1: xBD / xView2

Raw format:
- PNG satellite images
- JSON labels
- WKT building polygons
- building damage classes

Natural tasks:
- building localization
- object detection
- instance segmentation
- building damage assessment
- change/damage analysis

Recommended exports:
- COCO detection
- COCO instance segmentation
- YOLO detection
- YOLO segmentation
- GeoJSON

### Dataset 2: STURM-Flood

Raw format:
- Sentinel-1 GeoTIFF images
- Sentinel-2 GeoTIFF images
- GeoTIFF floodmap masks
- metadata CSV files

Natural tasks:
- semantic segmentation
- flood extent mapping
- binary water segmentation
- multi-class water segmentation

Recommended exports:
- raster masks
- COCO segmentation
- YOLO segmentation
- GeoJSON polygons

Lossy but possible exports:
- YOLO detection bounding boxes

Reason:
Flood regions are naturally pixel-level regions. Bounding boxes can be derived from masks, but they do not preserve the full flood shape.
