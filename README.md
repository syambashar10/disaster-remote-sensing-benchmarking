# Disaster Remote Sensing Dataset Benchmarking Framework

This repository is an early-stage research and engineering framework for organizing, inspecting, and standardizing disaster remote sensing datasets.

The long-term goal is to support multiple disaster datasets with different annotation formats, such as polygons, bounding boxes, GeoJSON files, raster masks, and segmentation labels, through a common metadata and dataloader structure.

## Project Motivation

Public disaster remote sensing datasets are often difficult to compare because each dataset has its own folder structure, annotation format, label definitions, metadata fields, and file types.

For example:

- xBD/xView2 uses satellite images with building polygon annotations and damage labels.
- STURM-Flood uses Sentinel-1 and Sentinel-2 GeoTIFF image tiles with raster floodmap masks.

This project aims to create a practical framework where each dataset keeps its original format, but dataset-specific loaders and metadata configs expose the data in a consistent structure for analysis, visualization, and future benchmarking.

## Current Status

The current version focuses on onboarding and inspecting the STURM-Flood dataset as a raster-mask semantic segmentation dataset.

Completed so far:

- Created the initial project structure.
- Linked local raw datasets without copying them into the repository.
- Added a STURM-Flood dataset configuration file.
- Inspected Sentinel-1 and Sentinel-2 folder structures.
- Verified image, floodmap, and metadata pairing.
- Inspected GeoTIFF raster properties.
- Visualized Sentinel-1 and Sentinel-2 samples with floodmap overlays.
- Saved inspection outputs and dataset notes.

## Repository Structure

```text
qcri_disaster_benchmarking/
├── configs/
│   └── sturm_flood_config.json
├── docs/
│   └── sturm_flood_dataset_notes.md
├── experiments/
│   └── inspect_sturm_flood.ipynb
├── outputs/
│   └── sturm_flood_inspection/
│       ├── sentinel1_sample_0_inspection.png
│       ├── sentinel2_sample_0_inspection.png
│       └── sturm_flood_inspection_summary.json
├── schemas/
├── loaders/
├── requirements.txt
├── .gitignore
└── README.md
Dataset Note

Raw datasets are not included in this repository because of their size.

Local dataset links are expected under:

datasets/
├── sturm_flood_raw
└── xbd_raw

These are symbolic links to local dataset folders and are ignored by Git.

STURM-Flood Inspection Summary

STURM-Flood is used as the second dataset in this framework because it is structurally different from xBD/xView2.

STURM-Flood contains:

Sentinel-1 image tiles and matching floodmaps
Sentinel-2 image tiles and matching floodmaps
Metadata CSV files for both Sentinel-1 and Sentinel-2
GeoTIFF raster data
Pixel-level floodmap masks

Inspection results:

Sentinel-1 images: 21,602
Sentinel-1 floodmaps: 21,602
Sentinel-1 metadata rows: 21,602

Sentinel-2 images: 2,675
Sentinel-2 floodmaps: 2,675
Sentinel-2 metadata rows: 2,675

The pairing check confirmed that every satellite image has a matching floodmap and metadata row.

Annotation Type

STURM-Flood uses raster mask annotations.

The original floodmap values are represented as integer pixel classes:

0  = coastline / area of interest
1  = flooded area
2  = river-related features
3  = open water
4  = reservoirs
5  = lakes
99 = no data

This makes STURM-Flood useful for testing raster-mask support in the framework.

Framework Direction

The planned framework design is:

raw dataset
    ↓
dataset-specific config
    ↓
dataset-specific loader
    ↓
common sample schema
    ↓
analysis, visualization, metadata catalog, and benchmarking

This avoids forcing all datasets into one raw format. Instead, each dataset loader handles its own structure while returning a shared representation.

Next Steps

Planned next steps:

Define the common sample schema.
Implement a base dataset loader interface.
Implement STURMFloodLoader.
Refactor previous xBD/xView2 work into XBDLoader.
Create a dataset catalog comparing xBD and STURM-Flood.
Add more datasets with different annotation formats.
Build comparative statistics and visualizations for a publishable framework.
Research Goal

The broader goal is to develop a reproducible disaster remote sensing dataset benchmarking and standardization framework that can support multiple datasets, annotation types, and metadata formats.

This project is being developed with the aim of producing both a useful engineering toolkit and a publishable research contribution.

