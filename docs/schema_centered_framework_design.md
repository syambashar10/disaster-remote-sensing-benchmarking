# Schema-Centered Disaster Remote Sensing Dataset Framework Design

## 1. Project Goal

The goal of this project is to build a verification-first, schema-centered framework for curating, validating, standardizing, filtering, and exporting heterogeneous disaster remote sensing datasets.

The framework should support current disaster datasets available at QCRI and remain extensible for future datasets with different formats, annotation types, metadata structures, and task definitions.

This project is not only a set of dataset-specific conversion scripts. The intended outcome is a reusable dataset curation framework that can support research, benchmarking, reproducible dataset preparation, and future publication.

## 2. Core Principle

Every dataset must pass through a verification-first workflow before it is converted or exported.

No assumptions should be made silently. Every field, label, annotation type, conversion, and output must be either:

- automatically verified from the dataset files using reproducible scripts, or
- marked as requiring human-in-the-loop review and confirmed before downstream use.

Structural and numeric properties can often be verified automatically. Semantic meaning requires human review unless official dataset documentation clearly proves it.

## 3. High-Level Architecture

The framework follows this architecture:

Raw dataset
→ dataset inspection
→ inspection report
→ draft dataset config
→ human verification gate
→ dataset adapter / loader
→ reusable representation converters
→ common schema
→ schema validation and capability checking
→ filtering and subsetting
→ generic exporters
→ output directory writer
→ metadata catalog
→ dataset versioning

## 4. Main Design Layers

### 4.1 Dataset Inspection Layer

The inspection layer profiles a raw dataset before any transformation occurs.

It checks folder structure, file counts, file extensions, image sizes, image bands, raster metadata, CRS, annotation files, annotation keys, class labels, mask values, metadata columns, image-label pairing, empty annotations, duplicates, and anomalies.

Outputs should include:

- inspection_report.json
- inspection_summary.md
- sample_manifest.csv
- class_inventory.csv
- metadata_inventory.csv
- anomalies.csv
- quality_report.csv when applicable
- duplicate_report.csv when applicable
- visual_previews/ when applicable

### 4.2 Image Quality Assessment Layer

The image quality layer evaluates whether images are usable for downstream machine learning or analysis.

Possible quality checks include cloud cover, blur score, ground sampling distance, missing pixels, invalid raster values, image corruption, and low-quality samples.

Low-quality images should not be silently deleted. They should be recorded in a quarantine log with the reason for quarantine.

Outputs should include:

- quality_report.csv
- quarantine_log.csv
- quality_thresholds.json

Thresholds such as acceptable cloud cover, blur, or GSD require human approval because acceptable quality depends on the task and dataset.

### 4.3 Duplicate Detection Layer

Duplicate detection should happen before train/validation/test split assignment to avoid data leakage.

Possible checks include file hash matching, perceptual hashing, image similarity, matching geospatial footprints, and spatial overlap.

Outputs should include:

- duplicate_report.csv
- duplicate_groups.json
- split_leakage_report.json when splits exist

Ambiguous duplicates require human review.

### 4.4 Dataset Config Layer

Each dataset should have a human-reviewed configuration file. Config files should be created from inspection outputs, then checked by the user or mentor.

A dataset config should contain:

- dataset_id
- display_name
- aliases
- dataset_root
- split definitions
- image folders
- annotation folders
- metadata files
- native annotation types
- task types
- disaster types
- label mappings
- metadata mappings
- quality thresholds
- supported exports
- unsupported exports
- lossy conversions
- human_review_status

Config files should not hide uncertainty. If a mapping or interpretation is not verified, it must be marked as not verified.

### 4.5 Dataset Adapter / Loader Layer

Each raw dataset may need a dataset-specific adapter because datasets differ in structure and format.

The adapter is responsible for reading raw files and producing structured records for the common schema. It should not contain exporter-specific logic.

Examples:

- xBD adapter reads JSON/WKT polygon annotations.
- STURM-Flood adapter reads Sentinel raster images and floodmap masks.
- Future adapters may read COCO JSON, YOLO text, CSV tables, GeoJSON files, or vision-language supervision files.

### 4.6 Raw Label Parsing Layer

The framework should support parsers for common annotation formats found in disaster remote sensing datasets.

Initial parser families:

- GeoJSON parser
- COCO JSON parser
- YOLO text parser
- Raster mask parser
- CSV annotation parser
- Dataset-specific raw JSON parser when required

The parser should preserve original labels, original IDs, source fields, and raw payloads when needed.

### 4.7 Reusable Representation Converter Layer

Converters should be representation-specific, not dataset-specific.

This prevents conversion logic from exploding as more datasets are added.

Examples:

- polygon_to_bbox
- mask_to_binary
- mask_to_polygon
- mask_to_bbox
- bbox_normalizer
- polygon_validator
- geo_to_pixel_transformer
- pixel_to_geo_transformer
- label_mapper
- metadata_normalizer

The goal is:

n dataset adapters + k reusable converters + m generic exporters

not:

n datasets × m exporters

### 4.8 Open Common Schema Layer

The common schema is the central intermediate representation of the framework.

It must support known annotation types while staying open enough to preserve future or unknown annotation types.

Core schema objects should include:

- DatasetCard
- CommonSample
- MediaRecord
- AnnotationRecord
- SupervisionRecord
- LabelMappingRecord
- CapabilityRecord
- ExtensionRecord

A CommonSample may represent a single image, pre/post image pair, raster tile, image-mask pair, multi-band satellite product, or vision-language sample.

Annotation representations may include:

- bbox
- rotated_bbox
- polygon
- multipolygon
- mask
- point
- line
- geo_bbox
- geo_polygon
- raster_label
- graph
- unknown_raw_payload

Supervision records may include:

- classification
- multi-label classification
- caption
- question-answer
- instruction-response
- change description
- report generation

Unknown annotation types should be preserved in extensions/raw_payload and exposed in the catalog. They should not be silently dropped or incorrectly converted.

### 4.9 Label Taxonomy and Mapping Layer

Different datasets may use different labels for the same concept. The framework must preserve both the original label and the mapped canonical label.

Example:

original_label = "no-damage"
canonical_label = "no_damage"

Taxonomies should be stored as versioned YAML files outside the code.

Possible taxonomy files:

- building_damage.yaml
- flood_extent.yaml
- disaster_type.yaml
- land_cover.yaml when needed
- infrastructure_damage.yaml when needed

Mappings should include confidence and review status.

Ambiguous mappings require human review.

### 4.10 Capability Checker

The capability checker determines whether a dataset can be exported to a target format.

Examples:

- YOLO detection requires bounding boxes.
- YOLO segmentation requires polygons.
- COCO detection requires images, categories, and bounding boxes.
- COCO segmentation requires polygons or valid mask-derived segmentations.
- GeoJSON requires geospatial geometry.
- Mask export requires raster or pixel-wise masks.
- Instruction export requires text supervision records.

If a requested export is unsupported, the framework must fail clearly and explain why.

### 4.11 Filter Engine

The filter engine creates a curated subset before export.

Possible filters include:

- disaster type
- geographic bounding box
- task type
- annotation type
- minimum annotation count
- image quality status
- dataset source
- split strategy

The filter engine must produce a filter_manifest.json that records every included and excluded sample with a reason.

### 4.12 Generic Exporter Layer

Exporters should read from the common schema only.

Initial generic exporters:

- generic_coco_exporter.py
- generic_yolo_detection_exporter.py
- generic_yolo_segmentation_exporter.py
- generic_geojson_exporter.py
- generic_mask_exporter.py

Exporters should validate required fields before writing outputs.

### 4.13 Metadata Catalog Layer

Each output dataset should include a metadata catalog.

The catalog should include:

- dataset_card.json
- class_map.yaml
- provenance.json
- license_summary.json
- processing_log.json
- filter_manifest.json
- config_snapshot.json
- statistics_summary.json

The goal is that another researcher can understand how the dataset was created without asking the original developer.

### 4.14 Directory Writer Layer

The directory writer creates a consistent output structure.

A possible structure is:

output_dataset/
  dataset_card.json
  class_map.yaml
  provenance.json
  processing_log.json
  config_snapshot.json
  train/
  val/
  test/

The exact structure may vary by export format, but it must be consistent and documented.

### 4.15 Dataset Versioning Layer

Every generated dataset should be versioned.

Versioning should include:

- content hash
- config snapshot
- taxonomy version
- code version or git commit
- changelog
- DVC-compatible design where possible

This supports reproducibility and future model benchmarking.

## 5. Human-in-the-Loop Policy

The framework must separate automatic verification from human semantic verification.

Automatically verifiable examples:

- number of files
- image dimensions
- raster band count
- CRS presence
- mask unique values
- JSON keys
- image-label pairing
- bounding box bounds
- polygon parse validity
- output file counts

Human review required examples:

- label meanings
- canonical label mappings
- disaster type meaning
- task type interpretation
- metadata semantic meaning
- quality thresholds
- lossy conversion acceptability
- license restrictions
- split strategy
- ambiguous duplicate handling
- unsupported annotation decisions

## 6. Research Direction

The publishable contribution is a verification-first, schema-centered, capability-aware dataset curation framework for heterogeneous disaster remote sensing datasets.

The research value comes from:

- supporting multiple heterogeneous disaster datasets
- preserving original and canonical labels
- representing multiple annotation types in one open schema
- separating dataset adapters from generic exporters
- documenting verification gates
- enabling reproducible filtering and export
- preserving metadata and provenance
- supporting future unknown annotations through extensions
