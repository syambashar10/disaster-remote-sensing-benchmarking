# Verification Gate Workflow

This document defines the verification policy for the disaster remote sensing dataset framework.

The purpose is to ensure that dataset inspection, label mapping, conversion, filtering, exporting, and cataloging are done with evidence rather than assumptions.

## 1. Verification Status Types

Every important decision or output should be assigned one of the following statuses.

### AUTO_VERIFIED

The item has been verified automatically using a reproducible script or deterministic check.

Examples:

- file count
- image size
- image band count
- raster dtype
- CRS presence
- mask unique values
- JSON key inventory
- missing image-label pairs
- bbox coordinate bounds
- polygon parse validity

### HUMAN_REVIEW_REQUIRED

The item cannot be safely verified by code alone because it involves semantic meaning, project intent, or scientific judgment.

Examples:

- label meaning
- canonical label mapping
- disaster type interpretation
- task type definition
- metadata semantic meaning
- lossy conversion acceptability
- quality thresholds
- license restrictions
- split strategy
- ambiguous duplicate handling

### NOT_VERIFIED

The item has not yet been checked.

No downstream step should treat this item as correct.

### BLOCKED

The item cannot be completed because required files, documentation, permissions, or human decisions are missing.

## 2. Core Rule

A dataset must not move from inspection to conversion unless all required structural checks are AUTO_VERIFIED and all required semantic checks are either approved by a human or clearly marked as unresolved.

## 3. Dataset Verification Gates

### Gate 1: Raw Dataset Availability

Checks:

- dataset root exists
- expected folders exist
- expected metadata files exist
- files are readable

Evidence:

- inspection_report.json
- folder_inventory.csv

Status:

- Mostly AUTO_VERIFIED

### Gate 2: File and Pairing Integrity

Checks:

- number of images
- number of annotations
- number of masks
- file extensions
- missing images
- missing labels
- missing masks
- duplicate filenames
- empty annotation files

Evidence:

- sample_manifest.csv
- pairing_report.csv
- anomalies.csv

Status:

- AUTO_VERIFIED

### Gate 3: Image and Raster Properties

Checks:

- image width and height
- number of bands
- dtype
- CRS if geospatial
- transform if geospatial
- bounds if geospatial
- corrupt files

Evidence:

- image_inventory.csv
- raster_inventory.csv
- anomalies.csv

Status:

- AUTO_VERIFIED for measured properties
- HUMAN_REVIEW_REQUIRED for deciding whether values are acceptable

### Gate 4: Annotation Format Detection

Checks:

- annotation file type
- JSON keys
- COCO fields
- YOLO line format
- GeoJSON geometry type
- raster mask values
- CSV columns

Evidence:

- annotation_inventory.csv
- parser_report.json
- class_inventory.csv

Status:

- AUTO_VERIFIED for detected structure
- HUMAN_REVIEW_REQUIRED for semantic interpretation

### Gate 5: Label Mapping

Checks:

- raw label inventory
- proposed canonical mapping
- unmapped labels
- ambiguous labels
- mapping confidence

Evidence:

- class_inventory.csv
- class_map.yaml
- label_mapping_review.csv

Status:

- HUMAN_REVIEW_REQUIRED unless official documentation proves the mapping directly

Rule:

No raw label should be silently mapped to a canonical label without evidence or approval.

### Gate 6: Metadata Mapping

Checks:

- metadata field inventory
- proposed common metadata fields
- fields to preserve in extensions
- unknown metadata fields

Evidence:

- metadata_inventory.csv
- metadata_mapping.yaml
- config review notes

Status:

- AUTO_VERIFIED for raw field existence
- HUMAN_REVIEW_REQUIRED for semantic meaning

### Gate 7: Quality Assessment

Checks:

- cloud cover when available
- blur score when applicable
- GSD when available
- invalid pixels
- corrupt images

Evidence:

- quality_report.csv
- quarantine_log.csv
- quality_thresholds.json

Status:

- AUTO_VERIFIED for computed metrics
- HUMAN_REVIEW_REQUIRED for thresholds

### Gate 8: Duplicate Detection

Checks:

- exact file hashes
- perceptual hashes
- near duplicates
- geospatial overlap where available
- split leakage

Evidence:

- duplicate_report.csv
- duplicate_groups.json
- split_leakage_report.json

Status:

- AUTO_VERIFIED for exact duplicates
- HUMAN_REVIEW_REQUIRED for ambiguous near duplicates

### Gate 9: Schema Conversion

Checks:

- required CommonSample fields
- media records valid
- annotation records valid
- original labels preserved
- canonical labels mapped
- metadata preserved
- raw payload preserved when needed

Evidence:

- schema_validation_report.json
- converted_sample_preview.json
- failed_samples.csv

Status:

- AUTO_VERIFIED for schema compliance
- HUMAN_REVIEW_REQUIRED for uncertain semantic mappings

### Gate 10: Representation Conversion

Checks:

- polygon to bbox correctness
- mask to polygon correctness
- mask to bbox correctness
- coordinate normalization
- CRS reprojection
- chipping/tiling correctness

Evidence:

- conversion_audit.json
- visual_qa_samples/
- failed_conversions.csv

Status:

- AUTO_VERIFIED for numeric checks
- HUMAN_REVIEW_REQUIRED for lossy conversion acceptability and visual alignment approval

### Gate 11: Filtering and Split Strategy

Checks:

- filter parameters
- included samples
- excluded samples
- exclusion reasons
- train/val/test ratio
- disaster distribution by split
- class distribution by split

Evidence:

- filter_manifest.json
- split_summary.json
- statistics_summary.json

Status:

- AUTO_VERIFIED for applying filter rules
- HUMAN_REVIEW_REQUIRED for deciding the split strategy and whether the filtered dataset is scientifically appropriate

### Gate 12: Export Validation

Checks:

- target format requirements
- output file counts
- labels preserved
- coordinates within image bounds
- class IDs valid
- missing output files
- format compliance

Evidence:

- export_validation_report.json
- output_manifest.json
- exporter_log.json

Status:

- AUTO_VERIFIED

### Gate 13: Visual QA

Checks:

- bbox overlay alignment
- polygon overlay alignment
- mask overlay alignment
- random samples
- edge cases
- empty annotations

Evidence:

- visual_qa_summary.json
- visual_qa_images/

Status:

- HUMAN_REVIEW_REQUIRED for final approval of visual correctness

### Gate 14: Metadata Catalog and Versioning

Checks:

- dataset_card.json exists
- class_map.yaml exists
- provenance.json exists
- license_summary.json exists
- config_snapshot.json exists
- content hash computed
- processing log saved

Evidence:

- dataset_card.json
- class_map.yaml
- provenance.json
- license_summary.json
- config_snapshot.json
- version_hash.txt
- processing_log.json

Status:

- AUTO_VERIFIED for file existence and consistency
- HUMAN_REVIEW_REQUIRED for license interpretation

## 4. Hard Stop Conditions

The pipeline must stop or require explicit approval when:

- labels are unmapped
- label meanings are ambiguous
- task type is unclear
- metadata meaning is unclear
- requested export is unsupported
- conversion is lossy and not approved
- CRS or coordinate transform is missing for geospatial conversion
- license is unknown
- duplicate handling is ambiguous
- validation fails

## 5. Evidence Standard

A claim is considered verified only if it is supported by saved evidence.

Examples:

Correct:

- "The dataset has 21,602 Sentinel-1 samples according to inspection_report.json."
- "Mask values [0, 1, 2, 3, 4, 5] were found by a full mask scan."
- "The mapping from no-damage to no_damage was approved in class_map.yaml."

Not acceptable:

- "It looks like this means flood."
- "Probably this class is background."
- "The exporter should work."
- "The split is probably fine."

## 6. Final Principle

Automated tools prove structure, counts, geometry, and format compliance.

Human review proves meaning, intent, thresholds, and scientific validity.
