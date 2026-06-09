# v0.3 Schema-Centered Implementation Checklist

This checklist defines the implementation plan for the v0.3 schema-centered refactor.

The current v0.2 work is treated as a validated prototype for STURM-Flood and xBD. The v0.3 goal is to refactor the project into a scalable, verification-first, schema-centered framework.

## Phase 0: Branch and Planning

- [x] Create branch refactor/schema-centered-v0.3
- [x] Create schema-centered framework design document
- [x] Create v0.3 implementation checklist
- [x] Create verification gate workflow document

## Phase 1: Project Structure

- [ ] Create disasterbench/inspection/
- [ ] Create disasterbench/parsers/
- [ ] Create disasterbench/taxonomy/
- [ ] Create disasterbench/filters/
- [ ] Create disasterbench/catalog/
- [ ] Create disasterbench/persist/
- [ ] Refactor disasterbench/schemas/
- [ ] Keep existing loaders/exporters/tests during refactor

## Phase 2: Common Schema

- [ ] Define DatasetCard schema
- [ ] Define CommonSample schema
- [ ] Define MediaRecord schema
- [ ] Define AnnotationRecord schema
- [ ] Define SupervisionRecord schema
- [ ] Define LabelMappingRecord schema
- [ ] Define CapabilityRecord schema
- [ ] Define ExtensionRecord schema
- [ ] Add schema validation utilities
- [ ] Add schema serialization to JSON
- [ ] Add schema tests

## Phase 3: Verification Gate System

- [ ] Define verification status enum
- [ ] Define verification evidence object
- [ ] Define human review requirement object
- [ ] Add AUTO_VERIFIED, HUMAN_REVIEW_REQUIRED, NOT_VERIFIED, BLOCKED statuses
- [ ] Add validation reports for every stage
- [ ] Add failure reason logging
- [ ] Add tests for verification status behavior

## Phase 4: Inspection Layer

- [ ] Implement file structure scanner
- [ ] Implement file count scanner
- [ ] Implement image property scanner
- [ ] Implement raster property scanner
- [ ] Implement metadata inventory scanner
- [ ] Implement annotation key scanner
- [ ] Implement class inventory scanner
- [ ] Implement pairing checker
- [ ] Implement anomaly logger
- [ ] Implement inspection_report.json writer
- [ ] Implement inspection_summary.md writer

## Phase 5: Image Quality Assessment

- [ ] Implement image corruption check
- [ ] Implement blur score check
- [ ] Implement cloud cover placeholder or dataset-specific extractor
- [ ] Implement GSD metadata checker
- [ ] Implement quality_report.csv
- [ ] Implement quarantine_log.csv
- [ ] Mark thresholds as HUMAN_REVIEW_REQUIRED

## Phase 6: Duplicate Detection

- [ ] Implement exact file hash duplicate check
- [ ] Implement perceptual hash duplicate check
- [ ] Add optional geospatial footprint overlap check
- [ ] Create duplicate_report.csv
- [ ] Mark ambiguous duplicates as HUMAN_REVIEW_REQUIRED
- [ ] Add split leakage check

## Phase 7: Parser Registry

- [ ] Add GeoJSON parser
- [ ] Add COCO JSON parser
- [ ] Add YOLO text parser
- [ ] Add raster mask parser
- [ ] Add CSV parser
- [ ] Add parser registry
- [ ] Preserve raw labels and raw payloads
- [ ] Add parser tests

## Phase 8: Taxonomy and Label Mapping

- [ ] Create taxonomies/building_damage.yaml
- [ ] Create taxonomies/flood_extent.yaml
- [ ] Create taxonomies/disaster_type.yaml
- [ ] Implement label mapper
- [ ] Preserve original_label
- [ ] Store canonical_label
- [ ] Add mapping confidence
- [ ] Add human_review_required flag for ambiguous labels
- [ ] Add unmapped label report
- [ ] Add class_map.yaml writer

## Phase 9: Dataset Config System

- [ ] Define dataset config schema
- [ ] Add config validator
- [ ] Add human_review_status field
- [ ] Add supported_exports field
- [ ] Add unsupported_exports field
- [ ] Add lossy_conversions field
- [ ] Add metadata mapping section
- [ ] Add quality threshold section
- [ ] Add tests for invalid configs

## Phase 10: Dataset Adapters

- [ ] Refactor STURM-Flood loader to output CommonSample
- [ ] Refactor xBD loader to output CommonSample
- [ ] Preserve raw metadata
- [ ] Preserve original labels
- [ ] Add capability detection
- [ ] Add loader validation tests
- [ ] Ensure previous v0.2 tests still pass or are migrated

## Phase 11: Representation Converters

- [ ] Add polygon_to_bbox converter
- [ ] Add mask_to_binary converter
- [ ] Add mask_to_polygon converter
- [ ] Add mask_to_bbox converter
- [ ] Add bbox validator
- [ ] Add polygon validator
- [ ] Add coordinate normalization utilities
- [ ] Add geo-to-pixel transform utilities when required
- [ ] Mark lossy conversions as HUMAN_REVIEW_REQUIRED when needed
- [ ] Add conversion audit tests

## Phase 12: Capability Checker

- [ ] Define export requirements per target format
- [ ] Implement capability checker
- [ ] Add explainable unsupported export errors
- [ ] Add tests for valid and invalid export requests

## Phase 13: Filter Engine

- [ ] Implement disaster type filter
- [ ] Implement task type filter
- [ ] Implement annotation type filter
- [ ] Implement geographic bbox filter
- [ ] Implement minimum annotation count filter
- [ ] Implement quality status filter
- [ ] Implement split strategy
- [ ] Write filter_manifest.json
- [ ] Add tests for inclusion and exclusion reasons

## Phase 14: Generic Exporters

- [ ] Implement generic COCO exporter
- [ ] Implement generic YOLO detection exporter
- [ ] Implement generic YOLO segmentation exporter
- [ ] Implement generic GeoJSON exporter
- [ ] Implement generic mask exporter
- [ ] Add format configs
- [ ] Add export integrity tests

## Phase 15: Metadata Catalog and Persistence

- [ ] Implement dataset_card.json writer
- [ ] Implement provenance.json writer
- [ ] Implement license_summary.json writer
- [ ] Implement processing_log.json writer
- [ ] Implement statistics_summary.json writer
- [ ] Implement directory writer
- [ ] Implement content hash generator
- [ ] Implement config_snapshot.json writer
- [ ] Implement changelog support

## Phase 16: Documentation and Paper Preparation

- [ ] Update README
- [ ] Add architecture diagram
- [ ] Add dataset onboarding guide
- [ ] Add verification guide
- [ ] Add exporter guide
- [ ] Add limitations section
- [ ] Add paper notes
- [ ] Add experiments and benchmark plan
