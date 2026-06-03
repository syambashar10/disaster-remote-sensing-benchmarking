# Changelog

## v0.1.0 - STURM-Flood Framework Integration

Initial DisasterBench framework release focused on STURM-Flood integration.

### Added

- Verification-first dataset workflow
- STURM-Flood dataset loader
- Common internal sample schema
- Raster mask to binary mask converter
- Raster mask to polygon converter
- Raster mask to bounding box converter
- GeoJSON exporter
- COCO segmentation exporter
- YOLO segmentation exporter
- YOLO detection bounding box exporter
- Original and binary mask exporter
- User-facing dataset export CLI
- Full STURM-Flood converter verification
- Visual conversion quality check utility
- Documentation for STURM-Flood pipeline status

### Notes

STURM-Flood is treated as a raster-mask semantic segmentation dataset. Detection-style bounding box export is supported as a derived/lossy representation.
