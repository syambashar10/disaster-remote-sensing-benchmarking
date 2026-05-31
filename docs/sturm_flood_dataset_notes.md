# STURM-Flood Dataset Notes

## Purpose

STURM-Flood is used as the second dataset in this disaster remote sensing benchmarking framework. It is included because it has a different annotation format from xBD/xView2. While xBD uses building polygons and damage labels, STURM-Flood uses GeoTIFF satellite image tiles and raster floodmap masks.

## Dataset Type

STURM-Flood is a flood extent mapping dataset for semantic segmentation. Each sample contains a satellite image tile and a corresponding floodmap mask.

## Dataset Structure

The local dataset contains:

```text
Dataset/
├── Sentinel1/
│   ├── S1/
│   └── Floodmaps/
├── Sentinel2/
│   ├── S2/
│   └── Floodmaps/
├── sentinel1_metadata.csv
└── sentinel2_metadata.csv
Sentinel-1

Sentinel-1 contains radar/SAR imagery. In this dataset, each Sentinel-1 tile has 2 bands: VV and VH.

Inspection result:

Sentinel-1 images: 21,602
Sentinel-1 floodmaps: 21,602
Metadata rows: 21,602
Tile size: 128 x 128
Sentinel-2

Sentinel-2 contains optical/multispectral imagery. In this dataset, each Sentinel-2 tile has 9 bands: B2, B3, B4, B8, B5, B6, B7, B11, and B12.

Inspection result:

Sentinel-2 images: 2,675
Sentinel-2 floodmaps: 2,675
Metadata rows: 2,675
Tile size: 128 x 128
Floodmap Masks

The floodmaps are single-band raster masks. Each pixel value represents a class.

Original mask values:

0  = coastline / area of interest
1  = flooded area
2  = river-related features
3  = open water
4  = reservoirs
5  = lakes
99 = no data

For binary water mapping, the water-related classes can be grouped as:

non-water = [0]
water = [1, 2, 3, 4, 5]
no-data = [99]
Pairing Check

The inspection confirmed that image files, floodmap files, and metadata rows are correctly paired by filename / tile_id.

Sentinel-1 missing pairs: 0
Sentinel-2 missing pairs: 0
Framework Relevance

STURM-Flood confirms that the framework must support raster-mask datasets, not only polygon or bounding-box datasets. This makes it useful for testing the common schema and dataloader design across different annotation types.

