# Dataset Download Links and Local Storage Documentation

This document records the datasets currently used or downloaded for the disaster remote sensing benchmarking framework. The links below point to the official dataset pages or the exact download platforms used during the project.

---

## 1. STURM-Flood

**Dataset name:** STURM-Flood  
**Task type:** Flood extent mapping / semantic segmentation  
**Data type:** Sentinel-1 and Sentinel-2 image tiles with raster flood/water masks  
**Source platform:** Zenodo  

**Download/source link:**

```text
https://zenodo.org/records/12748983

GitHub/code page:

https://github.com/STURM-WEO/STURM-Flood

Local server storage path:

/export/qcai-HumAI4SAT/RS_NDisaster_Datasets/01_sturm_flood/

Framework config used:

configs/sturm_flood_raster_pair_config.json

Loader family used:

RasterMaskPairLoader
2. xBD / xView2

Dataset name: xBD / xView2 Challenge Dataset
Task type: Building damage assessment, object detection, instance segmentation, change analysis
Data type: Satellite images with JSON polygon building annotations
Source platform used: Kaggle

Download/source link used:

https://www.kaggle.com/datasets/tunguz/xview2-challenge-dataset-train-and-test

Local server storage path:

/export/qcai-HumAI4SAT/RS_NDisaster_Datasets/02_xbd_xview2/

Framework config used:

configs/xbd_kaggle_config.json

Loader family used:

JsonPolygonLoader
3. DisasterM3

Dataset name: DisasterM3
Task type: Remote sensing vision-language disaster assessment and response
Data type: Bi-temporal satellite images with multimodal / instruction-style labels
Source platform: Hugging Face

Download/source link:

https://huggingface.co/datasets/Kingdrone-Junjue/DisasterM3

Dataset files page:

https://huggingface.co/datasets/Kingdrone-Junjue/DisasterM3/tree/main

Paper / project reference:

https://arxiv.org/abs/2505.21089

Local server storage path:

/export/qcai-HumAI4SAT/RS_NDisaster_Datasets/03_disasterm3/

Current framework status:

Downloaded and basic structure checked. Deeper inspection and loader-family decision still pending.
4. Sen2Fire

Dataset name: Sen2Fire
Task type: Wildfire detection / semantic segmentation
Data type: Sentinel-2 multi-spectral data and Sentinel-5P aerosol product packaged as NPZ files
Source platform: Zenodo

Download/source link:

https://zenodo.org/records/10881058

Paper reference:

https://arxiv.org/abs/2403.17884

Local server storage path:

/export/qcai-HumAI4SAT/RS_NDisaster_Datasets/04_sen2fire/

Framework config used:

configs/sen2fire_config.json

Loader family used:

NPZSegmentationLoader
5. MMFlood

Dataset name: MMFlood
Task type: Flood delineation / semantic segmentation
Data type: Sentinel-1 image data with mask, DEM, and hydrography data
Source platform: Zenodo

Download/source link:

https://zenodo.org/records/6534637

Local server storage path:

/export/qcai-HumAI4SAT/RS_NDisaster_Datasets/05_mmflood/

Framework config used:

configs/mmflood_config.json

Loader family used:

RasterMaskPairLoader
6. GDCLD

Dataset name: GDCLD: Globally Distributed Coseismic Landslide Dataset
Task type: Landslide mapping / semantic segmentation
Data type: High-resolution remote sensing images with binary landslide masks
Source platform: Zenodo

Download/source link:

https://zenodo.org/records/13612636

Related publication page:

https://essd.copernicus.org/articles/16/4817/2024/

Local server storage path:

/export/qcai-HumAI4SAT/RS_NDisaster_Datasets/06_gdcld/

Framework config used:

configs/gdcld_config.json

Loader family used:

RasterMaskPairLoader
Download Commands / Platform Notes

For Zenodo datasets, the dataset record page should be used as the official download reference. In the framework workspace, the Zenodo record ID can also be used with the internal download script to download all files from a record.

Example pattern:

python /export/qcai-HumAI4SAT/RS_NDisaster_Datasets/_scripts/download_zenodo_record.py \
  --record-id <ZENODO_RECORD_ID> \
  --output-dir <TARGET_DATASET_FOLDER>

Zenodo record IDs used:

STURM-Flood: 12748983
Sen2Fire:    10881058
MMFlood:     6534637
GDCLD:       13612636

For Kaggle xBD / xView2, the Kaggle CLI can be used after Kaggle authentication is configured:

kaggle datasets download \
  -d tunguz/xview2-challenge-dataset-train-and-test \
  -p /export/qcai-HumAI4SAT/RS_NDisaster_Datasets/02_xbd_xview2/

For DisasterM3, the Hugging Face CLI can be used:

hf download Kingdrone-Junjue/DisasterM3 \
  --repo-type dataset \
  --local-dir /export/qcai-HumAI4SAT/RS_NDisaster_Datasets/03_disasterm3/raw
Current Framework Registry Coverage

The current registry includes five fully integrated datasets:

xBD Kaggle  -> JsonPolygonLoader
Sen2Fire    -> NPZSegmentationLoader
STURM-Flood -> RasterMaskPairLoader
GDCLD       -> RasterMaskPairLoader
MMFlood     -> RasterMaskPairLoader

DisasterM3 has been downloaded and organized, but still requires deeper inspection before being added to the registry pipeline.
