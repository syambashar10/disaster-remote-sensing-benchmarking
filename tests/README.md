# Tests

This folder contains reproducible tests for the DisasterBench framework.

## STURM-Flood Loader Test

Run:

```bash
python tests/test_sturm_flood_loader.py

This test verifies that:

Sentinel-1 loader initializes correctly
Sentinel-2 loader initializes correctly
image-mask-metadata pairing has zero issues
Sentinel-1 has 21,602 indexed samples
Sentinel-2 has 2,675 indexed samples
Sentinel-1 samples have 2 bands
Sentinel-2 samples have 9 bands
loaded samples follow the common internal schema

The test output is saved to:

outputs/verification/sturm_flood_loader_test.json
Full STURM-Flood Verification

Run:

python -m disasterbench.tools.full_verify_sturm_flood

This performs a full scan of every STURM-Flood image and mask file.

It checks:

image count
mask count
metadata row count
image-mask-metadata pairing
image dimensions
mask dimensions
image band counts
mask band counts
image and mask CRS
image and mask bounds
image and mask transform alignment
image dtype
mask dtype
global mask values
unexpected/anomalous samples

Outputs:

outputs/verification/sturm_flood_full_verification_summary.json
outputs/verification/sturm_flood_full_verification_anomalies.csv

