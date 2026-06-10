"""
CLI for reusable raster / GeoTIFF inspection.

Example:
    python -m disasterbench.tools.inspect_rasters \
      --raster-root datasets/sturm_flood_raw/Sentinel1/Floodmaps \
      --collect-unique-values \
      --output outputs/runs/sturm_s1_floodmaps_raster_report.json
"""

from __future__ import annotations

import argparse
import json

from disasterbench.inspection.raster_inspector import (
    inspect_rasters,
    write_raster_inspection_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect raster files without dataset-specific assumptions."
    )

    parser.add_argument("--raster-root", required=True)
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".tif", ".tiff"],
        help="Raster extensions to inspect.",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--collect-unique-values",
        action="store_true",
        help="Read band 1 and count unique pixel values. Useful for masks.",
    )
    parser.add_argument(
        "--unique-values-max-files",
        type=int,
        default=None,
        help="Optional limit for unique-value scans.",
    )
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    result = inspect_rasters(
        raster_root=args.raster_root,
        extensions=args.extensions,
        max_files=args.max_files,
        collect_unique_values=args.collect_unique_values,
        unique_values_max_files=args.unique_values_max_files,
    )

    result_dict = result.to_dict()

    if args.output:
        write_raster_inspection_report(result, args.output)

    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
