"""
CLI for paired raster consistency inspection.

Example:
    python -m disasterbench.tools.inspect_paired_rasters \
      --reference-root datasets/sturm_flood_raw/Sentinel1/S1 \
      --candidate-root datasets/sturm_flood_raw/Sentinel1/Floodmaps \
      --output outputs/runs/sturm_s1_pair_consistency.json
"""

from __future__ import annotations

import argparse
import json

from disasterbench.inspection.paired_raster_consistency import (
    inspect_paired_raster_consistency,
    write_paired_raster_consistency_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check spatial consistency between paired raster files."
    )

    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument(
        "--reference-extensions",
        nargs="+",
        default=[".tif", ".tiff"],
    )
    parser.add_argument(
        "--candidate-extensions",
        nargs="+",
        default=[".tif", ".tiff"],
    )
    parser.add_argument("--strip-suffix", action="append", default=[])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--transform-tolerance", type=float, default=1e-9)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    result = inspect_paired_raster_consistency(
        reference_root=args.reference_root,
        candidate_root=args.candidate_root,
        reference_extensions=args.reference_extensions,
        candidate_extensions=args.candidate_extensions,
        strip_suffixes=args.strip_suffix,
        max_pairs=args.max_pairs,
        transform_tolerance=args.transform_tolerance,
    )

    result_dict = result.to_dict()

    if args.output:
        write_paired_raster_consistency_report(result, args.output)

    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
