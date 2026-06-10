"""
CLI for dataset verification pipeline.

Example:
    python -m disasterbench.tools.verify_dataset \
      --config configs/xbd_config.json \
      --output-root outputs/runs \
      --run-id xbd_verification_pipeline_test
"""

from __future__ import annotations

import argparse
import json

from disasterbench.pipelines import run_dataset_verification_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dataset verification pipeline from config."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit for paired raster checks. Use for sample/quick verification.",
    )

    args = parser.parse_args()

    result = run_dataset_verification_pipeline(
        config_path=args.config,
        output_root=args.output_root,
        run_id=args.run_id,
        max_pairs=args.max_pairs,
    )

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
