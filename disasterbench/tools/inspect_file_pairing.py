"""
CLI for reusable file pairing inspection.

Example:
    python -m disasterbench.tools.inspect_file_pairing \
      --reference-root datasets/xbd_raw/train/images \
      --reference-extensions .png \
      --candidate labels:datasets/xbd_raw/train/labels:.json \
      --candidate targets:datasets/xbd_raw/train/targets:.png \
      --output outputs/runs/xbd_pairing_report.json
"""

from __future__ import annotations

import argparse
import json

from disasterbench.inspection.file_pairing_inspector import (
    inspect_file_pairing,
    write_file_pairing_report,
)


def parse_candidate(value: str):
    parts = value.split(":", 2)

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Candidate must be in format name:path:.ext,.ext2"
        )

    name, path, extensions_text = parts
    extensions = [item.strip() for item in extensions_text.split(",") if item.strip()]

    if not name or not path or not extensions:
        raise argparse.ArgumentTypeError(
            "Candidate must include name, path, and at least one extension."
        )

    return name, path, extensions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect filename-stem pairing between folders."
    )

    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--reference-extensions", nargs="+", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help="Candidate in format name:path:.ext,.ext2",
    )
    parser.add_argument(
        "--strip-suffix",
        action="append",
        default=[],
        help="Optional filename suffix to strip before pairing.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    candidate_roots = {}
    candidate_extensions = {}

    for name, path, extensions in args.candidate:
        candidate_roots[name] = path
        candidate_extensions[name] = extensions

    result = inspect_file_pairing(
        reference_root=args.reference_root,
        candidate_roots=candidate_roots,
        reference_extensions=args.reference_extensions,
        candidate_extensions=candidate_extensions,
        strip_suffixes=args.strip_suffix,
        max_examples=args.max_examples,
    )

    result_dict = result.to_dict()

    if args.output:
        write_file_pairing_report(result, args.output)

    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
