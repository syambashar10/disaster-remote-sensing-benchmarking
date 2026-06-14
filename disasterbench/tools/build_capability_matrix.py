from __future__ import annotations

import argparse

from disasterbench.capabilities.capability_matrix import (
    build_capability_matrix,
    write_capability_matrix_json,
    write_capability_matrix_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build export capability matrix from dataset registry configs."
    )
    parser.add_argument("--registry", required=True, help="Path to dataset registry JSON.")
    parser.add_argument("--json-output", required=True, help="Output JSON report path.")
    parser.add_argument("--markdown-output", required=True, help="Output Markdown report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = build_capability_matrix(args.registry)

    write_capability_matrix_json(report, args.json_output)
    write_capability_matrix_markdown(report, args.markdown_output)

    print(f"Wrote JSON capability matrix to: {args.json_output}")
    print(f"Wrote Markdown capability matrix to: {args.markdown_output}")


if __name__ == "__main__":
    main()
