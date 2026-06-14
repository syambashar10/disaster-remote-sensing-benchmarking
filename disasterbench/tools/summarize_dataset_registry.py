from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a human-readable dataset registry summary report."
    )
    parser.add_argument("--registry", required=True, help="Path to dataset registry JSON.")
    parser.add_argument("--output", required=True, help="Output Markdown report path.")
    return parser.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list_text(values: List[str] | None) -> str:
    if not values:
        return "-"
    return ", ".join(values)


def main() -> None:
    args = parse_args()

    registry_path = Path(args.registry)
    registry = _read_json(registry_path)

    lines = []
    lines.append(f"# {registry.get('registry_name', 'Dataset Registry Summary')}")
    lines.append("")
    lines.append(f"**Version:** {registry.get('version', '-')}")
    lines.append("")
    lines.append("## Registered Datasets")
    lines.append("")
    lines.append("| Dataset | Loader family | Data family | Task types | Status | Human review |")
    lines.append("|---|---|---|---|---|---|")

    detailed_sections = []

    for entry in registry.get("datasets", []):
        dataset_id = entry["dataset_id"]
        config_path = entry["config_path"]
        config = _read_json(config_path)

        loader_name = config.get("loader_name", entry.get("loader_family", "-"))
        data_family = config.get("data_family", "-")
        task_types = _as_list_text(config.get("task_type", []))
        status = config.get("status", entry.get("status", "-"))
        human_review = str(config.get("human_review_required", False))

        lines.append(
            f"| {dataset_id} | {loader_name} | {data_family} | {task_types} | {status} | {human_review} |"
        )

        detailed_sections.append("")
        detailed_sections.append(f"## {dataset_id}")
        detailed_sections.append("")
        detailed_sections.append(f"- **Config:** `{config_path}`")
        detailed_sections.append(f"- **Dataset name:** {config.get('dataset_name', '-')}")
        detailed_sections.append(f"- **Loader:** {loader_name}")
        detailed_sections.append(f"- **Data family:** {data_family}")
        detailed_sections.append(f"- **Raw annotation type:** {config.get('raw_annotation_type', '-')}")
        detailed_sections.append(f"- **Image format:** {config.get('image_format', '-')}")
        detailed_sections.append(f"- **Annotation format:** {config.get('annotation_format', '-')}")
        detailed_sections.append(f"- **Status:** {status}")
        detailed_sections.append(f"- **Human review required:** {human_review}")
        detailed_sections.append("")
        detailed_sections.append("### Output support")
        detailed_sections.append("")
        detailed_sections.append(f"- **Recommended formats:** {_as_list_text(config.get('recommended_formats', []))}")
        detailed_sections.append(f"- **Lossy/derived formats:** {_as_list_text(config.get('lossy_or_derived_formats', []))}")
        detailed_sections.append(f"- **Unsupported formats:** {_as_list_text(config.get('unsupported_formats', []))}")

        label_schema = config.get("label_schema", {})
        if label_schema:
            detailed_sections.append("")
            detailed_sections.append("### Label schema")
            detailed_sections.append("")
            detailed_sections.append(f"- **Label values:** `{label_schema.get('label_values', {})}`")
            detailed_sections.append(f"- **Notes:** {label_schema.get('notes', '-')}")

    lines.extend(detailed_sections)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote registry summary report to: {output_path}")


if __name__ == "__main__":
    main()
