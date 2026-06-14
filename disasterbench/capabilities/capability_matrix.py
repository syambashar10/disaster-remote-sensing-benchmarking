from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _add_rows(
    rows: List[Dict[str, Any]],
    dataset_id: str,
    dataset_name: str,
    loader_name: str,
    data_family: str,
    formats: List[str],
    support_level: str,
) -> None:
    for output_format in formats:
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "loader_name": loader_name,
                "data_family": data_family,
                "output_format": output_format,
                "support_level": support_level,
            }
        )


def build_capability_matrix(registry_path: str | Path) -> Dict[str, Any]:
    registry_path = Path(registry_path)
    registry = read_json(registry_path)

    rows: List[Dict[str, Any]] = []

    for entry in registry.get("datasets", []):
        config_path = entry["config_path"]
        config = read_json(config_path)

        dataset_id = config.get("dataset_id", entry.get("dataset_id"))
        dataset_name = config.get("dataset_name", "-")
        loader_name = config.get("loader_name", entry.get("loader_family", "-"))
        data_family = config.get("data_family", "-")

        _add_rows(
            rows,
            dataset_id,
            dataset_name,
            loader_name,
            data_family,
            config.get("recommended_formats", []),
            "supported",
        )

        _add_rows(
            rows,
            dataset_id,
            dataset_name,
            loader_name,
            data_family,
            config.get("lossy_or_derived_formats", []),
            "lossy_or_derived",
        )

        _add_rows(
            rows,
            dataset_id,
            dataset_name,
            loader_name,
            data_family,
            config.get("unsupported_formats", []),
            "unsupported",
        )

    datasets = sorted({row["dataset_id"] for row in rows})
    output_formats = sorted({row["output_format"] for row in rows})

    matrix = {
        dataset_id: {
            output_format: "unknown"
            for output_format in output_formats
        }
        for dataset_id in datasets
    }

    for row in rows:
        matrix[row["dataset_id"]][row["output_format"]] = row["support_level"]

    return {
        "registry": str(registry_path),
        "registry_name": registry.get("registry_name"),
        "version": registry.get("version"),
        "dataset_count": len(datasets),
        "output_format_count": len(output_formats),
        "datasets": datasets,
        "output_formats": output_formats,
        "rows": rows,
        "matrix": matrix,
    }


def write_capability_matrix_json(report: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_capability_matrix_markdown(report: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Dataset Export Capability Matrix")
    lines.append("")
    lines.append(f"**Registry:** {report.get('registry_name', '-')}")
    lines.append(f"**Version:** {report.get('version', '-')}")
    lines.append("")
    lines.append("Legend:")
    lines.append("")
    lines.append("- `supported` = native / recommended output")
    lines.append("- `lossy_or_derived` = possible but derived from another annotation type")
    lines.append("- `unsupported` = not supported safely from current annotation type")
    lines.append("- `unknown` = not declared in config")
    lines.append("")

    output_formats = report["output_formats"]

    header = ["Dataset"] + output_formats
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for dataset_id in report["datasets"]:
        row = [dataset_id]
        for output_format in output_formats:
            row.append(report["matrix"][dataset_id].get(output_format, "unknown"))

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Detailed Rows")
    lines.append("")
    lines.append("| Dataset | Loader | Data family | Output format | Support level |")
    lines.append("|---|---|---|---|---|")

    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset_id"],
                    row["loader_name"],
                    row["data_family"],
                    row["output_format"],
                    row["support_level"],
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
