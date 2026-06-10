"""
Standard output package writer.

A processed dataset/run should always be written into a predictable directory
layout. This makes outputs easier to audit, reproduce, share, and compare.

This module creates the output structure and writes run_manifest.json plus a
config snapshot. It does not run dataset processing by itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


STANDARD_OUTPUT_DIRS = [
    "inspection",
    "filtering",
    "exports",
    "catalog",
    "logs",
]


@dataclass
class OutputPackage:
    """Metadata for one generated output package."""

    dataset_id: str
    run_id: str
    root_dir: str
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    directories: Dict[str, str] = field(default_factory=dict)
    config_snapshot_path: Optional[str] = None
    run_manifest_path: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def stable_json_hash(data: Dict[str, Any]) -> str:
    """
    Create a stable hash for a JSON-compatible dictionary.

    This can be used later for versioning config snapshots and processing runs.
    """

    encoded = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def create_output_package(
    dataset_id: str,
    output_root: str | Path,
    run_id: Optional[str] = None,
) -> OutputPackage:
    """
    Create the standard output directory structure.

    If run_id is not provided, a timestamp-based run_id is generated.
    """

    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{dataset_id}_{timestamp}"

    root_dir = Path(output_root) / run_id
    root_dir.mkdir(parents=True, exist_ok=True)

    directories: Dict[str, str] = {}

    for dirname in STANDARD_OUTPUT_DIRS:
        dir_path = root_dir / dirname
        dir_path.mkdir(parents=True, exist_ok=True)
        directories[dirname] = str(dir_path)

    return OutputPackage(
        dataset_id=dataset_id,
        run_id=run_id,
        root_dir=str(root_dir),
        directories=directories,
    )


def write_config_snapshot(
    package: OutputPackage,
    config: Dict[str, Any],
    filename: str = "config_snapshot.json",
) -> Path:
    """Write the exact config used for a run."""

    path = Path(package.root_dir) / filename

    snapshot = {
        "dataset_id": package.dataset_id,
        "run_id": package.run_id,
        "config_hash_sha256": stable_json_hash(config),
        "config": config,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, ensure_ascii=False)

    package.config_snapshot_path = str(path)
    return path


def write_run_manifest(
    package: OutputPackage,
    extra_metadata: Optional[Dict[str, Any]] = None,
    filename: str = "run_manifest.json",
) -> Path:
    """
    Write run_manifest.json.

    The run manifest records the generated folder structure and links to the
    config snapshot. More processing artifacts can be added later.
    """

    path = Path(package.root_dir) / filename

    manifest = package.to_dict()
    manifest["extra_metadata"] = extra_metadata or {}

    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    package.run_manifest_path = str(path)

    # Rewrite once so run_manifest_path is included inside the manifest.
    manifest = package.to_dict()
    manifest["extra_metadata"] = extra_metadata or {}

    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    return path


def create_standard_output_package(
    dataset_id: str,
    output_root: str | Path,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> OutputPackage:
    """
    Create standard output package and write config/run metadata.

    Returns the OutputPackage object with paths filled in.
    """

    package = create_output_package(
        dataset_id=dataset_id,
        output_root=output_root,
        run_id=run_id,
    )

    write_config_snapshot(package, config)
    write_run_manifest(package, extra_metadata=extra_metadata)

    return package
