"""
Reusable file pairing inspector.

This module checks whether files in one reference folder match files in one or
more candidate folders using normalized filename stems.

Example:
    images/sample_001.png
    labels/sample_001.json
    masks/sample_001.png

The shared stem is sample_001.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class FilePairingInspectionResult:
    reference_root: str
    reference_extensions: List[str]
    candidate_roots: Dict[str, str]
    candidate_extensions: Dict[str, List[str]]
    strip_suffixes: List[str]
    reference_file_count: int
    candidate_file_counts: Dict[str, int]
    matched_counts: Dict[str, int]
    missing_from_candidates: Dict[str, List[str]]
    extra_in_candidates: Dict[str, List[str]]
    duplicate_reference_stems: Dict[str, List[str]]
    duplicate_candidate_stems: Dict[str, Dict[str, List[str]]]
    missing_roots: Dict[str, str] = field(default_factory=dict)

    def has_missing_pairs(self) -> bool:
        return any(self.missing_from_candidates.values())

    def has_extra_pairs(self) -> bool:
        return any(self.extra_in_candidates.values())

    def has_duplicates(self) -> bool:
        return bool(self.duplicate_reference_stems) or any(
            duplicates for duplicates in self.duplicate_candidate_stems.values()
        )

    def to_dict(self) -> Dict:
        return asdict(self)


def normalize_extensions(extensions: Optional[Iterable[str]]) -> Optional[set[str]]:
    if extensions is None:
        return None

    normalized = set()

    for extension in extensions:
        extension = str(extension).lower().strip()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.add(extension)

    return normalized


def normalize_stem(path: Path, strip_suffixes: Optional[Iterable[str]] = None) -> str:
    stem = path.stem

    for suffix in strip_suffixes or []:
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def build_stem_index(
    root: str | Path,
    extensions: Optional[Iterable[str]] = None,
    strip_suffixes: Optional[Iterable[str]] = None,
) -> Dict[str, List[str]]:
    root = Path(root)
    allowed_extensions = normalize_extensions(extensions)

    index: Dict[str, List[str]] = defaultdict(list)

    if not root.exists() or not root.is_dir():
        return {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
            continue

        stem = normalize_stem(path, strip_suffixes=strip_suffixes)
        index[stem].append(str(path))

    return dict(index)


def find_duplicates(index: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        stem: paths
        for stem, paths in sorted(index.items())
        if len(paths) > 1
    }


def limit_values(values: Iterable[str], max_examples: int) -> List[str]:
    return list(sorted(values))[:max_examples]


def inspect_file_pairing(
    reference_root: str | Path,
    candidate_roots: Dict[str, str | Path],
    reference_extensions: Optional[Iterable[str]] = None,
    candidate_extensions: Optional[Dict[str, Iterable[str]]] = None,
    strip_suffixes: Optional[Iterable[str]] = None,
    max_examples: int = 20,
) -> FilePairingInspectionResult:
    reference_root_path = Path(reference_root)
    strip_suffixes_list = list(strip_suffixes or [])

    missing_roots: Dict[str, str] = {}

    if not reference_root_path.exists() or not reference_root_path.is_dir():
        missing_roots["reference"] = str(reference_root_path)

    reference_index = build_stem_index(
        reference_root_path,
        extensions=reference_extensions,
        strip_suffixes=strip_suffixes_list,
    )
    reference_stems = set(reference_index.keys())

    candidate_indexes = {}
    candidate_file_counts = {}
    matched_counts = {}
    missing_from_candidates = {}
    extra_in_candidates = {}
    duplicate_candidate_stems = {}

    candidate_extensions = candidate_extensions or {}

    for name, root in candidate_roots.items():
        candidate_root_path = Path(root)

        if not candidate_root_path.exists() or not candidate_root_path.is_dir():
            missing_roots[name] = str(candidate_root_path)

        index = build_stem_index(
            candidate_root_path,
            extensions=candidate_extensions.get(name),
            strip_suffixes=strip_suffixes_list,
        )
        candidate_indexes[name] = index

        candidate_stems = set(index.keys())

        candidate_file_counts[name] = sum(len(paths) for paths in index.values())
        matched_counts[name] = len(reference_stems & candidate_stems)

        missing_from_candidates[name] = limit_values(
            reference_stems - candidate_stems,
            max_examples=max_examples,
        )
        extra_in_candidates[name] = limit_values(
            candidate_stems - reference_stems,
            max_examples=max_examples,
        )
        duplicate_candidate_stems[name] = find_duplicates(index)

    return FilePairingInspectionResult(
        reference_root=str(reference_root_path),
        reference_extensions=sorted(normalize_extensions(reference_extensions) or []),
        candidate_roots={name: str(root) for name, root in candidate_roots.items()},
        candidate_extensions={
            name: sorted(normalize_extensions(values) or [])
            for name, values in candidate_extensions.items()
        },
        strip_suffixes=strip_suffixes_list,
        reference_file_count=sum(len(paths) for paths in reference_index.values()),
        candidate_file_counts=candidate_file_counts,
        matched_counts=matched_counts,
        missing_from_candidates=missing_from_candidates,
        extra_in_candidates=extra_in_candidates,
        duplicate_reference_stems=find_duplicates(reference_index),
        duplicate_candidate_stems=duplicate_candidate_stems,
        missing_roots=missing_roots,
    )


def write_file_pairing_report(
    result: FilePairingInspectionResult,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return output_path
