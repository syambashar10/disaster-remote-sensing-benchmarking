import json
import subprocess
import sys

from disasterbench.inspection.file_pairing_inspector import inspect_file_pairing


def test_file_pairing_inspector_detects_matches_missing_and_extra(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    targets = tmp_path / "targets"

    images.mkdir()
    labels.mkdir()
    targets.mkdir()

    (images / "sample_001.png").write_bytes(b"image")
    (images / "sample_002.png").write_bytes(b"image")
    (labels / "sample_001.json").write_text("{}", encoding="utf-8")
    (labels / "extra_label.json").write_text("{}", encoding="utf-8")
    (targets / "sample_001.png").write_bytes(b"mask")
    (targets / "sample_002.png").write_bytes(b"mask")

    result = inspect_file_pairing(
        reference_root=images,
        reference_extensions=[".png"],
        candidate_roots={
            "labels": labels,
            "targets": targets,
        },
        candidate_extensions={
            "labels": [".json"],
            "targets": [".png"],
        },
    )

    assert result.reference_file_count == 2
    assert result.candidate_file_counts["labels"] == 2
    assert result.candidate_file_counts["targets"] == 2
    assert result.matched_counts["labels"] == 1
    assert result.matched_counts["targets"] == 2
    assert result.missing_from_candidates["labels"] == ["sample_002"]
    assert result.extra_in_candidates["labels"] == ["extra_label"]
    assert result.missing_from_candidates["targets"] == []


def test_file_pairing_inspector_detects_duplicate_stems(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"

    images.mkdir()
    labels.mkdir()

    (images / "sample_001.png").write_bytes(b"image")
    (images / "sample_001.jpg").write_bytes(b"image")
    (labels / "sample_001.json").write_text("{}", encoding="utf-8")

    result = inspect_file_pairing(
        reference_root=images,
        reference_extensions=[".png", ".jpg"],
        candidate_roots={"labels": labels},
        candidate_extensions={"labels": [".json"]},
    )

    assert "sample_001" in result.duplicate_reference_stems
    assert result.has_duplicates()


def test_file_pairing_inspector_supports_suffix_stripping(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"

    images.mkdir()
    labels.mkdir()

    (images / "tile_001_pre_disaster.png").write_bytes(b"image")
    (labels / "tile_001_post_disaster.json").write_text("{}", encoding="utf-8")

    result = inspect_file_pairing(
        reference_root=images,
        reference_extensions=[".png"],
        candidate_roots={"labels": labels},
        candidate_extensions={"labels": [".json"]},
        strip_suffixes=["_pre_disaster", "_post_disaster"],
    )

    assert result.matched_counts["labels"] == 1
    assert result.missing_from_candidates["labels"] == []


def test_inspect_file_pairing_cli_writes_output(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"

    images.mkdir()
    labels.mkdir()

    (images / "sample_001.png").write_bytes(b"image")
    (labels / "sample_001.json").write_text("{}", encoding="utf-8")

    output_path = tmp_path / "pairing_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.inspect_file_pairing",
            "--reference-root",
            str(images),
            "--reference-extensions",
            ".png",
            "--candidate",
            f"labels:{labels}:.json",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_data = json.loads(completed.stdout)
    file_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_data["matched_counts"]["labels"] == 1
    assert file_data["matched_counts"]["labels"] == 1
    assert output_path.exists()
