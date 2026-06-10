import json
import subprocess
import sys

from disasterbench.inspection import (
    collect_configured_paths,
    inspect_dataset_config,
    run_dataset_inspection,
)


def make_fake_dataset(root):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "train" / "targets").mkdir(parents=True)

    (root / "train" / "images" / "sample_001.png").write_bytes(b"fake image")
    (root / "train" / "labels" / "sample_001.json").write_text(
        '{"features": []}',
        encoding="utf-8",
    )
    (root / "train" / "targets" / "sample_001.png").write_bytes(b"fake mask")


def test_collect_configured_paths_supports_legacy_config_patterns(tmp_path):
    config = {
        "dataset_id": "xbd",
        "dataset_path": str(tmp_path),
        "split_structure": {
            "train_images": "train/images",
            "train_labels": "train/labels",
        },
        "sentinel1": {
            "image_folder": "Sentinel1/S1",
            "metadata_file": "sentinel1_metadata.csv",
        },
    }

    paths = collect_configured_paths(config)

    fields = {item["field"] for item in paths}

    assert "split_structure.train_images" in fields
    assert "split_structure.train_labels" in fields
    assert "sentinel1.image_folder" in fields
    assert "sentinel1.metadata_file" in fields


def test_generic_inspector_counts_files_and_flags_semantic_review(tmp_path):
    dataset_root = tmp_path / "fake_xbd"
    make_fake_dataset(dataset_root)

    config = {
        "dataset_id": "xbd",
        "dataset_name": "xBD",
        "dataset_path": str(dataset_root),
        "task_type": ["building_damage_assessment"],
        "raw_annotation_type": "polygon",
        "damage_classes": ["no-damage", "destroyed"],
        "split_structure": {
            "train_images": "train/images",
            "train_labels": "train/labels",
            "train_targets": "train/targets",
        },
    }

    report = inspect_dataset_config(config)

    assert report.dataset_id == "xbd"
    assert not report.has_blocking_issues()
    assert report.file_counts["total_files"] == 3
    assert report.file_counts[".png"] == 2
    assert report.file_counts[".json"] == 1
    assert report.requires_human_review()
    assert report.pairing_summary["configured_paths"]


def test_generic_inspector_blocks_missing_dataset_root():
    config = {
        "dataset_id": "missing_dataset",
        "dataset_path": "/definitely/not/a/real/path",
        "task_type": ["semantic_segmentation"],
    }

    report = inspect_dataset_config(config)

    assert report.has_blocking_issues()
    assert report.requires_human_review()


def test_run_dataset_inspection_writes_standard_outputs(tmp_path):
    dataset_root = tmp_path / "fake_xbd"
    make_fake_dataset(dataset_root)

    config = {
        "dataset_id": "xbd",
        "dataset_path": str(dataset_root),
        "task_type": ["building_damage_assessment"],
        "split_structure": {
            "train_images": "train/images",
            "train_labels": "train/labels",
        },
    }

    config_path = tmp_path / "xbd_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = run_dataset_inspection(
        config_path=config_path,
        output_root=tmp_path / "runs",
        run_id="xbd_inspection_test",
    )

    package = result["package"]
    outputs = result["inspection_outputs"]

    assert package.run_id == "xbd_inspection_test"
    assert outputs["inspection_report_json"].exists()
    assert outputs["inspection_summary_md"].exists()
    assert (tmp_path / "runs" / "xbd_inspection_test" / "config_snapshot.json").exists()
    assert (tmp_path / "runs" / "xbd_inspection_test" / "run_manifest.json").exists()


def test_inspect_dataset_cli_writes_outputs(tmp_path):
    dataset_root = tmp_path / "fake_xbd"
    make_fake_dataset(dataset_root)

    config = {
        "dataset_id": "xbd",
        "dataset_path": str(dataset_root),
        "task_type": ["building_damage_assessment"],
        "split_structure": {
            "train_images": "train/images",
            "train_labels": "train/labels",
        },
    }

    config_path = tmp_path / "xbd_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.inspect_dataset",
            "--config",
            str(config_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-id",
            "cli_inspection_test",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)

    assert output["dataset_id"] == "xbd"
    assert output["run_id"] == "cli_inspection_test"
    assert output["blocking_issues"] is False
    assert output["human_review_required"] is True
    assert (tmp_path / "runs" / "cli_inspection_test").exists()
