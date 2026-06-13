from pathlib import Path

from disasterbench.inspection.dataset_discovery import discover_dataset


def test_discover_dataset_detects_image_label_pairing(tmp_path: Path):
    dataset_root = tmp_path / "sample_dataset"
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for stem in ["sample_001", "sample_002"]:
        (images_dir / f"{stem}.png").write_bytes(b"fake image")
        (labels_dir / f"{stem}.json").write_text('{"features": []}', encoding="utf-8")

    output_root = tmp_path / "outputs"

    report = discover_dataset(
        dataset_root=dataset_root,
        dataset_id="sample_dataset",
        output_root=output_root,
        max_samples=2,
    )

    assert report["dataset_id"] == "sample_dataset"
    assert report["file_summary"]["total_files"] == 4
    assert report["file_summary"]["extension_counts"][".png"] == 2
    assert report["file_summary"]["extension_counts"][".json"] == 2

    assert report["pairing_suggestions"]
    best_pair = report["pairing_suggestions"][0]
    assert best_pair["reference_root"] == "images"
    assert best_pair["candidate_root"] == "labels"
    assert best_pair["matched_count"] == 2
    assert best_pair["match_score"] == 1.0

    discovery_dir = output_root / "sample_dataset_discovery"
    assert (discovery_dir / "discovery_report.json").exists()
    assert (discovery_dir / "suggested_config.json").exists()
    assert (discovery_dir / "file_inventory.csv").exists()
    assert (discovery_dir / "review_notes.md").exists()


def test_discover_dataset_detects_npz_dataset(tmp_path: Path):
    import numpy as np

    dataset_root = tmp_path / "npz_dataset"
    scene_dir = dataset_root / "scene1"
    scene_dir.mkdir(parents=True)

    np.savez(
        scene_dir / "sample_patch.npz",
        image=np.zeros((12, 32, 32), dtype="float32"),
        label=np.zeros((32, 32), dtype="uint8"),
    )

    output_root = tmp_path / "outputs"

    report = discover_dataset(
        dataset_root=dataset_root,
        dataset_id="npz_dataset",
        output_root=output_root,
        max_samples=1,
    )

    assert report["file_summary"]["total_files"] == 1
    assert report["file_summary"]["extension_counts"][".npz"] == 1
    assert "array_package_requires_npz_key_inspection" in report["task_hints"]

    npz_samples = report["sample_inspection"]["npz_samples"]
    assert len(npz_samples) == 1
    assert npz_samples[0]["keys"] == ["image", "label"]
    assert npz_samples[0]["arrays"]["image"]["shape"] == [12, 32, 32]
    assert npz_samples[0]["arrays"]["label"]["shape"] == [32, 32]
