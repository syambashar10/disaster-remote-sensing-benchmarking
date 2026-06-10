import json

from disasterbench.packaging import (
    create_output_package,
    create_standard_output_package,
    stable_json_hash,
    write_config_snapshot,
    write_run_manifest,
)


def test_stable_json_hash_is_order_independent():
    data_a = {"dataset_id": "xbd", "value": 1}
    data_b = {"value": 1, "dataset_id": "xbd"}

    assert stable_json_hash(data_a) == stable_json_hash(data_b)


def test_create_output_package_creates_standard_directories(tmp_path):
    package = create_output_package(
        dataset_id="xbd",
        output_root=tmp_path,
        run_id="xbd_test_run",
    )

    assert package.dataset_id == "xbd"
    assert package.run_id == "xbd_test_run"

    for dirname in ["inspection", "filtering", "exports", "catalog", "logs"]:
        assert dirname in package.directories
        assert (tmp_path / "xbd_test_run" / dirname).exists()


def test_write_config_snapshot_records_hash_and_config(tmp_path):
    package = create_output_package(
        dataset_id="xbd",
        output_root=tmp_path,
        run_id="xbd_test_run",
    )
    config = {
        "dataset_id": "xbd",
        "task_types": ["building_damage_assessment"],
    }

    snapshot_path = write_config_snapshot(package, config)

    assert snapshot_path.exists()
    assert package.config_snapshot_path == str(snapshot_path)

    loaded = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert loaded["dataset_id"] == "xbd"
    assert loaded["run_id"] == "xbd_test_run"
    assert loaded["config"] == config
    assert "config_hash_sha256" in loaded


def test_write_run_manifest_records_package_metadata(tmp_path):
    package = create_output_package(
        dataset_id="sturm_flood",
        output_root=tmp_path,
        run_id="sturm_test_run",
    )

    manifest_path = write_run_manifest(
        package,
        extra_metadata={"stage": "test"},
    )

    assert manifest_path.exists()
    assert package.run_manifest_path == str(manifest_path)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert loaded["dataset_id"] == "sturm_flood"
    assert loaded["run_id"] == "sturm_test_run"
    assert loaded["extra_metadata"]["stage"] == "test"
    assert "inspection" in loaded["directories"]


def test_create_standard_output_package_writes_all_metadata(tmp_path):
    config = {
        "dataset_id": "xbd",
        "human_review_status": "human_review_required",
    }

    package = create_standard_output_package(
        dataset_id="xbd",
        output_root=tmp_path,
        config=config,
        run_id="xbd_standard_run",
        extra_metadata={"created_by": "unit_test"},
    )

    assert package.config_snapshot_path is not None
    assert package.run_manifest_path is not None

    assert (tmp_path / "xbd_standard_run" / "config_snapshot.json").exists()
    assert (tmp_path / "xbd_standard_run" / "run_manifest.json").exists()
    assert (tmp_path / "xbd_standard_run" / "catalog").exists()
