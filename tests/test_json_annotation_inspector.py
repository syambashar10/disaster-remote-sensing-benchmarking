import json
import subprocess
import sys

from disasterbench.inspection.json_annotation_inspector import inspect_json_annotations


def test_json_annotation_inspector_detects_json_structure(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    (labels_dir / "sample_001.json").write_text(
        json.dumps(
            {
                "features": {
                    "xy": [
                        {
                            "wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))",
                            "properties": {
                                "feature_type": "building",
                                "subtype": "destroyed",
                            },
                        }
                    ],
                    "lng_lat": [],
                },
                "metadata": {
                    "width": 1024,
                    "height": 1024,
                },
            }
        ),
        encoding="utf-8",
    )

    (labels_dir / "empty.json").write_text("{}", encoding="utf-8")
    (labels_dir / "bad.json").write_text("{bad json", encoding="utf-8")

    result = inspect_json_annotations(labels_dir)

    assert result.total_json_files == 3
    assert result.scanned_json_files == 3
    assert result.valid_json_files == 2
    assert result.invalid_json_files == 1
    assert result.empty_json_content_files == 1

    assert result.top_level_key_counts["features"] == 1
    assert result.annotation_signal_counts["xbd_like_features_xy_list"] == 1
    assert result.annotation_signal_counts["xbd_like_features_lng_lat_list"] == 1
    assert result.geometry_like_key_counts["wkt"] == 1
    assert result.class_like_key_counts["feature_type"] == 1
    assert result.class_like_key_counts["subtype"] == 1
    assert len(result.read_errors) == 1


def test_json_annotation_inspector_supports_max_files(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    for index in range(5):
        (labels_dir / f"sample_{index}.json").write_text(
            json.dumps({"annotations": []}),
            encoding="utf-8",
        )

    result = inspect_json_annotations(labels_dir, max_files=2)

    assert result.total_json_files == 5
    assert result.scanned_json_files == 2
    assert result.scan_limited is True
    assert result.annotation_signal_counts["coco_like_annotations_list"] == 2


def test_json_annotation_inspector_missing_root_returns_error(tmp_path):
    missing_dir = tmp_path / "missing"

    result = inspect_json_annotations(missing_dir)

    assert result.total_json_files == 0
    assert result.scanned_json_files == 0
    assert result.valid_json_files == 0
    assert result.invalid_json_files == 0
    assert len(result.read_errors) == 1


def test_inspect_json_annotations_cli_writes_output(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    (labels_dir / "sample.json").write_text(
        json.dumps({"annotations": [{"bbox": [0, 0, 10, 10], "category_id": 1}]}),
        encoding="utf-8",
    )

    output_path = tmp_path / "json_inspection.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "disasterbench.tools.inspect_json_annotations",
            "--json-root",
            str(labels_dir),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout_data = json.loads(completed.stdout)
    file_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert stdout_data["total_json_files"] == 1
    assert file_data["total_json_files"] == 1
    assert file_data["annotation_signal_counts"]["coco_like_annotations_list"] == 1
    assert file_data["geometry_like_key_counts"]["bbox"] == 1
    assert file_data["class_like_key_counts"]["category_id"] == 1


def test_json_annotation_inspector_counts_full_lists_by_default(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    records = [
        {"wkt": f"POLYGON (({i} 0, {i+1} 0, {i+1} 1, {i} 0))", "label": "building"}
        for i in range(75)
    ]

    (labels_dir / "large_list.json").write_text(
        json.dumps({"features": {"xy": records}}),
        encoding="utf-8",
    )

    result = inspect_json_annotations(labels_dir)

    assert result.geometry_like_key_counts["wkt"] == 75
    assert result.class_like_key_counts["label"] == 75
    assert result.list_truncation_counts == {}


def test_json_annotation_inspector_can_limit_nested_list_traversal(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    records = [
        {"wkt": f"POLYGON (({i} 0, {i+1} 0, {i+1} 1, {i} 0))", "label": "building"}
        for i in range(75)
    ]

    (labels_dir / "large_list.json").write_text(
        json.dumps({"features": {"xy": records}}),
        encoding="utf-8",
    )

    result = inspect_json_annotations(labels_dir, max_list_items_per_list=50)

    assert result.geometry_like_key_counts["wkt"] == 50
    assert result.class_like_key_counts["label"] == 50
    assert result.list_truncation_counts["features.xy"] == 1


def test_json_annotation_inspector_counts_class_like_values(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    (labels_dir / "sample.json").write_text(
        json.dumps(
            {
                "features": {
                    "xy": [
                        {"properties": {"subtype": "destroyed", "feature_type": "building"}},
                        {"properties": {"subtype": "minor-damage", "feature_type": "building"}},
                        {"properties": {"subtype": "destroyed", "feature_type": "building"}},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = inspect_json_annotations(labels_dir)

    assert result.value_inventory_counts["features.xy.properties.subtype"] == {
        "destroyed": 2,
        "minor-damage": 1,
    }
    assert result.value_inventory_counts["features.xy.properties.feature_type"] == {
        "building": 3,
    }


def test_json_annotation_inspector_counts_numeric_label_values(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    (labels_dir / "sample.json").write_text(
        json.dumps(
            {
                "annotations": [
                    {"category_id": 1, "bbox": [0, 0, 10, 10]},
                    {"category_id": 2, "bbox": [0, 0, 10, 10]},
                    {"category_id": 1, "bbox": [0, 0, 10, 10]},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = inspect_json_annotations(labels_dir)

    assert result.value_inventory_counts["annotations.category_id"] == {
        "1": 2,
        "2": 1,
    }
