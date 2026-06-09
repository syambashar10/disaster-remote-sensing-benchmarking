import json

from disasterbench.configs import (
    load_dataset_config,
    normalize_dataset_config,
    validate_dataset_config_dict,
    validate_dataset_config_file,
)
from disasterbench.schemas.verification import VerificationStatus


def make_minimal_config(human_review_status=VerificationStatus.NOT_VERIFIED.value):
    return {
        "dataset_id": "xbd",
        "display_name": "xBD",
        "dataset_root": "/data/xbd",
        "task_types": ["building_damage_assessment"],
        "disaster_types": ["wildfire"],
        "native_annotation_types": ["polygon"],
        "supported_exports": ["coco_detection", "yolo_detection"],
        "unsupported_exports": ["raster_mask"],
        "lossy_conversions": [],
        "label_mappings": {
            "no-damage": {
                "canonical_label": "no_damage",
                "review_status": human_review_status,
            }
        },
        "human_review_status": human_review_status,
    }


def test_config_missing_required_fields_fails():
    config = {
        "dataset_id": "bad_config",
    }

    result = validate_dataset_config_dict(config)

    assert not result.passed
    assert any(issue.severity == "error" for issue in result.issues)
    assert any(issue.field == "display_name" for issue in result.issues)


def test_config_with_semantic_fields_requires_human_review_until_approved():
    config = make_minimal_config(
        human_review_status=VerificationStatus.HUMAN_REVIEW_REQUIRED.value
    )

    result = validate_dataset_config_dict(config)

    assert result.passed
    assert result.requires_human_review
    assert any(
        issue.severity == "human_review_required"
        for issue in result.issues
    )


def test_human_approved_config_passes_without_review_flag():
    config = make_minimal_config(
        human_review_status=VerificationStatus.HUMAN_APPROVED.value
    )

    result = validate_dataset_config_dict(config)

    assert result.passed
    assert not result.requires_human_review


def test_invalid_list_field_fails():
    config = make_minimal_config()
    config["task_types"] = "building_damage_assessment"

    result = validate_dataset_config_dict(config)

    assert not result.passed
    assert any(issue.field == "task_types" for issue in result.issues)


def test_label_mapping_without_canonical_label_requires_review():
    config = make_minimal_config()
    config["label_mappings"] = {
        "mystery_label": {
            "review_status": VerificationStatus.NOT_VERIFIED.value,
        }
    }

    result = validate_dataset_config_dict(config)

    assert result.requires_human_review
    assert any("canonical_label" in issue.message for issue in result.issues)


def test_normalize_dataset_config_preserves_raw_config():
    config = make_minimal_config(
        human_review_status=VerificationStatus.HUMAN_APPROVED.value
    )

    normalized = normalize_dataset_config(config)

    assert normalized.dataset_id == "xbd"
    assert normalized.display_name == "xBD"
    assert normalized.task_types == ["building_damage_assessment"]
    assert normalized.raw_config["dataset_id"] == "xbd"


def test_load_and_validate_dataset_config_file(tmp_path):
    config = make_minimal_config(
        human_review_status=VerificationStatus.HUMAN_APPROVED.value
    )
    config_path = tmp_path / "dataset_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_dataset_config(config_path)
    result = validate_dataset_config_file(config_path)

    assert loaded["dataset_id"] == "xbd"
    assert result.passed
    assert not result.requires_human_review
