import yaml

from disasterbench.schemas.verification import VerificationStatus
from disasterbench.taxonomy import (
    load_taxonomy_yaml,
    normalize_taxonomy_dict,
    validate_taxonomy_dict,
    validate_taxonomy_yaml,
)


def make_taxonomy(review_status=VerificationStatus.HUMAN_APPROVED.value):
    return {
        "taxonomy_id": "building_damage",
        "version": "0.1.0",
        "description": "Canonical building damage taxonomy.",
        "labels": {
            "no_damage": {
                "display_name": "No Damage",
                "description": "Building appears undamaged.",
            },
            "minor_damage": {
                "display_name": "Minor Damage",
                "description": "Building has minor visible damage.",
            },
            "destroyed": {
                "display_name": "Destroyed",
                "description": "Building is destroyed.",
            },
        },
        "mappings": [
            {
                "source_dataset": "xbd",
                "original_label": "no-damage",
                "canonical_label": "no_damage",
                "confidence": 1.0,
                "review_status": review_status,
                "evidence": ["docs/xbd_pipeline_status.md"],
            }
        ],
    }


def test_valid_human_approved_taxonomy_passes():
    taxonomy = make_taxonomy()

    result = validate_taxonomy_dict(taxonomy)

    assert result.passed
    assert not result.requires_human_review


def test_missing_required_taxonomy_fields_fails():
    taxonomy = {
        "taxonomy_id": "bad_taxonomy",
    }

    result = validate_taxonomy_dict(taxonomy)

    assert not result.passed
    assert any(issue.field == "version" for issue in result.issues)
    assert any(issue.field == "labels" for issue in result.issues)


def test_mapping_without_human_approval_requires_review():
    taxonomy = make_taxonomy(
        review_status=VerificationStatus.HUMAN_REVIEW_REQUIRED.value
    )

    result = validate_taxonomy_dict(taxonomy)

    assert result.passed
    assert result.requires_human_review
    assert any(
        issue.severity == "human_review_required"
        for issue in result.issues
    )


def test_invalid_confidence_fails():
    taxonomy = make_taxonomy()
    taxonomy["mappings"][0]["confidence"] = 1.5

    result = validate_taxonomy_dict(taxonomy)

    assert not result.passed
    assert any("confidence" in issue.field for issue in result.issues)


def test_unknown_canonical_label_fails():
    taxonomy = make_taxonomy()
    taxonomy["mappings"][0]["canonical_label"] = "not_defined"

    result = validate_taxonomy_dict(taxonomy)

    assert not result.passed
    assert any("canonical_label" in issue.field for issue in result.issues)


def test_ambiguous_mapping_requires_human_review():
    taxonomy = make_taxonomy()
    taxonomy["mappings"].append(
        {
            "source_dataset": "xbd",
            "original_label": "no-damage",
            "canonical_label": "minor_damage",
            "confidence": 0.5,
            "review_status": VerificationStatus.HUMAN_REVIEW_REQUIRED.value,
            "evidence": [],
        }
    )

    result = validate_taxonomy_dict(taxonomy)

    assert result.requires_human_review
    assert any("Ambiguous mapping" in issue.message for issue in result.issues)


def test_normalize_taxonomy_preserves_mapping_records():
    taxonomy = make_taxonomy()

    normalized = normalize_taxonomy_dict(taxonomy)

    assert normalized.taxonomy_id == "building_damage"
    assert normalized.version == "0.1.0"
    assert normalized.mappings[0].original_label == "no-damage"
    assert normalized.mappings[0].canonical_label == "no_damage"


def test_load_and_validate_taxonomy_yaml(tmp_path):
    taxonomy = make_taxonomy()
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(yaml.safe_dump(taxonomy), encoding="utf-8")

    loaded = load_taxonomy_yaml(taxonomy_path)
    result = validate_taxonomy_yaml(taxonomy_path)

    assert loaded["taxonomy_id"] == "building_damage"
    assert result.passed
    assert not result.requires_human_review
