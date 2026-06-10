import csv
import json

from disasterbench.filtering import (
    FilterDecision,
    FilterManifest,
    FilterManifestEntry,
    FilterReason,
    write_standard_filter_manifest_outputs,
)
from disasterbench.schemas.verification import VerificationStatus


def test_filter_manifest_counts_decisions():
    manifest = FilterManifest(
        dataset_id="xbd",
        filter_config_id="test_filter_v1",
    )

    manifest.add_entry(
        FilterManifestEntry(
            sample_id="sample_001",
            decision=FilterDecision.INCLUDED,
            reasons=[
                FilterReason(
                    code="passes_all_filters",
                    message="Sample passed all configured filters.",
                )
            ],
        )
    )
    manifest.add_entry(
        FilterManifestEntry(
            sample_id="sample_002",
            decision=FilterDecision.EXCLUDED,
            reasons=[
                FilterReason(
                    code="missing_annotation",
                    message="Sample has no annotation.",
                )
            ],
        )
    )

    counts = manifest.decision_counts()

    assert counts["included"] == 1
    assert counts["excluded"] == 1
    assert counts["quarantined"] == 0


def test_filter_manifest_tracks_included_and_excluded_ids():
    manifest = FilterManifest(
        dataset_id="xbd",
        filter_config_id="test_filter_v1",
        entries=[
            FilterManifestEntry(
                sample_id="included_001",
                decision=FilterDecision.INCLUDED,
            ),
            FilterManifestEntry(
                sample_id="excluded_001",
                decision=FilterDecision.EXCLUDED,
            ),
        ],
    )

    assert manifest.included_sample_ids() == ["included_001"]
    assert manifest.excluded_sample_ids() == ["excluded_001"]


def test_filter_manifest_human_review_detection():
    manifest = FilterManifest(
        dataset_id="sturm_flood",
        filter_config_id="duplicate_filter_v1",
        entries=[
            FilterManifestEntry(
                sample_id="sample_001",
                decision=FilterDecision.HUMAN_REVIEW_REQUIRED,
                reasons=[
                    FilterReason(
                        code="possible_duplicate",
                        message="Duplicate status is ambiguous.",
                    )
                ],
                verification_status=VerificationStatus.HUMAN_REVIEW_REQUIRED.value,
            )
        ],
    )

    assert manifest.requires_human_review()


def test_filter_manifest_to_dict_serializes_enums():
    manifest = FilterManifest(
        dataset_id="xbd",
        filter_config_id="test_filter_v1",
        entries=[
            FilterManifestEntry(
                sample_id="sample_001",
                decision=FilterDecision.QUARANTINED,
                reasons=[
                    FilterReason(
                        code="quality_issue",
                        message="Image quality score is below threshold.",
                        field="quality.blur_score",
                        value=0.2,
                    )
                ],
            )
        ],
    )

    data = manifest.to_dict()

    assert data["dataset_id"] == "xbd"
    assert data["entries"][0]["decision"] == "quarantined"
    assert data["entries"][0]["reasons"][0]["code"] == "quality_issue"


def test_write_standard_filter_manifest_outputs(tmp_path):
    manifest = FilterManifest(
        dataset_id="xbd",
        filter_config_id="test_filter_v1",
        entries=[
            FilterManifestEntry(
                sample_id="sample_001",
                decision=FilterDecision.INCLUDED,
                reasons=[
                    FilterReason(
                        code="passes_all_filters",
                        message="Sample passed all filters.",
                    )
                ],
                source_files=["images/sample_001.png", "labels/sample_001.json"],
            )
        ],
    )

    outputs = write_standard_filter_manifest_outputs(manifest, tmp_path)

    json_path = outputs["filter_manifest_json"]
    csv_path = outputs["filter_manifest_csv"]

    assert json_path.exists()
    assert csv_path.exists()

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["decision_counts"]["included"] == 1

    with csv_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["sample_id"] == "sample_001"
    assert rows[0]["decision"] == "included"
