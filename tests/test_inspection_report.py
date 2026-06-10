import json

from disasterbench.inspection import (
    InspectionIssueSeverity,
    InspectionReport,
    build_inspection_summary_markdown,
    write_standard_inspection_outputs,
)
from disasterbench.schemas.verification import human_review_required


def test_inspection_report_serializes_core_fields():
    report = InspectionReport(
        dataset_id="xbd",
        dataset_root="/data/xbd",
        file_counts={".png": 10, ".json": 10},
        annotation_types_detected=["polygon"],
        class_values_detected=["no-damage", "destroyed"],
        metadata_fields_detected=["capture_date", "sensor"],
    )

    data = report.to_dict()

    assert data["dataset_id"] == "xbd"
    assert data["dataset_root"] == "/data/xbd"
    assert data["file_counts"][".png"] == 10
    assert "polygon" in data["annotation_types_detected"]


def test_blocking_issue_is_detected():
    report = InspectionReport(dataset_id="bad", dataset_root="/missing")
    report.add_issue(
        severity=InspectionIssueSeverity.ERROR,
        field="dataset_root",
        message="Dataset root does not exist.",
    )

    assert report.has_blocking_issues()


def test_human_review_issue_is_detected():
    report = InspectionReport(dataset_id="sturm_flood", dataset_root="/data/sturm")
    report.add_issue(
        severity=InspectionIssueSeverity.HUMAN_REVIEW_REQUIRED,
        field="mask_values",
        message="Mask value meanings require human approval.",
    )

    assert report.requires_human_review()
    assert not report.has_blocking_issues()


def test_verification_record_can_trigger_human_review_required():
    report = InspectionReport(dataset_id="sturm_flood", dataset_root="/data/sturm")
    report.verification.append(
        human_review_required(
            name="mask_value_meaning",
            description="Mask value meanings need mentor approval.",
        )
    )

    assert report.requires_human_review()


def test_build_inspection_summary_markdown_contains_key_sections():
    report = InspectionReport(
        dataset_id="xbd",
        dataset_root="/data/xbd",
        file_counts={".png": 10},
        annotation_types_detected=["polygon"],
    )

    markdown = build_inspection_summary_markdown(report)

    assert "# Inspection Summary: xbd" in markdown
    assert "## File Counts" in markdown
    assert "## Annotation Types Detected" in markdown
    assert ".png" in markdown


def test_write_standard_inspection_outputs(tmp_path):
    report = InspectionReport(
        dataset_id="xbd",
        dataset_root="/data/xbd",
        file_counts={".png": 10, ".json": 10},
    )

    outputs = write_standard_inspection_outputs(report, tmp_path)

    json_path = outputs["inspection_report_json"]
    md_path = outputs["inspection_summary_md"]

    assert json_path.exists()
    assert md_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    summary = md_path.read_text(encoding="utf-8")

    assert loaded["dataset_id"] == "xbd"
    assert "Inspection Summary" in summary
