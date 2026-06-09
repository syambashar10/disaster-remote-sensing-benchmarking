from disasterbench.schemas import (
    AnnotationRecord,
    AnnotationType,
    CommonSample,
    LabelRecord,
    MediaRecord,
    MediaType,
    VerificationStatus,
    auto_verified,
    human_review_required,
    validate_common_sample,
)


def test_verification_record_distinguishes_auto_and_human_review():
    auto_record = auto_verified(
        name="file_count",
        description="File count was computed by full folder scan.",
    )
    human_record = human_review_required(
        name="label_mapping",
        description="Raw label meaning needs human approval.",
    )

    assert auto_record.status == VerificationStatus.AUTO_VERIFIED
    assert auto_record.is_usable_without_human_review()
    assert not auto_record.requires_human_review()

    assert human_record.status == VerificationStatus.HUMAN_REVIEW_REQUIRED
    assert human_record.requires_human_review()
    assert not human_record.is_usable_without_human_review()


def test_common_sample_serializes_nested_schema_records():
    media = MediaRecord(
        media_id="image_1",
        media_type=MediaType.IMAGE,
        path="images/sample.png",
        width=1024,
        height=1024,
        bands=3,
    )

    label = LabelRecord(
        original_label="no-damage",
        canonical_label="no_damage",
        taxonomy_name="building_damage",
        taxonomy_version="0.1.0",
        mapping_confidence=1.0,
    )

    annotation = AnnotationRecord(
        annotation_id="ann_1",
        annotation_type=AnnotationType.POLYGON,
        label=label,
        representations={
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "bbox": [0, 0, 10, 10],
        },
    )

    sample = CommonSample(
        dataset_id="xbd",
        sample_id="sample_1",
        split="train",
        task_types=["building_damage_assessment"],
        disaster_types=["wildfire"],
        media=[media],
        annotations=[annotation],
        source_files=["images/sample.png", "labels/sample.json"],
    )

    data = sample.to_dict()

    assert data["dataset_id"] == "xbd"
    assert data["sample_id"] == "sample_1"
    assert data["media"][0]["media_type"] == "image"
    assert data["annotations"][0]["annotation_type"] == "polygon"
    assert data["annotations"][0]["label"]["original_label"] == "no-damage"
    assert data["annotations"][0]["label"]["canonical_label"] == "no_damage"
    assert sample.has_representation("bbox")
    assert sample.annotation_count() == 1
    assert sample.media_count() == 1


def test_validate_common_sample_passes_for_valid_minimal_sample():
    sample = CommonSample(
        dataset_id="test_dataset",
        sample_id="sample_001",
        media=[
            MediaRecord(
                media_id="image_001",
                media_type=MediaType.IMAGE,
                path="images/image_001.png",
                width=256,
                height=256,
                bands=3,
            )
        ],
    )

    result = validate_common_sample(sample)

    assert result.passed
    assert result.issues == []


def test_validate_common_sample_warns_when_canonical_label_missing():
    sample = CommonSample(
        dataset_id="test_dataset",
        sample_id="sample_001",
        media=[
            MediaRecord(
                media_id="image_001",
                media_type=MediaType.IMAGE,
                path="images/image_001.png",
                width=256,
                height=256,
                bands=3,
            )
        ],
        annotations=[
            AnnotationRecord(
                annotation_id="ann_001",
                annotation_type=AnnotationType.BBOX,
                label=LabelRecord(
                    original_label="uncertain_raw_label",
                    canonical_label=None,
                ),
                representations={"bbox": [0, 0, 10, 10]},
            )
        ],
    )

    result = validate_common_sample(sample)

    assert result.passed
    assert any(issue.severity == "warning" for issue in result.issues)
    assert any("human review" in issue.message for issue in result.issues)
