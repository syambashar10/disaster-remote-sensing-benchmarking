from disasterbench.capabilities import (
    TargetFormat,
    check_dataset_export_capability,
    check_sample_export_capability,
    get_sample_representations,
)
from disasterbench.schemas import (
    AnnotationRecord,
    AnnotationType,
    CommonSample,
    LabelRecord,
    MediaRecord,
    MediaType,
)


def make_sample_with_representations(*representation_names):
    representations = {
        name: [0, 0, 10, 10] if name == "bbox" else [[0, 0], [1, 0], [1, 1]]
        for name in representation_names
    }

    return CommonSample(
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
                annotation_type=AnnotationType.POLYGON,
                label=LabelRecord(
                    original_label="building",
                    canonical_label="building",
                ),
                representations=representations,
            )
        ],
    )


def test_get_sample_representations_from_annotations():
    sample = make_sample_with_representations("bbox", "polygon")

    representations = get_sample_representations(sample)

    assert representations == {"bbox", "polygon"}


def test_yolo_detection_requires_bbox():
    sample = make_sample_with_representations("bbox")

    result = check_sample_export_capability(sample, TargetFormat.YOLO_DETECTION)

    assert result.supported
    assert "bbox" in result.available_representations
    assert result.missing_requirements == []


def test_yolo_segmentation_rejects_bbox_only_sample():
    sample = make_sample_with_representations("bbox")

    result = check_sample_export_capability(
        sample,
        TargetFormat.YOLO_SEGMENTATION,
    )

    assert not result.supported
    assert "polygon" in result.missing_requirements[0]
    assert "bbox" in result.available_representations


def test_coco_segmentation_accepts_polygon():
    sample = make_sample_with_representations("polygon")

    result = check_sample_export_capability(
        sample,
        TargetFormat.COCO_SEGMENTATION,
    )

    assert result.supported


def test_geojson_requires_geo_geometry():
    sample = make_sample_with_representations("polygon")

    result = check_sample_export_capability(sample, TargetFormat.GEOJSON)

    assert not result.supported
    assert "geo_polygon" in result.reason
    assert "geo_bbox" in result.reason


def test_mask_export_accepts_mask_media():
    sample = CommonSample(
        dataset_id="test_dataset",
        sample_id="sample_001",
        media=[
            MediaRecord(
                media_id="mask_001",
                media_type=MediaType.MASK,
                path="masks/mask_001.tif",
                width=128,
                height=128,
                bands=1,
            )
        ],
    )

    result = check_sample_export_capability(sample, TargetFormat.RASTER_MASK)

    assert result.supported
    assert "mask" in result.available_representations


def test_dataset_capability_fails_if_any_sample_lacks_requirement():
    good_sample = make_sample_with_representations("bbox")
    bad_sample = make_sample_with_representations("polygon")

    result = check_dataset_export_capability(
        [good_sample, bad_sample],
        TargetFormat.YOLO_DETECTION,
    )

    assert not result.supported
    assert "1 out of 2" in result.reason
