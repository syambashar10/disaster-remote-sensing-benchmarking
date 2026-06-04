import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.loaders.xbd_loader import XBDLoader
from disasterbench.exporters.xbd_geojson_exporter import export_xbd_geojson


def normalize_points(points):
    return [[round(float(x), 8), round(float(y), 8)] for x, y in points]


def main():
    max_samples = 50
    output_path = Path("outputs/geojson/xbd_integrity_sample.geojson")

    result = export_xbd_geojson(
        output_path=output_path,
        max_samples=max_samples,
        split="train",
    )

    loader = XBDLoader(split="train")

    expected_features = []

    for index in range(max_samples):
        sample = loader.load_sample(index)

        for annotation in sample["annotations"]:
            expected_features.append(
                {
                    "sample_id": sample["sample_id"],
                    "annotation_id": annotation["annotation_id"],
                    "uid": annotation["uid"],
                    "damage_class": annotation["damage_class"],
                    "pixel_bbox": [round(float(v), 6) for v in annotation["bbox"]],
                    "geo_polygon": normalize_points(annotation["geo_polygon"]),
                }
            )

    with output_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    exported_features = geojson["features"]

    assert result["features_exported"] == len(exported_features)
    assert len(exported_features) == len(expected_features)

    for expected, exported in zip(expected_features, exported_features):
        props = exported["properties"]
        coords = exported["geometry"]["coordinates"][0]

        exported_geo_polygon = normalize_points(coords)
        exported_bbox = [round(float(v), 6) for v in props["pixel_bbox"]]

        assert props["sample_id"] == expected["sample_id"]
        assert props["annotation_id"] == expected["annotation_id"]
        assert props["uid"] == expected["uid"]
        assert props["damage_class"] == expected["damage_class"]
        assert exported_bbox == expected["pixel_bbox"]
        assert exported_geo_polygon == expected["geo_polygon"]

    verification_output = {
        "test_name": "xbd_geojson_integrity_test",
        "status": "passed",
        "samples_checked": max_samples,
        "features_checked": len(expected_features),
        "export_output": str(output_path),
        "result": result,
    }

    verification_path = Path("outputs/verification/xbd_geojson_integrity_test.json")
    verification_path.parent.mkdir(parents=True, exist_ok=True)

    with verification_path.open("w", encoding="utf-8") as f:
        json.dump(verification_output, f, indent=2)

    print("xBD GeoJSON integrity test passed.")
    print(json.dumps(verification_output, indent=2))


if __name__ == "__main__":
    main()
