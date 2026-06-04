import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.exporters.xbd_geojson_exporter import export_xbd_geojson


def main():
    output_path = Path("outputs/geojson/xbd_sample.geojson")

    result = export_xbd_geojson(
        output_path=output_path,
        max_samples=20,
        split="train",
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    assert geojson["type"] == "FeatureCollection"
    assert geojson["metadata"]["dataset_id"] == "xbd"
    assert geojson["metadata"]["coordinate_space"] == "lng_lat"
    assert geojson["metadata"]["samples_exported"] == 20
    assert geojson["metadata"]["features_exported"] == len(geojson["features"])

    assert len(geojson["features"]) > 0

    first_feature = geojson["features"][0]

    assert first_feature["type"] == "Feature"
    assert first_feature["geometry"]["type"] == "Polygon"
    assert "coordinates" in first_feature["geometry"]

    coords = first_feature["geometry"]["coordinates"][0]
    assert len(coords) >= 4

    lon, lat = coords[0]
    assert -180 <= lon <= 180
    assert -90 <= lat <= 90

    props = first_feature["properties"]
    assert props["dataset_id"] == "xbd"
    assert props["category"] == "building"
    assert "damage_class" in props
    assert "pixel_bbox" in props
    assert len(props["pixel_bbox"]) == 4

    verification_output = {
        "test_name": "xbd_geojson_exporter_test",
        "status": "passed",
        "result": result,
        "first_feature_properties": props,
    }

    verification_path = Path("outputs/verification/xbd_geojson_exporter_test.json")
    verification_path.parent.mkdir(parents=True, exist_ok=True)

    with verification_path.open("w", encoding="utf-8") as f:
        json.dump(verification_output, f, indent=2)

    print("xBD GeoJSON exporter test passed.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
