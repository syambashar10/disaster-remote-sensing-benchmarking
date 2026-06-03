import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.exporters.geojson_exporter import export_sturm_flood_geojson


def test_geojson(sensor: str, max_samples: int):
    output_path = Path(f"outputs/geojson/sturm_flood_{sensor}_sample.geojson")

    result = export_sturm_flood_geojson(
        sensor=sensor,
        output_path=output_path,
        max_samples=max_samples,
        target_values=[1, 2, 3, 4, 5],
        min_area_pixels=2.0,
        coordinate_space="geo",
    )

    if not output_path.exists():
        raise AssertionError(f"GeoJSON output was not created: {output_path}")

    with output_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    assert geojson["type"] == "FeatureCollection"
    assert "features" in geojson
    assert geojson["metadata"]["dataset_id"] == "sturm_flood"
    assert geojson["metadata"]["sensor"] == sensor
    assert geojson["metadata"]["samples_exported"] == max_samples
    assert geojson["metadata"]["features_exported"] == len(geojson["features"])

    for feature in geojson["features"]:
        assert feature["type"] == "Feature"
        assert feature["geometry"]["type"] == "Polygon"
        assert "coordinates" in feature["geometry"]
        assert len(feature["geometry"]["coordinates"]) >= 1
        assert feature["properties"]["dataset_id"] == "sturm_flood"
        assert feature["properties"]["sensor"] == sensor

    return result


def main():
    output = {
        "sentinel1": test_geojson("sentinel1", max_samples=20),
        "sentinel2": test_geojson("sentinel2", max_samples=20),
    }

    output_path = Path("outputs/verification/sturm_flood_geojson_exporter_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("STURM-Flood GeoJSON exporter test passed.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
