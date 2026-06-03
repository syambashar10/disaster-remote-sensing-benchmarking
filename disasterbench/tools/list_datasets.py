import json
from pathlib import Path


def load_registry(registry_path="configs/dataset_registry.json"):
    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    registry = load_registry()
    datasets = registry.get("datasets", {})

    print("\nAvailable datasets:\n")

    for dataset_id, info in datasets.items():
        print(f"- {dataset_id}")
        print(f"  Name: {info.get('name')}")
        print(f"  Raw annotation type: {info.get('raw_annotation_type')}")
        print(f"  Tasks: {', '.join(info.get('task_type', []))}")
        print(f"  Recommended formats: {', '.join(info.get('recommended_formats', []))}")

        lossy_formats = info.get("lossy_formats", [])
        if lossy_formats:
            print(f"  Lossy but possible formats: {', '.join(lossy_formats)}")

        print(f"  Status: {info.get('status')}")
        print()


if __name__ == "__main__":
    main()
