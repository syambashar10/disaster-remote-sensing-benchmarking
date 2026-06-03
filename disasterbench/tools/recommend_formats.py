import argparse
import json
from pathlib import Path


def load_registry(registry_path="configs/dataset_registry.json"):
    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def recommend_formats(dataset_id, registry_path="configs/dataset_registry.json"):
    registry = load_registry(registry_path)
    datasets = registry.get("datasets", {})

    if dataset_id not in datasets:
        available = ", ".join(datasets.keys())
        raise ValueError(f"Unknown dataset '{dataset_id}'. Available datasets: {available}")

    info = datasets[dataset_id]

    print(f"\nDataset: {info.get('name')}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Raw annotation type: {info.get('raw_annotation_type')}")
    print(f"Tasks: {', '.join(info.get('task_type', []))}")

    print("\nRecommended formats:")
    for fmt in info.get("recommended_formats", []):
        print(f"- {fmt}")

    lossy_formats = info.get("lossy_formats", [])
    if lossy_formats:
        print("\nLossy but possible formats:")
        for fmt in lossy_formats:
            print(f"- {fmt}")

    unsupported = info.get("unsupported_formats", [])
    if unsupported:
        print("\nUnsupported formats/tasks:")
        for fmt in unsupported:
            print(f"- {fmt}")

    print(f"\nStatus: {info.get('status')}\n")


def main():
    parser = argparse.ArgumentParser(description="Recommend export formats for a dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset ID, for example: xbd or sturm_flood")
    args = parser.parse_args()

    recommend_formats(args.dataset)


if __name__ == "__main__":
    main()
