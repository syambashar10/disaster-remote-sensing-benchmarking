import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from disasterbench.loaders.loader_factory import get_loader, list_supported_datasets


def main():
    supported = list_supported_datasets()

    assert "sturm_flood" in supported
    assert "xbd" in supported

    sturm_loader = get_loader("sturm_flood", sensor="sentinel1")
    xbd_loader = get_loader("xbd", split="train")

    sturm_stats = sturm_loader.get_statistics()
    xbd_stats = xbd_loader.get_statistics()

    assert sturm_stats["dataset_id"] == "sturm_flood"
    assert xbd_stats["dataset_id"] == "xbd"

    sturm_sample = sturm_loader.load_sample(0)
    xbd_sample = xbd_loader.load_sample(0)

    assert sturm_sample["dataset_id"] == "sturm_flood"
    assert xbd_sample["dataset_id"] == "xbd"

    output = {
        "test_name": "loader_factory_test",
        "status": "passed",
        "supported_datasets": supported,
        "sturm_sample_id": sturm_sample["sample_id"],
        "xbd_sample_id": xbd_sample["sample_id"],
        "sturm_dataset_id": sturm_stats["dataset_id"],
        "xbd_dataset_id": xbd_stats["dataset_id"],
    }

    output_path = Path("outputs/verification/loader_factory_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("Loader factory test passed.")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
