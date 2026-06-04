from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader
from disasterbench.loaders.xbd_loader import XBDLoader


SUPPORTED_DATASETS = {
    "sturm_flood": STURMFloodLoader,
    "xbd": XBDLoader,
}


def list_supported_datasets():
    return sorted(SUPPORTED_DATASETS.keys())


def get_loader(dataset_id: str, **kwargs):
    """
    Create a dataset loader by dataset ID.

    Examples:
    - get_loader("sturm_flood", sensor="sentinel1")
    - get_loader("sturm_flood", sensor="sentinel2")
    - get_loader("xbd", split="train")
    """
    dataset_id = dataset_id.lower()

    if dataset_id not in SUPPORTED_DATASETS:
        supported = ", ".join(list_supported_datasets())
        raise ValueError(f"Unsupported dataset_id '{dataset_id}'. Supported datasets: {supported}")

    loader_class = SUPPORTED_DATASETS[dataset_id]
    return loader_class(**kwargs)
