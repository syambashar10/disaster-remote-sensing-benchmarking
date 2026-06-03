from disasterbench.loaders.sturm_flood_loader import STURMFloodLoader


def create_loader(dataset_id: str, **kwargs):
    dataset_id = dataset_id.lower()

    if dataset_id == "sturm_flood":
        return STURMFloodLoader(**kwargs)

    if dataset_id == "xbd":
        raise NotImplementedError(
            "XBDLoader is planned but not implemented in the main framework yet. "
            "The previous probation xBD work will be refactored into this loader."
        )

    raise ValueError(f"Unknown dataset_id: {dataset_id}")
