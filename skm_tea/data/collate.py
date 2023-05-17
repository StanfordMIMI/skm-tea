from torch.utils.data.dataloader import default_collate as _default_collate

__all__ = ["default_collate"]


def default_collate(batch: list):
    metadata = None
    instances = None
    if any("metadata" in b for b in batch):
        metadata = [b.pop("metadata", None) for b in batch]
    if any("instances" in b for b in batch):
        instances = [b.pop("instances", None) for b in batch]
    out_dict = _default_collate(batch)
    if metadata is not None:
        out_dict["metadata"] = metadata
    if instances is not None:
        out_dict["instances"] = instances
    return out_dict


def collate_by_supervision(batch: list):
    """Collate supervised/unsupervised batch examples.

    This collate function is required when training with semi-supervised
    models, such as :cls:`N2RModel` and :cls:`VortexModel`.

    Args:
        batch (list): The list of dictionaries.

    Returns:
        Dict[str, Dict]: A dictionary with 2 keys, ``'supervised'`` and ``'unsupervised'``.
    """
    # return default_collate(batch)
    profiler = [x.pop("_profiler", None) for x in batch]
    profiler = [x for x in profiler if x is not None]

    supervised = [x for x in batch if not x["is_unsupervised"]]
    unsupervised = [x for x in batch if x["is_unsupervised"]]

    out_dict = {}
    if len(supervised) > 0:
        supervised = default_collate(supervised)
        out_dict["supervised"] = supervised
    if len(unsupervised) > 0:
        unsupervised = default_collate(unsupervised)
        out_dict["unsupervised"] = unsupervised
    assert len(out_dict) > 0

    out_dict["_profiler"] = _default_collate(profiler)
    return out_dict
