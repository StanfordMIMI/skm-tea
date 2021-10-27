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
