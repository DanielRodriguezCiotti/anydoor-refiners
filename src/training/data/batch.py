from dataclasses import dataclass
import torch


@dataclass
class AnyDoorBatch:
    filename : list[str]
    object : torch.Tensor
    background : torch.Tensor
    collage : torch.Tensor
    background_box : torch.Tensor
    sizes : torch.Tensor
    background_image : torch.Tensor | None = None
    loss_mask : torch.Tensor | None = None


def collate_fn(batch: list) -> AnyDoorBatch | None:
    # Filter out None entries
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    final_batch = {}
    for key in batch[0].keys():
        if key == "filename":
            final_batch[key] = [item[key] for item in batch]
        else:
            final_batch[key] = torch.stack([item[key] for item in batch])
    return AnyDoorBatch(**final_batch)