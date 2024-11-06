import torch
from torch import  device as Device, dtype as DType
import refiners.fluxion.layers as fl


class DINOv2EncoderMock(fl.Module):
    def __init__(
        self,
        object_embedding: torch.Tensor,
        negative_object_embedding: torch.Tensor,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        super().__init__()
        self.embedding = object_embedding
        self.negative_embedding = negative_object_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.norm(x) == 0:
            return self.negative_embedding
        else:
            return self.embedding


class ControlNetMock(fl.Module):
    def __init__(
        self,
        control_features: list[torch.Tensor],
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        super().__init__()
        self.control = control_features

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.control
    
class AnydoorAutoencoderMock(fl.Identity):
    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        super().__init__()
        

