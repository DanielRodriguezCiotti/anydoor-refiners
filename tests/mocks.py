from typing import Any
import torch
from torch import  device as Device, dtype as DType
import refiners.fluxion.layers as fl
from refiners.foundationals.latent_diffusion.auto_encoder import (
    LatentDiffusionAutoencoder,
)

from anydoor_refiners.controlnet import ControlNet

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


class ControlNetMock(ControlNet):
    def __init__(
        self,
        control_features: list[torch.Tensor],
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        # Initialize the fl.Module class
        fl.Module.__init__(self)
        self.control = control_features

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.control
    
    def set_timestep(self, timestep):
        return None
    def set_dinov2_object_embedding(self, dinov2_object_embedding):
        return None
    
class AnydoorAutoencoderMock(LatentDiffusionAutoencoder):
    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ):
        fl.Module.__init__(self)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
        

