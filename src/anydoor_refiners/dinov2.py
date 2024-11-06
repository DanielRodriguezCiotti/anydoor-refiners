import refiners.fluxion.layers as fl
import torch
from refiners.foundationals.dinov2 import DINOv2_giant
from torch import device as Device, dtype as DType


class DINOv2Encoder(fl.Chain):
    def __init__(
        self,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        super().__init__(
            fl.Lambda(lambda x: (x - mean) / std),
            DINOv2_giant(device=device, dtype=dtype),
            fl.Linear(1536, 1024, device=device, dtype=dtype),
        )


if __name__ == "__main__":
    encoder = DINOv2Encoder()
    print("Model loaded successfully")
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        y = encoder(x)
        print(y.shape)
    print("Model run successfully")