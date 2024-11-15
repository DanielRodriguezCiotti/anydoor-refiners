import torch
import sys

sys.path.append("./AnyDoor/")
from ldm.modules.encoders.modules import FrozenDinoV2Encoder


if __name__ == "__main__":
    model = FrozenDinoV2Encoder(device="cpu", freeze=True)
    print("Model loaded successfully")
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        y = model.forward(x)
        print(y.shape)
    print("Model run successfully")
