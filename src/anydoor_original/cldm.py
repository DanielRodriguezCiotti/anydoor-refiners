import torch
import sys

sys.path.append("./AnyDoor/")
from cldm.model import create_model

model = create_model("../src/anydoor_original/configs/anydoor.yaml")
print("Model loaded successfully")

if __name__ == "__main__":

    with torch.no_grad():
        x = torch.randn(1, 4, 32, 32)
        timestep = torch.full((1,), 1, dtype=torch.long)
        mocked_control_image = torch.randn(1, 4, 32, 32)
        cond = {
            "c_crossattn": [torch.randn(1, 2, 1024)],
            "c_concat": [mocked_control_image],
        }
        y1 = model.apply_model(x, timestep, cond)
    print("Model run successfully")
