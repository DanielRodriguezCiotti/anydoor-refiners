import torch
import sys

sys.path.append("./AnyDoor/")
from cldm.model import create_model
from cldm.ddim_hacked import DDIMSampler


if __name__ == "__main__":
    model = create_model("src/anydoor_original/configs/anydoor.yaml")
    sampler = DDIMSampler(model)
    print("Model loaded successfully")
    with torch.no_grad():
        x = torch.randn(1, 4, 32, 32)
        timestep = torch.full((1,), 1, dtype=torch.long)
        mocked_control_image = torch.randn(1, 4, 32, 32)
        cond = {
            "c_concat": [mocked_control_image],
            "c_crossattn": [torch.randn(1, 2, 1024)],
        }
        un_cond = {
            "c_concat": [mocked_control_image],
            "c_crossattn": [torch.randn(1, 2, 1024)],
        }

        samples, intermediates = sampler.sample(
            S=50,
            batch_size=1,
            shape=(4, 32, 32),
            conditioning=cond,
            verbose=False,
            unconditional_guidance_scale=5.0,
            unconditional_conditioning=un_cond,
        )
    print("Model run successfully")
