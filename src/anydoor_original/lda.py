import torch
import sys
from omegaconf import OmegaConf

sys.path.append("./AnyDoor/")
from ldm.util import instantiate_from_config

conf = OmegaConf.load("src/anydoor_original/configs/anydoor.yaml")

model = instantiate_from_config(conf.model.params.first_stage_config)
print(model)