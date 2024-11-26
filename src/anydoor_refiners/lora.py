from refiners.fluxion.adapters.lora import LinearLora, LoraAdapter
from anydoor_refiners.model import AnyDoor
import refiners.fluxion.layers as fl
import torch

def build_lora(model : AnyDoor, rank: int, scale : float) -> None:
    device = model.device
    dtype = model.dtype
    for dinov2_cross_attn_block in model.unet.layers(fl.Residual):
        if dinov2_cross_attn_block._get_name() == "DinoV2CrossAttention":
            dinov2_cross_attn_block_layers = {}
            dinov2_cross_attn_block_layers["proj_in"] = {'layer' : dinov2_cross_attn_block.layer(0,fl.Chain).layer(-1,fl.Linear), 'parent' : dinov2_cross_attn_block.layer(0,fl.Chain)}
            dinov2_cross_attn_block_layers["proj_out"] = {'layer' :dinov2_cross_attn_block.layer(-1,fl.Chain).layer(0,fl.Linear), 'parent' : dinov2_cross_attn_block.layer(-1,fl.Chain)}
            prefix_att = ["q","k","v"]
            for i,attention in enumerate(dinov2_cross_attn_block.layers(fl.Attention)):
                for j,linear in enumerate(attention.layer(-3,fl.Distribute).layers(fl.Linear)):
                    dinov2_cross_attn_block_layers[f"attn_{i}_{prefix_att[j]}"] = {'layer' : linear, 'parent' : attention.layer(-3,fl.Distribute)}
                dinov2_cross_attn_block_layers[f"attn_{i}_out"] = {'layer' :attention.layer(-1,fl.Linear), 'parent' : attention}
            dinov2_cross_attn_block_layers["ffn_1"] ={'layer' : dinov2_cross_attn_block.layer(-2,fl.Chain).layer(-1,fl.Chain).layer(-1,fl.Residual).layer(1,fl.Linear), 'parent' : dinov2_cross_attn_block.layer(-2,fl.Chain).layer(-1,fl.Chain).layer(-1,fl.Residual)}
            dinov2_cross_attn_block_layers["ffn_2"] = {'layer' :dinov2_cross_attn_block.layer(-2,fl.Chain).layer(-1,fl.Chain).layer(-1,fl.Residual).layer(-1,fl.Linear), 'parent' : dinov2_cross_attn_block.layer(-2,fl.Chain).layer(-1,fl.Chain).layer(-1,fl.Residual)}
                
            for key,layer_dict in dinov2_cross_attn_block_layers.items():
                adapter = LoraAdapter(layer_dict['layer'],LinearLora(key,layer_dict['layer'].in_features,layer_dict['layer'].out_features,rank=rank,scale=scale,device=device,dtype=dtype))
                adapter.inject(layer_dict['parent'])

def get_lora_weights(base: fl.Module) -> dict[str, torch.Tensor]:
    lora_state_dict = { k:v for k,v in base.state_dict().items() if "LinearLora" in k}
    return lora_state_dict

def set_lora_weights(base: fl.Module, lora_state_dict: dict[str, torch.Tensor]):
    state_dict = base.state_dict()
    for k,v in lora_state_dict.items():
        state_dict[k] = v
    base.load_state_dict(state_dict)
    