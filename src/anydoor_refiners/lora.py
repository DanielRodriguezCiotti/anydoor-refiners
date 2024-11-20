from refiners.fluxion.adapters.lora import LinearLora, LoraAdapter
from anydoor_refiners.model import AnyDoor
import refiners.fluxion.layers as fl
import torch

def build_lora(model : AnyDoor, rank: int, scale : float) -> None:
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
                adapter = LoraAdapter(layer_dict['layer'],LinearLora(key,layer_dict['layer'].in_features,layer_dict['layer'].out_features,rank=rank,scale=scale))
                adapter.inject(layer_dict['parent'])

def get_lora_weights(base: fl.Chain) -> dict[str, torch.Tensor]:
    prev_parent: fl.Chain | None = None
    lora_weights: dict[str, torch.Tensor] = {}
    n = 0
    for lora_adapter, parent in base.walk(LoraAdapter):
        for lora in lora_adapter.lora_layers :
        # lora = next((l for l in lora_adapter.lora_layers if l.name == name), None)
            if lora is None:
                continue
            n = (parent == prev_parent) and n + 1 or 1
            pfx = f"{parent.get_path()}.{n}.{lora_adapter.target.__class__.__name__}"
            lora_weights[f"{pfx}.down.weight"] = lora.down.weight
            lora_weights[f"{pfx}.up.weight"] = lora.up.weight
            prev_parent = parent
    return lora_weights


def load_lora_weights(base: fl.Chain, weights: dict[str, torch.Tensor]) -> None:
    prev_parent: fl.Chain | None = None
    n = 0
    for lora_adapter, parent in base.walk(LoraAdapter):
        for lora in lora_adapter.lora_layers :
            if lora is None:
                continue
            n = (parent == prev_parent) and n + 1 or 1
            pfx = f"{parent.get_path()}.{n}.{lora_adapter.target.__class__.__name__}"
            lora.down.weight = weights[f"{pfx}.down.weight"]
            lora.up.weight = weights[f"{pfx}.up.weight"]
            prev_parent = parent
    