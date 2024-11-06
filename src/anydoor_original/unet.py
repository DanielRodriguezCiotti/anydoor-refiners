import torch
import sys

sys.path.append("./AnyDoor/")
from ldm.modules.diffusionmodules.openaimodel import UNetModel


model = UNetModel(
    image_size=32,
    in_channels=4,
    model_channels=320,
    out_channels=4,
    num_res_blocks=2,
    attention_resolutions=[4, 2, 1],
    dropout=0,
    channel_mult=[1, 2, 4, 4],
    conv_resample=True,
    dims=2,
    num_classes=None,
    use_checkpoint=True,
    use_fp16=False,
    num_heads=-1,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    resblock_updown=False,
    use_new_attention_order=False,
    use_spatial_transformer=True,
    transformer_depth=1,
    context_dim=1024,
    n_embed=None,
    legacy=False,
    disable_self_attentions=None,
    num_attention_blocks=None,
    disable_middle_self_attn=False,
    use_linear_in_transformer=True,
)
print("Model loaded successfully")
if __name__ == "__main__":
    with torch.no_grad():
        x = torch.randn(1, 4, 32, 32)
        timestep = torch.full((1,), 1, dtype=torch.long)
        object_embedding = torch.randn(1, 2, 1024)
        y1 = model.forward(x, timestep, object_embedding)
    print("Model run successfully")
