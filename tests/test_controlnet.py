import pytest
import torch
import json
from typing import Tuple
import sys
sys.path.append("./AnyDoor/")
from cldm.cldm import (
    ControlNet as AnyDoorControlNet,
)
from src.anydoor_refiners.controlnet import (
    ControlNet as RefinersControlNet,
)
from utils.weight_mapper import get_converted_state_dict


@pytest.fixture
def setup_models() -> Tuple[AnyDoorControlNet, RefinersControlNet]:
    """
    Sets up and returns both model instances with predefined configurations.

    Returns:
    --------
    Tuple[AnyDoorControlNet, RefinersControlNet]
        Tuple containing initialized instances of UNetModel and SD1UNet.
    """
    # Define UNetModel
    unet = AnyDoorControlNet(
        image_size=32,  # unused, but included for completeness
        in_channels=4,
        model_channels=320,
        hint_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        dropout=0,
        channel_mult=[1, 2, 4, 4],
        conv_resample=True,
        dims=2,
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

    # Define SD1UNet
    refiners_unet = RefinersControlNet(in_channels=4)

    return unet, refiners_unet


@pytest.fixture
def load_weight_mapping():
    """
    Loads the weights mapping JSON file for UNetModel and SD1UNet.

    Returns:
    --------
    Dict[str, str]
        Dictionary containing the layer mapping between UNetModel and SD1UNet.
    """
    with open("tests/weights_mapping/control_net.json", "r") as f:
        return json.load(f)


# Deactivate the following test because the models are not the same
@pytest.mark.skip
def test_parameter_count_match(setup_models):
    """
    Tests if both both models have the same number of trainable parameters.
    """
    anydoor, refiners = setup_models

    def count_parameters(model):
        return sum(param.numel() for param in model.parameters() if param.requires_grad)

    assert count_parameters(anydoor) == count_parameters(
        refiners
    ), "Trainable parameter counts do not match"


def test_model_output_similarity(setup_models, load_weight_mapping):
    """
    Tests if the output of both models are similar within a specified tolerance.
    """
    anydoor, refiners = setup_models
    weight_mapping = load_weight_mapping

    # Convert source state_dict to match target model's structure
    converted_state_dict = get_converted_state_dict(
        source_state_dict=anydoor.state_dict(),
        target_state_dict=refiners.state_dict(),
        mapping=weight_mapping,
    )
    refiners.load_state_dict(converted_state_dict)

    # Define inputs for testing
    x = torch.randn(1, 4, 512, 512)
    timestep = torch.full((1,), 1, dtype=torch.long)
    object_embedding = torch.randn(1, 257, 1024)

    with torch.no_grad():
        refiners.set_timestep(timestep)
        refiners.set_dinov2_object_embedding(object_embedding)
        y1 = refiners.forward(x)
        y2 = anydoor.forward(torch.zeros(1), x, timestep, object_embedding)

        for i,tensor in enumerate(y2):
            assert torch.norm(tensor-y1[i]) == 0, f"Model outputs are not similar within the threshold. Norm diff: {torch.norm(tensor-y1[i])} - {i} - Shapes: {tensor.shape} - {y1[i].shape}"
