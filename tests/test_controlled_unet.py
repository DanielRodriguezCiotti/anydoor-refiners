import pytest
import torch
import json
from typing import Tuple
import sys
sys.path.append("./AnyDoor/")
from cldm.cldm import (
    ControlledUnetModel as AnyDoorControlledUnet,
)
from src.anydoor_refiners.unet import (
    UNet as RefinersControlledUnet,
)
from utils.weight_mapper import get_converted_state_dict


@pytest.fixture
def setup_models() -> Tuple[AnyDoorControlledUnet, RefinersControlledUnet]:
    """
    Sets up and returns both UNetModel and SD1UNet instances with predefined configurations.

    Returns:
    --------
    Tuple[UNetModel, SD1UNet]
        Tuple containing initialized UNetModel and SD1UNet models.
    """
    # Define UNetModel
    unet = AnyDoorControlledUnet(
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

    # Define SD1UNet
    refiners_unet = RefinersControlledUnet(in_channels=4)

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
    with open("tests/weights_mapping/unet.json", "r") as f:
        return json.load(f)


@pytest.fixture
def control_shapes():
    """
    Returns the shapes of the control tensors for each layer in the ControlledUNet model.

    Returns:
    --------
    List[List[int]]
        List containing the shapes of the control tensors for each layer in the ControlledUNet model.

    """
    return [
        [1, 320, 32, 32],
        [1, 320, 32, 32],
        [1, 320, 32, 32],
        [1, 320, 16, 16],
        [1, 640, 16, 16],
        [1, 640, 16, 16],
        [1, 640, 8, 8],
        [1, 1280, 8, 8],
        [1, 1280, 8, 8],
        [1, 1280, 4, 4],
        [1, 1280, 4, 4],
        [1, 1280, 4, 4],
        [1, 1280, 4, 4],
    ]


def test_parameter_count_match(setup_models):
    """
    Tests if both UNetModel and SD1UNet models have the same number of trainable parameters.
    """
    unet, refiners_unet = setup_models

    def count_parameters(model):
        return sum(param.numel() for param in model.parameters() if param.requires_grad)

    assert count_parameters(unet) == count_parameters(
        refiners_unet
    ), "Trainable parameter counts do not match"


def test_model_output_similarity(setup_models, load_weight_mapping, control_shapes):
    """
    Tests that UNetModel and SD1UNet produce similar outputs within a threshold after weight alignment.

    Converts UNetModel's state dictionary to match SD1UNet's structure and then verifies the output similarity.
    """
    anydoor_unet, refiners_unet = setup_models
    weight_mapping = load_weight_mapping

    # Convert source state_dict to match target model's structure
    converted_state_dict = get_converted_state_dict(
        source_state_dict=anydoor_unet.state_dict(),
        target_state_dict=refiners_unet.state_dict(),
        mapping=weight_mapping,
    )
    refiners_unet.load_state_dict(converted_state_dict)

    # Define inputs for testing
    x = torch.randn(1, 4, 32, 32)
    timestep = torch.full((1,), 961, dtype=torch.long)
    object_embedding = torch.randn(1, 10, 1024)
    control = [torch.randn(*shape) for shape in control_shapes]

    with torch.no_grad():
        # Set contexts for SD1UNet model
        refiners_unet.set_timestep(timestep)
        refiners_unet.set_dinov2_object_embedding(object_embedding)
        refiners_unet.set_control_residuals(control)
        # Forward pass on both models
        y1 = refiners_unet.forward(x)
        y2 = anydoor_unet.forward(x, timestep, object_embedding, control.copy())

        # Check similarity within a specified tolerance
        print(f"Norm diff: {torch.norm(y1 - y2)}")
        assert torch.norm(y1 - y2) == 0, f"Model outputs are not similar within the threshold. Norm diff: {torch.norm(y1 - y2)}"
