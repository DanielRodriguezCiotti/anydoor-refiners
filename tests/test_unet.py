import pytest
import torch
import json
from typing import Tuple
from src.anydoor_original.unet import UNetModel
from src.anydoor_refiners.unet import SD1UNet
from utils.weight_mapper import get_converted_state_dict


@pytest.fixture
def setup_models() -> Tuple[UNetModel, SD1UNet]:
    """
    Sets up and returns both UNetModel and SD1UNet instances with predefined configurations.

    Returns:
    --------
    Tuple[UNetModel, SD1UNet]
        Tuple containing initialized UNetModel and SD1UNet models.
    """
    # Define UNetModel
    unet = UNetModel(
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
    refiners_unet = SD1UNet(in_channels=4)

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


def test_model_output_similarity(setup_models, load_weight_mapping):
    """
    Tests that UNetModel and SD1UNet produce similar outputs within a threshold after weight alignment.

    Converts UNetModel's state dictionary to match SD1UNet's structure and then verifies the output similarity.
    """
    unet, refiners_unet = setup_models
    weight_mapping = load_weight_mapping

    # Convert source state_dict to match target model's structure
    converted_state_dict = get_converted_state_dict(
        source_state_dict=unet.state_dict(),
        target_state_dict=refiners_unet.state_dict(),
        mapping=weight_mapping,
    )
    refiners_unet.load_state_dict(converted_state_dict)

    # Define inputs for testing
    x = torch.randn(1, 4, 32, 32)
    timestep = torch.full((1,), 1, dtype=torch.long)
    timestep_embedding = torch.randn(1, 1, 1024)

    with torch.no_grad():
        # Set contexts for SD1UNet model
        refiners_unet.set_context("diffusion", {"timestep": timestep})
        refiners_unet.set_context(
            "cross_attention_block", {"dinov2_garment_embedding": timestep_embedding}
        )

        # Forward pass on both models
        y1 = unet.forward(x, timestep, timestep_embedding)
        y2 = refiners_unet.forward(x)

        # Check similarity within a specified tolerance
        assert torch.allclose(
            y1, y2, atol=1e-6
        ), f"Model outputs are not similar within the threshold. Norm diff: {torch.norm(y1 - y2)}"
