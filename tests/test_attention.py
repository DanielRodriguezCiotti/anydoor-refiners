import pytest
import torch
import json
from typing import Tuple
from src.anydoor_refiners.attention import CrossAttentionBlock2d
import sys
sys.path.append("./AnyDoor/")
from ldm.modules.attention import SpatialTransformer

from utils.weight_mapper import get_converted_state_dict


# Fixture to set up and return both models with the specified configuration
@pytest.fixture
def setup_models() -> Tuple[SpatialTransformer, CrossAttentionBlock2d]:
    # Define model configuration parameters with descriptive names
    input_channels = 320  # Number of input channels for the model
    num_heads = 5  # Number of attention heads
    head_dim = 64  # Dimension of each attention head
    num_layers = 1  # Depth of attention layers
    context_dim = 1024  # Dimension of the context embedding
    use_linear_projection = True  # Whether to use linear projection in attention

    # Initialize the SpatialTransformer model
    spatial_transformer = SpatialTransformer(
        in_channels=input_channels,
        n_heads=num_heads,
        d_head=head_dim,
        depth=num_layers,
        context_dim=context_dim,
        use_linear=use_linear_projection,
    )

    # Initialize the CrossAttentionBlock2d model
    cross_attention_block = CrossAttentionBlock2d(
        channels=input_channels,
        context_embedding_dim=context_dim,
        context_key="key",  # Key to set the context in cross_attention_block
        num_attention_heads=num_heads,
        num_attention_layers=num_layers,
        num_groups=32,  # Number of groups for grouped attention
        use_bias=False,
        use_linear_projection=use_linear_projection,
    )

    return spatial_transformer, cross_attention_block


# Helper function to count trainable parameters in a model
def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


# Test to check if both models have the same number of trainable parameters
def test_parameter_count_match(setup_models):
    spatial_transformer, cross_attention_block = setup_models

    # Assert that the trainable parameter counts of both models are equal
    assert count_parameters(spatial_transformer) == count_parameters(
        cross_attention_block
    ), "Trainable parameter counts do not match"


# Test to verify that both models produce similar outputs within a specified threshold
def test_model_output_similarity(setup_models):
    spatial_transformer, cross_attention_block = setup_models

    # Convert the source model's state dict to match the target model's structure
    with open("tests/weights_mapping/cross_attention_block_2d.json", "r") as f:
        weight_mapping = json.load(f)
    converted_state_dict = get_converted_state_dict(
        source_state_dict=spatial_transformer.state_dict(),
        target_state_dict=cross_attention_block.state_dict(),
        mapping=weight_mapping,
    )
    cross_attention_block.load_state_dict(converted_state_dict)

    # Define input tensors
    input_channels = 320  # Must match the model's input channel configuration
    context_dim = 1024  # Must match the model's context dimension configuration
    input_tensor = torch.randn(1, input_channels, 32, 32)  # Example input tensor
    context_tensor = torch.randn(1, 1, context_dim)  # Example context tensor

    with torch.no_grad():
        # Set the context for the CrossAttentionBlock2d model
        cross_attention_block.set_context(
            "cross_attention_block", {"key": context_tensor}
        )
        # Forward pass through both models
        y_target = cross_attention_block.forward(input_tensor)
        y_source = spatial_transformer.forward(input_tensor, context=context_tensor)
        # Assert that the model outputs are similar within a specified threshold
        assert torch.allclose(
            y_source, y_target, atol=1e-10
        ), "Model outputs are not similar within the threshold"
