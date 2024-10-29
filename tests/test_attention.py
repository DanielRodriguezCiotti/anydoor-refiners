import pytest
import torch
from anydoor_original.attention import SpatialTransformer
from src.anydoor_refiners.attention import CrossAttentionBlock2d
from refiners.conversion.model_converter import ModelConverter


# Fixture to set up and return both models with the specified configuration
@pytest.fixture
def setup_models():
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

    # Define input tensors
    input_channels = 320  # Must match the model's input channel configuration
    context_dim = 1024  # Must match the model's context dimension configuration
    input_tensor = torch.randn(1, input_channels, 32, 32)  # Example input tensor
    context_tensor = torch.randn(1, 1, context_dim)  # Example context tensor

    # Set the context for the CrossAttentionBlock2d model
    cross_attention_block.set_context("cross_attention_block", {"key": context_tensor})

    # Initialize a ModelConverter to handle the conversion and comparison of outputs
    converter = ModelConverter(
        source_model=spatial_transformer,
        target_model=cross_attention_block,
        threshold=0.001,  # Threshold for similarity comparison
    )

    # Run the converter and assert similarity within the threshold
    with torch.no_grad():
        assert converter.run(
            source_args={"x": input_tensor, "context": context_tensor},
            target_args=(input_tensor,),
        ), "Model outputs are not similar within the threshold"
