from typing import Dict
import torch


def build_detailed_mapping(
    source_state_dict: Dict[str, torch.Tensor],
    target_state_dict: Dict[str, torch.Tensor],
    mapping: Dict[str, str],
) -> Dict[str, str]:
    """
    Builds a detailed mapping of layers in the source and target models by
    adding suffixes like ".weight" and ".bias" to the layer names where applicable.

    Parameters:
    -----------
    source_state_dict : Dict[str, torch.Tensor]
        The state dictionary of the source model.
    target_state_dict : Dict[str, torch.Tensor]
        The state dictionary of the target model.
    mapping : Dict[str, str]
        A dictionary containing the mapping between target and source model layer names.

    Returns:
    --------
    Dict[str, str]
        A dictionary with a detailed mapping of each parameter in source_state_dict to target_state_dict.

    Example:
    --------
    Input mapping:
    {
        "layer1": "layer2",
        "layer3": "layer4"
    }

    Output detailed mapping:
    {
        "layer1.weight": "layer2.weight",
        "layer1.bias": "layer2.bias",
        "layer3.weight": "layer4.weight",
        "layer3.bias": "layer4.bias"
    }
    """
    detailed_mapping = {}
    for source_key in source_state_dict.keys():
        # Split to get the base layer name and parameter suffix (e.g., ".weight" or ".bias")
        suffix = source_key.split(".")[-1]
        prefix = ".".join(source_key.split(".")[:-1])

        # Check if the prefix exists in the mapping dictionary
        if prefix not in mapping:
            print(f"[Warning] Layer '{prefix}' not found in mapping.")
            continue

        # Map the source prefix to the target layer name
        target_prefix = mapping[prefix]
        target_key = f"{target_prefix}.{suffix}"

        # Ensure target_key exists in the target model's state dict
        if target_key in target_state_dict:
            detailed_mapping[source_key] = target_key
        else:
            print(
                f"[Warning] '{target_key}' does not exist in target model's state dict."
            )

    return detailed_mapping


def get_converted_state_dict(
    source_state_dict: Dict[str, torch.Tensor],
    target_state_dict: Dict[str, torch.Tensor],
    mapping: Dict[str, str],
) -> Dict[str, torch.Tensor]:
    """
    Converts the source model's state dictionary to match the target model's structure,
    using the provided mapping to guide the conversion.

    Parameters:
    -----------
    source_state_dict : Dict[str, torch.Tensor]
        The state dictionary of the source model.
    target_state_dict : Dict[str, torch.Tensor]
        The state dictionary of the target model (used as a template for compatible shapes).
    mapping : Dict[str, str]
        A dictionary containing the mapping between target and source model layer names.

    Returns:
    --------
    Dict[str, torch.Tensor]
        A converted state dictionary that aligns with the target model's layer names.
    """
    # Reverse the mapping to get the source-to-target mapping
    mapping = {v: k for k, v in mapping.items()}
    # Build the detailed mapping between source and target layer names
    detailed_mapping = build_detailed_mapping(
        source_state_dict, target_state_dict, mapping
    )

    # Prepare a new state_dict for the target model with weights copied from the source model
    converted_state_dict = target_state_dict.copy()
    for source_key, target_key in detailed_mapping.items():
        # Copy weights from the source state dict to the target state dict
        converted_state_dict[target_key] = source_state_dict[source_key]

    return converted_state_dict
