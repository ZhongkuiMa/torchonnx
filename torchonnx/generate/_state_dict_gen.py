"""State dict generation from semantic IR.

Builds PyTorch state_dict directly from ParameterInfo and ConstantInfo.
"""

__docformat__ = "restructuredtext"
__all__ = ["build_state_dict"]

import torch

from ..analyze import (
    SemanticModelIR,
    ParameterInfo,
    ConstantInfo,
    OperatorClass,
)

# PyTorch layers that don't have buffers - their constants are constructor args only
_LAYERS_WITHOUT_BUFFERS = {
    "Dropout",
    "nn.Dropout",
    "ReLU",
    "nn.ReLU",
    "Sigmoid",
    "nn.Sigmoid",
    "Tanh",
    "nn.Tanh",
    "Softmax",
    "nn.Softmax",
    "LeakyReLU",
    "nn.LeakyReLU",
    "ELU",
    "nn.ELU",
    "Flatten",
    "nn.Flatten",
    "Upsample",
    "nn.Upsample",
}


def build_state_dict(
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str] | None = None,
    used_constant_onnx_names: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build PyTorch state_dict from semantic IR.

    Maps parameters and constants to their state_dict keys:
    - Layer parameters: "{layer_name}.{pytorch_name}" (e.g., "conv1.weight")
    - Standalone parameters/constants: "{code_name}" (e.g., "p0", "c0")

    Handles shared parameters: if ONNX shares weights across multiple layers,
    each PyTorch layer still needs its own state_dict entry (they share values).

    Only includes constants that are actually used in the forward method.

    :param semantic_ir: Semantic IR from Stage 3
    :param layer_name_mapping: Optional mapping of ONNX name -> clean name
    :param used_constant_onnx_names: Set of constant ONNX names actually used in forward
    :return: PyTorch state_dict mapping
    """
    state_dict: dict[str, torch.Tensor] = {}

    # Default to empty mapping if not provided
    if layer_name_mapping is None:
        layer_name_mapping = {}

    # If no used_constants specified, assume all are used (backward compatibility)
    if used_constant_onnx_names is None:
        used_constant_onnx_names = {const.onnx_name for const in semantic_ir.constants}

    # Build mapping of onnx_name -> [list of layer_names] for layer parameters
    # Use list to handle shared parameters across multiple layers
    onnx_to_layers: dict[str, list[str]] = {}
    param_role_mapping: dict[tuple[str, str], str] = (
        {}
    )  # (onnx_name, layer_name) -> pytorch_name

    for layer in semantic_ir.layers:
        if layer.operator_class == OperatorClass.LAYER:
            # Track which parameters belong to this layer
            for inp in layer.inputs:
                if isinstance(inp, ParameterInfo):
                    if inp.onnx_name not in onnx_to_layers:
                        onnx_to_layers[inp.onnx_name] = []
                    onnx_to_layers[inp.onnx_name].append(layer.name)
                    param_role_mapping[(inp.onnx_name, layer.name)] = inp.pytorch_name

    # Add parameters to state_dict (handle shared parameters)
    for param in semantic_ir.parameters:
        if param.onnx_name in onnx_to_layers:
            # Layer parameter(s): add entry for EACH layer that uses this param
            for onnx_layer_name in onnx_to_layers[param.onnx_name]:
                clean_layer_name = layer_name_mapping.get(
                    onnx_layer_name, onnx_layer_name
                )
                pytorch_name = param_role_mapping.get(
                    (param.onnx_name, onnx_layer_name), param.pytorch_name
                )
                key = f"{clean_layer_name}.{pytorch_name}"
                state_dict[key] = param.data
        else:
            # Standalone parameter: use code_name
            key = param.code_name
            state_dict[key] = param.data

    # Add constants (buffers) to state_dict - only those actually used
    for const in semantic_ir.constants:
        # Skip constants not used in forward method
        if const.onnx_name not in used_constant_onnx_names:
            continue

        # Check if this constant belongs to a layer
        const_layer = None
        const_layer_type = None
        for layer in semantic_ir.layers:
            if layer.operator_class == OperatorClass.LAYER:
                for inp in layer.inputs:
                    if (
                        isinstance(inp, ConstantInfo)
                        and inp.onnx_name == const.onnx_name
                    ):
                        const_layer = layer.name
                        const_layer_type = layer.pytorch_type
                        break
                if const_layer:
                    break

        # Skip constants for layers that don't have buffers (e.g., Dropout)
        if const_layer and const_layer_type in _LAYERS_WITHOUT_BUFFERS:
            continue

        if const_layer:
            # Layer buffer: use cleaned layer_name.code_name
            clean_layer_name = layer_name_mapping.get(const_layer, const_layer)
            key = f"{clean_layer_name}.{const.code_name}"
        else:
            # Standalone buffer: use code_name
            key = const.code_name

        state_dict[key] = const.data

    return state_dict
