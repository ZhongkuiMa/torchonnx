"""State dict generation from semantic IR.

Builds PyTorch state_dict directly from ParameterInfo and ConstantInfo.
"""

__docformat__ = "restructuredtext"
__all__ = ["build_state_dict"]

import torch

from torchonnx.analyze import ConstantInfo, OperatorClass, ParameterInfo, SemanticModelIR

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


def _build_param_mappings(
    semantic_ir: SemanticModelIR,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], str]]:
    """Build parameter-to-layer mappings.

    :param semantic_ir: Semantic IR
    :return: Tuple of (onnx_to_layers, param_role_mapping)
    """
    onnx_to_layers: dict[str, list[str]] = {}
    param_role_mapping: dict[tuple[str, str], str] = {}

    for layer in semantic_ir.layers:
        if layer.operator_class != OperatorClass.LAYER:
            continue
        for inp in layer.inputs:
            if isinstance(inp, ParameterInfo):
                if inp.onnx_name not in onnx_to_layers:
                    onnx_to_layers[inp.onnx_name] = []
                onnx_to_layers[inp.onnx_name].append(layer.name)
                param_role_mapping[(inp.onnx_name, layer.name)] = inp.pytorch_name

    return onnx_to_layers, param_role_mapping


def _find_layer_for_constant(
    semantic_ir: SemanticModelIR, const_onnx_name: str
) -> tuple[str | None, str | None]:
    """Find layer that uses a constant, if any.

    :param semantic_ir: Semantic IR
    :param const_onnx_name: Constant ONNX name
    :return: Tuple of (layer_name, layer_type) or (None, None)
    """
    for layer in semantic_ir.layers:
        if layer.operator_class != OperatorClass.LAYER:
            continue
        for inp in layer.inputs:
            if isinstance(inp, ConstantInfo) and inp.onnx_name == const_onnx_name:
                return layer.name, layer.pytorch_type
    return None, None


def _add_parameters_to_dict(
    state_dict: dict[str, torch.Tensor],
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str],
    onnx_to_layers: dict[str, list[str]],
    param_role_mapping: dict[tuple[str, str], str],
) -> None:
    """Add parameters to state_dict.

    :param state_dict: State dict to update
    :param semantic_ir: Semantic IR
    :param layer_name_mapping: ONNX name -> clean name mapping
    :param onnx_to_layers: Parameter ONNX name -> list of layer names
    :param param_role_mapping: (ONNX name, layer name) -> PyTorch parameter name
    """
    for param in semantic_ir.parameters:
        if param.onnx_name in onnx_to_layers:
            for onnx_layer_name in onnx_to_layers[param.onnx_name]:
                clean_layer_name = layer_name_mapping.get(onnx_layer_name, onnx_layer_name)
                pytorch_name = param_role_mapping.get(
                    (param.onnx_name, onnx_layer_name), param.pytorch_name
                )
                key = f"{clean_layer_name}.{pytorch_name}"
                state_dict[key] = param.data
        else:
            state_dict[param.code_name] = param.data


def _add_constants_to_dict(
    state_dict: dict[str, torch.Tensor],
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str],
    used_constant_onnx_names: set[str],
) -> None:
    """Add constants (buffers) to state_dict.

    :param state_dict: State dict to update
    :param semantic_ir: Semantic IR
    :param layer_name_mapping: ONNX name -> clean name mapping
    :param used_constant_onnx_names: Set of constant ONNX names actually used
    """
    for const in semantic_ir.constants:
        if const.onnx_name not in used_constant_onnx_names:
            continue

        const_layer, const_layer_type = _find_layer_for_constant(semantic_ir, const.onnx_name)

        if const_layer and const_layer_type in _LAYERS_WITHOUT_BUFFERS:
            continue

        if const_layer:
            clean_layer_name = layer_name_mapping.get(const_layer, const_layer)
            key = f"{clean_layer_name}.{const.code_name}"
        else:
            key = const.code_name

        state_dict[key] = const.data


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

    if layer_name_mapping is None:
        layer_name_mapping = {}

    if used_constant_onnx_names is None:
        used_constant_onnx_names = {const.onnx_name for const in semantic_ir.constants}

    onnx_to_layers, param_role_mapping = _build_param_mappings(semantic_ir)
    _add_parameters_to_dict(
        state_dict, semantic_ir, layer_name_mapping, onnx_to_layers, param_role_mapping
    )
    _add_constants_to_dict(state_dict, semantic_ir, layer_name_mapping, used_constant_onnx_names)

    return state_dict
