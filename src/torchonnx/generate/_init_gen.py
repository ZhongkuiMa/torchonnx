"""Generate __init__ method from semantic IR.

Creates parameter registration, buffer registration, and layer instantiation code.
"""

__docformat__ = "restructuredtext"
__all__ = ["build_layer_name_mapping", "generate_init_method"]

import torch

from torchonnx.analyze import (
    ConstantInfo,
    OperatorClass,
    ParameterInfo,
    SemanticLayerIR,
    SemanticModelIR,
)
from torchonnx.generate._templates import INDENT, INIT_TEMPLATE
from torchonnx.generate._utils import format_argument

# PyTorch dtype to string representation
_DTYPE_TO_STR = {
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.float16: "torch.float16",
    torch.int64: "torch.int64",
    torch.int32: "torch.int32",
    torch.int16: "torch.int16",
    torch.int8: "torch.int8",
    torch.uint8: "torch.uint8",
    torch.bool: "torch.bool",
}


def _extract_layer_base_type(pytorch_type: str) -> str:
    """Extract base name from PyTorch type.

    Examples:
        'nn.Conv2d' -> 'conv2d'
        'nn.ReLU' -> 'relu'
        'nn.BatchNorm2d' -> 'batchnorm2d'
        'nn.Flatten' -> 'flatten'

    :param pytorch_type: Full PyTorch type string
    :return: Lowercase base name with digits preserved
    """
    # Get the class name: "nn.Conv2d" -> "Conv2d"
    name = pytorch_type.split(".")[-1]
    # Keep the full name including digits
    return name.lower()


def build_layer_name_mapping(semantic_ir: SemanticModelIR) -> dict[str, str]:
    """Build mapping from ONNX layer names to clean Python names.

    Generates names like 'conv1', 'relu2', 'flatten1' based on layer type.

    :param semantic_ir: Semantic IR from Stage 3
    :return: Mapping of ONNX name -> clean Python name
    """
    layer_counters: dict[str, int] = {}
    name_mapping: dict[str, str] = {}

    for layer in semantic_ir.layers:
        if layer.operator_class == OperatorClass.LAYER:
            base_type = _extract_layer_base_type(layer.pytorch_type)
            layer_counters[base_type] = layer_counters.get(base_type, 0) + 1
            clean_name = f"{base_type}{layer_counters[base_type]}"
            name_mapping[layer.name] = clean_name

    return name_mapping


def _build_layer_io_sets(semantic_ir: SemanticModelIR) -> tuple[set[str], set[str]]:
    """Build sets of parameters and constants that belong to layers.

    :param semantic_ir: Semantic IR from Stage 3
    :return: Tuple of (layer_params, layer_consts)
    """
    layer_params: set[str] = set()
    layer_consts: set[str] = set()

    for layer in semantic_ir.layers:
        if layer.operator_class == OperatorClass.LAYER:
            for inp in layer.inputs:
                if isinstance(inp, ParameterInfo):
                    layer_params.add(inp.onnx_name)
                elif isinstance(inp, ConstantInfo):
                    layer_consts.add(inp.onnx_name)

    return layer_params, layer_consts


def _register_standalone_parameters(
    lines: list[str],
    semantic_ir: SemanticModelIR,
    layer_params: set[str],
) -> None:
    """Register standalone parameters (not used by any layer).

    :param lines: List to append registration lines to
    :param semantic_ir: Semantic IR from Stage 3
    :param layer_params: Set of parameter ONNX names used by layers
    """
    lines.extend(
        [
            f"{INDENT}{INDENT}self.register_parameter("
            f'"{param.code_name}", '
            f"nn.Parameter(torch.empty({list(param.shape)})))"
            for param in semantic_ir.parameters
            if param.onnx_name not in layer_params
        ]
    )


def _register_standalone_buffers(
    lines: list[str],
    semantic_ir: SemanticModelIR,
    layer_consts: set[str],
    used_constant_onnx_names: set[str],
) -> None:
    """Register standalone buffer constants (not used by any layer).

    :param lines: List to append registration lines to
    :param semantic_ir: Semantic IR from Stage 3
    :param layer_consts: Set of constant ONNX names used by layers
    :param used_constant_onnx_names: Set of constant ONNX names used in forward
    """
    for const in semantic_ir.constants:
        if const.onnx_name not in layer_consts and const.onnx_name in used_constant_onnx_names:
            dtype_str = _DTYPE_TO_STR.get(const.dtype, "torch.float32")
            lines.append(
                f"{INDENT}{INDENT}self.register_buffer("
                f'"{const.code_name}", '
                f"torch.empty({list(const.shape)}, dtype={dtype_str}))"
            )


def _instantiate_layers(
    lines: list[str],
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str],
) -> None:
    """Instantiate all layer objects with clean names.

    :param lines: List to append instantiation lines to
    :param semantic_ir: Semantic IR from Stage 3
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    """
    for layer in semantic_ir.layers:
        if layer.operator_class == OperatorClass.LAYER:
            # Build argument string from layer.arguments
            args_str = _format_layer_arguments(layer)

            # Get clean layer name from mapping
            clean_name = layer_name_mapping.get(layer.name, layer.name)

            # Generate layer instantiation
            lines.append(f"{INDENT}{INDENT}self.{clean_name} = {layer.pytorch_type}({args_str})")


def generate_init_method(
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str] | None = None,
    used_constant_onnx_names: set[str] | None = None,
) -> str:
    """Generate __init__ method code.

    Generates:
    1. Parameter registration for standalone parameters
    2. Buffer registration for standalone constants (only those used in forward)
    3. Layer instantiation for LAYER operators

    :param semantic_ir: Semantic IR from Stage 3
    :param layer_name_mapping: Optional mapping of ONNX name -> clean name
    :param used_constant_onnx_names: Set of constant ONNX names actually used in forward
    :return: Complete __init__ method code
    """
    lines: list[str] = []

    # Build layer name mapping if not provided
    if layer_name_mapping is None:
        layer_name_mapping = build_layer_name_mapping(semantic_ir)

    # If no used_constants specified, assume all are used (backward compatibility)
    if used_constant_onnx_names is None:
        used_constant_onnx_names = {const.onnx_name for const in semantic_ir.constants}

    # Build sets of parameters/constants that belong to layers
    layer_params, layer_consts = _build_layer_io_sets(semantic_ir)

    # Register standalone parameters
    _register_standalone_parameters(lines, semantic_ir, layer_params)

    # Register standalone buffers (constants) - only those actually used
    _register_standalone_buffers(lines, semantic_ir, layer_consts, used_constant_onnx_names)

    # Instantiate layers with clean names
    _instantiate_layers(lines, semantic_ir, layer_name_mapping)

    # Assemble init method
    body = "\n".join(lines) if lines else f"{INDENT}{INDENT}pass"

    return INIT_TEMPLATE.format(
        indent=INDENT,
        body=body,
    )


def _format_layer_arguments(layer: SemanticLayerIR) -> str:
    """Format layer constructor arguments.

    :param layer: Semantic layer IR
    :return: Formatted argument string
    """
    args: list[str] = []

    for arg in layer.arguments:
        formatted_value = format_argument(arg.value)
        if arg.pytorch_name:
            # Named argument
            args.append(f"{arg.pytorch_name}={formatted_value}")
        else:
            # Positional argument
            args.append(formatted_value)

    return ", ".join(args)
