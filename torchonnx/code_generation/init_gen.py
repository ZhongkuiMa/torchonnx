"""__init__ method code generation for PyTorch modules.

This module generates the __init__ method for converted PyTorch modules,
instantiating all layers with their constructor arguments and registering parameters.
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_init_method"]

from typing import Any

from onnx import TensorProto, numpy_helper

from ..ir import LayerIR
from ..type_inference import is_functional_operation, is_functional_operation_with_args


def format_constructor_args(args: dict[str, Any]) -> str:
    """Format constructor arguments as Python code string.

    :param args: Dictionary of constructor arguments
    :return: Formatted argument string for layer instantiation
    """
    if not args:
        return ""

    arg_strs: list[str] = []

    for key, value in args.items():
        if isinstance(value, bool):
            arg_strs.append(f"{key}={value}")
        elif isinstance(value, (int, float)):
            arg_strs.append(f"{key}={value}")
        elif isinstance(value, tuple):
            arg_strs.append(f"{key}={value}")
        elif isinstance(value, str):
            arg_strs.append(f'{key}="{value}"')
        else:
            arg_strs.append(f"{key}={repr(value)}")

    return ", ".join(arg_strs)


def generate_parameter_registrations(
    initializers: dict[str, TensorProto],
    name_mapping: dict[str, str],
    layers: list[LayerIR],
    used_params: set[str] | None = None,
) -> list[str]:
    """Generate parameter registration code for ONNX initializers.

    Only registers parameters that are NOT part of parametric layers and are actually used in forward().
    Parametric layers (Conv2d, BatchNorm2d, etc.) have their own internal
    parameters that are automatically created by PyTorch.

    :param initializers: ONNX initializers
    :param name_mapping: Mapping from ONNX names to simplified names
    :param layers: List of LayerIR to check which parameters belong to layers
    :param used_params: Set of parameter names actually used in forward() method
    :return: List of code lines for parameter registrations
    """
    from ..type_inference import is_parametric_layer

    lines: list[str] = []

    if not initializers or not name_mapping:
        return lines

    layer_params: set[str] = set()
    for layer in layers:
        if is_parametric_layer(layer.layer_type):
            for _, tensor_name in layer.parameters.items():
                layer_params.add(tensor_name)

    lines.append("")
    lines.append("# Register parameters")

    for onnx_name, simplified_name in name_mapping.items():
        if onnx_name not in initializers:
            continue

        if onnx_name in layer_params:
            continue

        # Only register if parameter is actually used in forward()
        if used_params is not None and simplified_name not in used_params:
            continue

        tensor_proto = initializers[onnx_name]
        numpy_array = numpy_helper.to_array(tensor_proto)
        shape = tuple(numpy_array.shape)

        shape_str = f"({shape})" if len(shape) == 0 else str(shape)
        param_line = f"self.{simplified_name} = nn.Parameter(torch.empty{shape_str})"
        lines.append(param_line)

    return lines


def generate_init_method(
    layers: list[LayerIR],
    initializers: dict[str, TensorProto] | None = None,
    name_mapping: dict[str, str] | None = None,
    used_params: set[str] | None = None,
) -> str:
    """Generate __init__ method with clean layer names and parameter registrations.

    Creates initialization code that instantiates all layers as
    nn.Module attributes using their clean layer names.
    Skips functional operations which are inlined in forward().
    Registers only ONNX initializers that are actually used in forward().

    Example output:
        def __init__(self):
            super().__init__()

            # Register parameters
            self.weight1 = nn.Parameter(torch.empty(5, 50))
            self.bias1 = nn.Parameter(torch.empty(50))

            self.flatten1 = nn.Flatten()
            self.relu1 = nn.ReLU()

    :param layers: List of LayerIR from Stage 1 compiler
    :param initializers: ONNX initializers dictionary
    :param name_mapping: Mapping from ONNX names to simplified names
    :param used_params: Set of parameter names actually used in forward() method
    :return: Python code string for __init__ method
    """
    lines: list[str] = [
        "def __init__(self):",
        '    """Initialize module."""',
        "    super().__init__()",
    ]

    if initializers and name_mapping:
        param_lines = generate_parameter_registrations(
            initializers, name_mapping, layers, used_params
        )
        lines.extend(
            f"    {line}" if line and not line.startswith("#") else f"    {line}"
            for line in param_lines
        )
        lines.append("")

    for layer in layers:
        if is_functional_operation(layer.layer_type):
            continue

        if is_functional_operation_with_args(layer.layer_type):
            continue

        args_str = format_constructor_args(layer.constructor_args)
        layer_def = f"    self.{layer.layer_name} = nn.{layer.layer_type}({args_str})"
        lines.append(layer_def)

    return "\n".join(lines)
