"""LAYER handlers for code generation.

Handlers for nn.Module layers (LAYER operator class).
Generate code like: x1 = self.conv1(x0)
"""

__docformat__ = "restructuredtext"
__all__ = ["register_layer_handlers"]

from torchonnx.analyze import SemanticLayerIR, VariableInfo
from torchonnx.generate._handlers._registry import register_handler


def _handle_generic_layer(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Generate code for generic layer operations.

    Produces: output = self.layer_name(input)

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    # Get input code names (should be VariableInfo)
    input_names = [inp.code_name for inp in layer.inputs if isinstance(inp, VariableInfo)]

    # Get output code name
    output_name = layer.outputs[0].code_name if layer.outputs else "x_out"

    # Get clean layer name from mapping
    clean_name = layer_name_mapping.get(layer.name, layer.name)

    # Generate call
    if len(input_names) == 1:
        return f"{output_name} = self.{clean_name}({input_names[0]})"
    # Multiple inputs (rare for layers)
    inputs_str = ", ".join(input_names)
    return f"{output_name} = self.{clean_name}({inputs_str})"


def _handle_batchnorm2d(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle BatchNorm2d operations with 3D input support.

    BatchNorm2d expects 4D input (N, C, H, W). For 3D ONNX inputs (N, C, L),
    reshape to (N, C, 1, L) using unsqueeze(2), then squeeze(2) back.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    # Get input code names (should be VariableInfo)
    input_names = [inp.code_name for inp in layer.inputs if isinstance(inp, VariableInfo)]

    if len(input_names) != 1:
        # BatchNorm should have exactly one variable input
        inputs_str = ", ".join(input_names) if input_names else "None"
        output_name = layer.outputs[0].code_name if layer.outputs else "x_out"
        clean_name = layer_name_mapping.get(layer.name, layer.name)
        return f"{output_name} = self.{clean_name}({inputs_str})"

    input_name = input_names[0]
    output_name = layer.outputs[0].code_name if layer.outputs else "x_out"
    clean_name = layer_name_mapping.get(layer.name, layer.name)

    # Get input shape if available
    input_info = next((inp for inp in layer.inputs if isinstance(inp, VariableInfo)), None)
    input_shape = input_info.shape if input_info and input_info.shape else None

    # Handle different input dimensions
    # - 2D (N, C): unsqueeze(2, 3) to (N, C, 1, 1), then squeeze(2, 3) back
    # - 3D (N, C, L): unsqueeze(2) to (N, C, 1, L), then squeeze(2) back
    # - 4D (N, C, H, W): direct call

    if input_shape and len(input_shape) == 2:
        # 2D input: add two spatial dimensions
        expr = f"{input_name}.unsqueeze(2).unsqueeze(3)"
        return f"{output_name} = self.{clean_name}({expr}).squeeze(2).squeeze(2)"
    if input_shape and len(input_shape) == 3:
        # 3D input: add one spatial dimension
        return f"{output_name} = self.{clean_name}({input_name}.unsqueeze(2)).squeeze(2)"
    if input_shape and len(input_shape) == 4:
        # 4D input: direct call
        return f"{output_name} = self.{clean_name}({input_name})"
    # Shape information missing - this should not happen with valid ONNX
    raise ValueError(
        f"Cannot determine input dimensionality statically for layer {layer.name}. "
        f"Missing shape information for input."
    )


def register_layer_handlers() -> None:
    """Register all LAYER handlers.

    Most layers use the generic pattern: output = self.layer_name(input)
    Some layers (like BatchNorm2d) need special handling for dimension mismatches.
    """
    layer_types = [
        # Convolution
        "nn.Conv1d",
        "nn.Conv2d",
        "nn.ConvTranspose1d",
        "nn.ConvTranspose2d",
        # Pooling
        "nn.MaxPool2d",
        "nn.AvgPool2d",
        "nn.AdaptiveAvgPool2d",
        # Linear
        "nn.Linear",
        # Activation functions
        "nn.ReLU",
        "nn.LeakyReLU",
        "nn.Sigmoid",
        "nn.Tanh",
        "nn.Softmax",
        "nn.ELU",
        "nn.GELU",
        # Dropout
        "nn.Dropout",
        # Upsampling
        "nn.Upsample",
        # Shape operations
        "nn.Flatten",
    ]

    for layer_type in layer_types:
        register_handler(layer_type, _handle_generic_layer)

    # Special handlers for layers that need dimension adjustment
    register_handler("nn.BatchNorm2d", _handle_batchnorm2d)
