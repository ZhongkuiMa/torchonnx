"""Parameter identification for PyTorch layers from ONNX nodes.

This module identifies which ONNX initializers belong to each PyTorch layer,
mapping parameter types (weight, bias, etc.) to ONNX initializer names.
"""

__docformat__ = "restructuredtext"
__all__ = ["identify_layer_parameters"]

from onnx import NodeProto, TensorProto


def identify_layer_parameters(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    layer_type: str,
) -> dict[str, str]:
    """Identify which ONNX initializers belong to this layer.

    Maps PyTorch parameter types to ONNX initializer names.
    For example: {"weight": "conv1_W", "bias": "conv1_B"}

    :param node: ONNX node
    :param initializers: All ONNX initializers
    :param layer_type: PyTorch layer type
    :return: Mapping from parameter type to ONNX initializer name
    """
    parameters = {}

    if layer_type in ("Conv1d", "Conv2d"):
        # Conv1d/Conv2d parameters: weight (required), bias (optional)
        # inputs[0] = input tensor
        # inputs[1] = weight
        # inputs[2] = bias (optional)
        if len(node.input) > 1 and node.input[1] in initializers:
            parameters["weight"] = node.input[1]

        if len(node.input) > 2 and node.input[2] in initializers:
            parameters["bias"] = node.input[2]

    elif layer_type in ("ConvTranspose1d", "ConvTranspose2d"):
        # ConvTranspose1d/ConvTranspose2d has same parameter structure as Conv
        if len(node.input) > 1 and node.input[1] in initializers:
            parameters["weight"] = node.input[1]

        if len(node.input) > 2 and node.input[2] in initializers:
            parameters["bias"] = node.input[2]

    elif layer_type == "Linear":
        # Linear (from Gemm or MatMul): weight (required), bias (optional)
        # inputs[0] = input tensor
        # inputs[1] = weight
        # inputs[2] = bias (optional)
        if len(node.input) > 1 and node.input[1] in initializers:
            parameters["weight"] = node.input[1]

        if len(node.input) > 2 and node.input[2] in initializers:
            parameters["bias"] = node.input[2]

    elif layer_type == "BatchNorm2d":
        # BatchNorm2d parameters:
        # inputs[0] = input tensor
        # inputs[1] = scale (gamma) -> weight
        # inputs[2] = bias (beta) -> bias
        # inputs[3] = running_mean
        # inputs[4] = running_var
        if len(node.input) > 1 and node.input[1] in initializers:
            parameters["weight"] = node.input[1]

        if len(node.input) > 2 and node.input[2] in initializers:
            parameters["bias"] = node.input[2]

        if len(node.input) > 3 and node.input[3] in initializers:
            parameters["running_mean"] = node.input[3]

        if len(node.input) > 4 and node.input[4] in initializers:
            parameters["running_var"] = node.input[4]

    elif layer_type == "Upsample":
        # Upsample inputs (used in constructor, not in state_dict):
        # inputs[0] = input tensor
        # inputs[1] = scales (optional, constructor only)
        # inputs[2] = sizes (optional, constructor only)
        # Mark as parameters to exclude from forward(), but these won't be in state_dict
        if len(node.input) > 1 and node.input[1] in initializers:
            parameters["scales"] = node.input[1]

        if len(node.input) > 2 and node.input[2] in initializers:
            parameters["sizes"] = node.input[2]

    # Other layer types (activations, pooling, etc.) have no parameters
    # Return empty dict for non-parametric layers

    return parameters
