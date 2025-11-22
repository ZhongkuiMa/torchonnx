"""Layer naming utilities for generating clean PyTorch layer names.

This module generates clean, meaningful layer names (conv1, bn2, fc3) instead
of using complex ONNX node names like /model/layer1/conv/Conv or onnx::Conv_123.
"""

__docformat__ = "restructuredtext"
__all__ = ["sanitize_layer_name"]

from onnx import NodeProto


PREFIX_MAP: dict[str, str] = {
    "Conv1d": "conv",
    "Conv2d": "conv",
    "ConvTranspose1d": "convtranspose",
    "ConvTranspose2d": "convtranspose",
    "Linear": "fc",
    "BatchNorm2d": "bn",
    "MaxPool2d": "pool",
    "AvgPool2d": "avgpool",
    "AdaptiveAvgPool2d": "adaptiveavgpool",
    "ReLU": "relu",
    "LeakyReLU": "leakyrelu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softmax": "softmax",
    "Dropout": "dropout",
    "Flatten": "flatten",
    "Reshape": "reshape",
    "Transpose": "transpose",
    "Concat": "concat",
    "Add": "add",
    "Sub": "sub",
    "Mul": "mul",
    "Div": "div",
}


def get_layer_prefix(layer_type: str) -> str:
    """Get short prefix for PyTorch layer type.

    :param layer_type: PyTorch layer type (e.g., "Conv2d", "Linear")
    :return: Short prefix (e.g., "conv", "fc")
    """
    return PREFIX_MAP.get(layer_type, layer_type.lower())


def sanitize_layer_name(
    node: NodeProto | None,
    layer_type: str,
    counter: dict[str, int],
) -> str:
    """Generate clean, meaningful layer name.

    Converts complex ONNX node names to clean Python identifiers:
    - /model/layer1/conv/Conv → conv1
    - onnx::Gemm_123 → fc1
    - model.0.bn → bn1

    :param node: ONNX node (unused, kept for API consistency)
    :param layer_type: PyTorch layer type (e.g., "Conv2d", "Linear")
    :param counter: Counter dict for each layer type prefix
    :return: Clean layer name (e.g., "conv1", "bn2", "fc3")
    """
    prefix = get_layer_prefix(layer_type)

    if prefix not in counter:
        counter[prefix] = 1
    else:
        counter[prefix] += 1

    layer_name = f"{prefix}{counter[prefix]}"

    assert layer_name.isidentifier(), f"Generated invalid identifier: {layer_name}"

    return layer_name


def validate_layer_name(name: str) -> bool:
    """Validate that layer name is a valid Python identifier without ONNX patterns.

    :param name: Layer name to validate
    :return: True if valid, False otherwise
    """
    if not name.isidentifier():
        return False

    if "/" in name or "::" in name or "." in name:
        return False

    return True
