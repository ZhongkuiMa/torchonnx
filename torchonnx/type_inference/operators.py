"""ONNX to PyTorch operator type mapping.

This module provides utilities for inferring PyTorch layer types from ONNX operators.
Reference: torchonnx/_torch_args.py _TORCH_ATTRS_MAP
"""

__docformat__ = "restructuredtext"
__all__ = [
    "infer_pytorch_layer_type",
    "is_parametric_layer",
    "is_functional_operation",
    "get_functional_operator",
]

from onnx import NodeProto


ONNX_TO_PYTORCH_TYPE: dict[str, str] = {
    # Pooling
    "MaxPool": "MaxPool2d",
    "AveragePool": "AvgPool2d",
    "GlobalAveragePool": "AdaptiveAvgPool2d",
    # Linear and matrix operations
    "Gemm": "Linear",
    # Normalization
    "BatchNormalization": "BatchNorm2d",
    # Activation functions
    "Relu": "ReLU",
    "LeakyRelu": "LeakyReLU",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Softmax": "Softmax",
    "Elu": "ELU",
    "Gelu": "GELU",
    # Dropout and regularization
    "Dropout": "Dropout",
    # Upsampling
    "Resize": "Upsample",
    "Upsample": "Upsample",
}


FUNCTIONAL_OPERATIONS: dict[str, str] = {
    "MatMul": "@",
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
    "Pow": "**",
    "Equal": "==",
}


PARAMETRIC_LAYERS: set[str] = {
    "Conv1d",
    "Conv2d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "Linear",
    "BatchNorm2d",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    "Dropout",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "ELU",
    "GELU",
}


def infer_pytorch_layer_type(node: NodeProto) -> str:
    """Infer PyTorch layer type from ONNX node.

    :param node: ONNX node
    :return: PyTorch layer type (e.g., "Conv2d", "Linear") or operation type
    """
    from .functional import is_functional_operation_with_args

    if node.op_type in ONNX_TO_PYTORCH_TYPE:
        return ONNX_TO_PYTORCH_TYPE[node.op_type]
    elif node.op_type in FUNCTIONAL_OPERATIONS:
        return node.op_type
    elif is_functional_operation_with_args(node.op_type):
        return node.op_type
    else:
        from .functional import FUNCTIONAL_OPERATIONS_WITH_ARGS

        all_supported = sorted(
            list(ONNX_TO_PYTORCH_TYPE.keys())
            + list(FUNCTIONAL_OPERATIONS.keys())
            + list(FUNCTIONAL_OPERATIONS_WITH_ARGS.keys())
        )
        raise ValueError(
            f"Unsupported ONNX operator: {node.op_type}. "
            f"Supported operators: {all_supported}"
        )


def is_parametric_layer(layer_type: str) -> bool:
    """Check if layer type has learnable parameters.

    :param layer_type: PyTorch layer type
    :return: True if layer has learnable parameters
    """
    return layer_type in PARAMETRIC_LAYERS


def is_functional_operation(layer_type: str) -> bool:
    """Check if operation is a functional operation (not a layer).

    :param layer_type: Operation type
    :return: True if operation is functional
    """
    return layer_type in FUNCTIONAL_OPERATIONS


def get_functional_operator(layer_type: str) -> str:
    """Get Python operator for functional operation.

    :param layer_type: Operation type
    :return: Python operator string (e.g., "+", "-", "@")
    """
    if layer_type not in FUNCTIONAL_OPERATIONS:
        raise ValueError(f"Not a functional operation: {layer_type}")
    return FUNCTIONAL_OPERATIONS[layer_type]
