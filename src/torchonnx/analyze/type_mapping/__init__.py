"""ONNX to PyTorch mapping utilities."""

__docformat__ = "restructuredtext"
__all__ = [
    "convert_to_operation",
    "convert_to_operator",
    "convert_to_pytorch_type",
    "extract_layer_args",
    "extract_operation_args",
    "is_layer_with_args",
    "is_operation",
    "is_operator",
]
from onnx import NodeProto, TensorProto, numpy_helper

from torchonnx.analyze.type_mapping._layers import (
    ONNX_TO_PYTORCH_LAYERS,
    extract_layer_args,
    is_layer_with_args,
)
from torchonnx.analyze.type_mapping._operations import (
    ONNX_TO_PYTORCH_OPERATIONS,
    convert_to_operation,
    convert_to_operator,
    extract_operation_args,
    is_operation,
    is_operator,
)


def _convert_to_operator_function(onnx_op_type: str) -> str:
    """Convert ONNX operator to PyTorch function name.

    :param onnx_op_type: ONNX operator type
    :return: PyTorch function name (e.g., "torch.add")
    """
    # Map common operators to torch.* functions
    operator_map = {
        "Add": "torch.add",
        "Sub": "torch.sub",
        "Mul": "torch.mul",
        "Div": "torch.div",
        "MatMul": "torch.matmul",
        "Pow": "torch.pow",
        "Neg": "torch.neg",
        "Equal": "torch.equal",
    }
    return operator_map.get(onnx_op_type, f"torch.{onnx_op_type.lower()}")


def _convert_to_operation_function(onnx_op_type: str) -> str:
    """Convert ONNX operation to PyTorch function name.

    :param onnx_op_type: ONNX operation type
    :return: PyTorch function name
    """
    # Get from the operations mapping
    return ONNX_TO_PYTORCH_OPERATIONS.get(onnx_op_type, onnx_op_type)


def _get_conv_pytorch_type(node: NodeProto, initializers: dict[str, TensorProto]) -> str:
    """Get PyTorch Conv type from weight shape."""
    if len(node.input) >= 2 and node.input[1] in initializers:
        weight_tensor = initializers[node.input[1]]
        weight_array = numpy_helper.to_array(weight_tensor)
        weight_ndim = len(weight_array.shape)
        if weight_ndim == 3:
            return "nn.Conv1d"
        if weight_ndim == 4:
            return "nn.Conv2d"
        raise NotImplementedError(f"Unsupported Conv: {weight_ndim}D")
    return "nn.Conv2d"


def _get_convtranspose_pytorch_type(node: NodeProto, initializers: dict[str, TensorProto]) -> str:
    """Get PyTorch ConvTranspose type from weight shape."""
    if len(node.input) >= 2 and node.input[1] in initializers:
        weight_tensor = initializers[node.input[1]]
        weight_array = numpy_helper.to_array(weight_tensor)
        weight_ndim = len(weight_array.shape)
        if weight_ndim == 3:
            return "nn.ConvTranspose1d"
        if weight_ndim == 4:
            return "nn.ConvTranspose2d"
        raise NotImplementedError(f"Unsupported ConvTranspose: {weight_ndim}D")
    return "nn.ConvTranspose2d"


def convert_to_pytorch_type(node: NodeProto, initializers: dict[str, TensorProto]) -> str:
    """Infer PyTorch layer type from ONNX node.

    :param node: ONNX node
    :param initializers: Optional ONNX initializers for weight shape inspection
    :return: PyTorch layer type (e.g., "Conv2d", "Linear") or operation type
    """
    if node.op_type == "Conv":
        return _get_conv_pytorch_type(node, initializers)
    if node.op_type == "ConvTranspose":
        return _get_convtranspose_pytorch_type(node, initializers)
    if (
        node.op_type == "Gemm"
        and node.op_type in ONNX_TO_PYTORCH_LAYERS
        and len(node.input) >= 2
        and node.input[1] not in initializers
    ):
        return "F.linear"
    if node.op_type in ONNX_TO_PYTORCH_LAYERS:
        layer_type = ONNX_TO_PYTORCH_LAYERS[node.op_type]
        return f"nn.{layer_type}"
    if is_operator(node.op_type):
        return _convert_to_operator_function(node.op_type)
    if is_operation(node.op_type):
        return _convert_to_operation_function(node.op_type)
    raise ValueError(f"Unsupported ONNX operator: {node.op_type}. ")
