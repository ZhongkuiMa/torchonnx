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


def _get_conv_pytorch_type(node: NodeProto, initializers: dict[str, TensorProto]) -> str:
    """Get PyTorch Conv type from weight shape.

    A dynamic weight (not a static initializer, e.g. spectral-normalized
    conv) cannot become an ``nn.Conv2d`` module since its kernel shape is
    unknown at construction time; emit the functional ``F.conv`` instead.
    """
    if len(node.input) >= 2 and node.input[1] in initializers:
        weight_tensor = initializers[node.input[1]]
        weight_array = numpy_helper.to_array(weight_tensor)
        weight_ndim = len(weight_array.shape)
        if weight_ndim == 3:
            return "nn.Conv1d"
        if weight_ndim == 4:
            return "nn.Conv2d"
        raise NotImplementedError(f"Unsupported Conv: {weight_ndim}D")
    return "F.conv"


def _get_convtranspose_pytorch_type(node: NodeProto, initializers: dict[str, TensorProto]) -> str:
    """Get PyTorch ConvTranspose type from weight shape.

    A dynamic weight (not a static initializer) cannot become an
    ``nn.ConvTranspose2d`` module; emit the functional ``F.conv_transpose``.
    """
    if len(node.input) >= 2 and node.input[1] in initializers:
        weight_tensor = initializers[node.input[1]]
        weight_array = numpy_helper.to_array(weight_tensor)
        weight_ndim = len(weight_array.shape)
        if weight_ndim == 3:
            return "nn.ConvTranspose1d"
        if weight_ndim == 4:
            return "nn.ConvTranspose2d"
        raise NotImplementedError(f"Unsupported ConvTranspose: {weight_ndim}D")
    return "F.conv_transpose"


def convert_to_pytorch_type(node: NodeProto, initializers: dict[str, TensorProto]) -> str:
    """Infer PyTorch layer type from ONNX node.

    :param node: ONNX node.
    :param initializers: Optional ONNX initializers for weight shape inspection.



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
        return convert_to_operator(node.op_type)
    if is_operation(node.op_type):
        return convert_to_operation(node.op_type)
    raise ValueError(f"Unsupported ONNX operator: {node.op_type}. ")
