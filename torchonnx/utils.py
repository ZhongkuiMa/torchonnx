__docformat__ = "restructuredtext"
__all__ = [
    "DTYPE_NUMPY2TORCH",
    "DTYPE_ONNX2TORCH",
    "EXTRACT_ATTR_MAP",
    "get_tensor_shape",
    "onnx_tensor_to_pytorch",
    "get_attribute_value",
    "reformat_io_shape",
    "parse_input_names",
    "parse_output_names",
    "get_initializers",
    "initializer_to_tensor",
    "initializer_to_list",
    "initializer_to_tuple",
    "initializer_to_int",
    "extract_opset_version",
]

import numpy as np
import onnx
import torch
from onnx import (
    AttributeProto,
    ValueInfoProto,
    numpy_helper,
    ModelProto,
    NodeProto,
    TensorProto,
)
from torch import Tensor

DTYPE_NUMPY2TORCH = {
    np.dtypes.Int32DType: torch.int32,
    np.dtypes.Int64DType: torch.int64,
    np.dtypes.Float32DType: torch.float32,
    np.dtypes.Float64DType: torch.float64,
}

DTYPE_ONNX2TORCH = {
    1: torch.float32,
    2: torch.uint8,
    3: torch.int8,
    4: None,  # torch.uint16 not available in PyTorch 2.5.1
    5: torch.int16,
    6: torch.int32,
    7: torch.int64,
    8: None,
    9: torch.bool,
    10: torch.float16,
    11: torch.double,
    12: None,  # torch.uint32 not available in PyTorch 2.5.1
    13: None,  # torch.uint64 not available in PyTorch 2.5.1
    14: torch.complex64,
    15: torch.complex128,
}

EXTRACT_ATTR_MAP = {
    0: lambda x: None,  # UNDEFINED
    1: lambda x: x.f,  # FLOAT
    2: lambda x: x.i,  # INT
    3: lambda x: x.s.decode("utf-8"),  # STRING
    4: lambda x: onnx.numpy_helper.to_array(x.t),  # TENSOR
    5: lambda x: x.g,  # GRAPH
    6: lambda x: tuple(x.floats),  # FLOATS
    7: lambda x: tuple(x.ints),  # INTS
    8: lambda x: None,  # STRINGS
    9: lambda x: None,  # TENSORS
    10: lambda x: None,  # GRAPHS
    11: lambda x: None,  # SPARSE_TENSOR
}


def parse_input_names(
    node: NodeProto, initializer: dict[str, TensorProto]
) -> list[str]:
    input_names = []
    for name in node.input:
        if name in initializer:
            name = f'self.data["{name}"]'
        input_names.append(name)
    return input_names


def parse_output_names(
    node: NodeProto, initializer: dict[str, TensorProto]
) -> list[str]:
    output_names = []
    for name in node.output:
        if name in initializer:
            name = f'self.data["{name}"]'
        output_names.append(name)
    return output_names


def get_initializers(model: ModelProto) -> dict[str, TensorProto]:
    return {initializer.name: initializer for initializer in model.graph.initializer}


def initializer_to_tensor(initializer: TensorProto) -> Tensor:
    return torch.tensor(onnx.numpy_helper.to_array(initializer))


def initializer_to_list(initializer: TensorProto) -> list:
    return onnx.numpy_helper.to_array(initializer).tolist()


def initializer_to_tuple(initializer: TensorProto) -> tuple:
    return tuple(onnx.numpy_helper.to_array(initializer).tolist())


def initializer_to_int(initializer: TensorProto) -> int:
    return int(onnx.numpy_helper.to_array(initializer))


def get_tensor_shape(tensor: TensorProto) -> list[int]:
    """Extract shape from ONNX tensor.

    :param tensor: ONNX TensorProto
    :return: Shape as list of integers
    """
    return list(tensor.dims)


def onnx_tensor_to_pytorch(tensor: TensorProto) -> torch.Tensor:
    """Convert ONNX tensor to PyTorch tensor.

    :param tensor: ONNX TensorProto
    :return: PyTorch tensor
    """
    numpy_array = numpy_helper.to_array(tensor)
    return torch.from_numpy(numpy_array)


def get_attribute_value(
    attr: AttributeProto,
) -> int | float | str | list[int] | list[float]:
    """Extract value from ONNX attribute.

    :param attr: ONNX AttributeProto
    :return: Attribute value in appropriate Python type
    """
    if attr.type == AttributeProto.INT:
        return attr.i
    elif attr.type == AttributeProto.FLOAT:
        return attr.f
    elif attr.type == AttributeProto.STRING:
        return attr.s.decode("utf-8")
    elif attr.type == AttributeProto.INTS:
        return list(attr.ints)
    elif attr.type == AttributeProto.FLOATS:
        return list(attr.floats)
    elif attr.type == AttributeProto.TENSOR:
        return numpy_helper.to_array(attr.t)
    else:
        raise ValueError(f"Unsupported attribute type: {attr.type}")


def reformat_io_shape(node: ValueInfoProto) -> list[int]:
    """Extract shape from ONNX ValueInfo (input/output node).

    Handles batch dimension normalization:
    - If first dimension is 0, set it to 1
    - Ensures batch dimension is present and set to 1

    Reference: torchonnx/test/check_torch_model.py:12-24

    :param node: ONNX ValueInfoProto
    :return: Shape as list of integers with normalized batch dimension
    """
    shape = [d.dim_value for d in node.type.tensor_type.shape.dim]

    if len(shape) == 0:
        return shape

    if shape[0] == 0:
        shape[0] = 1
    elif len(shape) > 1:
        if shape[0] != 1:
            shape = [1] + shape

    return shape


def extract_opset_version(model: ModelProto) -> int:
    """Extract ONNX opset version from model.

    :param model: ONNX model
    :return: Opset version
    """
    if not model.opset_import:
        raise ValueError("Model has no opset_import")

    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            return opset.version

    raise ValueError("Model has no primary opset (domain='' or 'ai.onnx')")
