__docformat__ = "restructuredtext"
__all__ = [
    "EXTRACT_ATTR_MAP",
    "parse_input_names",
    "parse_output_names",
    "get_initializers",
    "initializer_to_tensor",
    "initializer_to_list",
    "initializer_to_tuple",
    "initializer_to_int",
]

import onnx
import torch
from onnx import ModelProto, NodeProto, TensorProto
from torch import Tensor

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
