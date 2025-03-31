__docformat__ = "restructuredtext"
__all__ = ["parse_input_names", "parse_output_names", "get_initializer_shapes"]

from typing import Any

import onnx

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
    node: onnx.NodeProto, initializer_names: list[str] | dict[str, Any]
) -> list[str]:
    input_names = []
    for name in node.input:
        if name in initializer_names:
            name = f"self.data['{name}']"
        input_names.append(name)
    return input_names


def parse_output_names(
    node: onnx.NodeProto, initializer_names: list[str] | dict[str, Any]
) -> list[str]:
    output_names = []
    for name in node.output:
        if name in initializer_names:
            name = f"self.data['{name}']"
        output_names.append(name)
    return output_names


def get_initializer_shapes(model: onnx.ModelProto) -> dict[str, tuple]:
    initializer_shapes = {}
    for initializer in model.graph.initializer:
        shape = []
        for dim in initializer.dims:
            shape.append(dim)
        initializer_shapes[initializer.name] = tuple(shape)

    return initializer_shapes
