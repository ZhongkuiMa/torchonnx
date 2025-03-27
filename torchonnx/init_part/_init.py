__docformat__ = "restructuredtext"
__all__ = ["gen_init"]

import onnx

from ._ops import *


def _gen_init_method() -> str:
    s = "    def __init__(self):\n"
    s += "        super(Module, self).__init__()\n"
    s += "\n"
    return s


def _get_initializer_shapes(model: onnx.ModelProto) -> dict[str, tuple]:
    initializer_shapes = {}
    for initializer in model.graph.initializer:
        shape = []
        for dim in initializer.dims:
            shape.append(dim)
        initializer_shapes[initializer.name] = tuple(shape)

    return initializer_shapes


_PARSE_NODE_MAP = {
    "Gemm": parse_gemm,
    "Conv": parse_conv,
    "Relu": parse_relu,
    "Add": parse_add,
    "Flatten": parse_flatten,
}


def gen_init(model: onnx.ModelProto) -> str:
    initializer_shapes = _get_initializer_shapes(model)

    content = _gen_init_method()

    for node in model.graph.node:
        op_type = node.op_type
        _parse_node = _PARSE_NODE_MAP.get(op_type)
        if _parse_node is None:
            print(content)
            raise NotImplementedError(f"Invalid op_type: {op_type}")
        content += _parse_node(node, initializer_shapes)

    content += "\n"

    return content
