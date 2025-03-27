__docformat__ = "restructuredtext"
__all__ = ["gen_forward"]

import onnx

from ._ops import *


def _parse_onnx_outputs(model: onnx.ModelProto) -> list[str]:
    output_names = []
    for output in model.graph.output:
        output_names.append(output.name)

    return output_names


def _gen_forward_return(outputs: list) -> str:
    s = "        return "
    for output in outputs:
        s += f"{output}, "
    s = s[:-2]  # Remove the last comma and space
    s += "\n"

    return s


def _parse_onnx_inputs(onnx_model: onnx.ModelProto) -> tuple[list[str], list[tuple]]:
    input_names = []
    input_shapes = []
    for input in onnx_model.graph.input:
        input_names.append(input.name)
        input_shape = []
        for dim in input.type.tensor_type.shape.dim:
            input_shape.append(dim.dim_value)
        input_shape.pop(0)  # Remove the batch dimension
        input_shapes.append(tuple(input_shape))

    return input_names, input_shapes


def _gen_forward_method(inputs_name: list) -> str:
    s = "    def forward(self, "
    for name in inputs_name:
        s += f"{name}: Tensor, "
    s = s[:-2]  # Remove the last comma and space
    s += ") -> Tensor:\n"

    return s


_PARSE_NODE_MAP = {
    "Gemm": parse_gemm,
    "Conv": parse_conv,
    "Relu": parse_relu,
    "Add": parse_add,
    "Flatten": parse_flatten,
}


def gen_forward(model: onnx.ModelProto) -> str:
    input_names, input_shapes = _parse_onnx_inputs(onnx_model=model)
    content = _gen_forward_method(input_names)
    content += "\n"

    for node in model.graph.node:
        op_type = node.op_type
        _parse_node = _PARSE_NODE_MAP.get(op_type)
        if _parse_node is None:
            print(content)
            raise NotImplementedError(f"Invalid op_type: {op_type}")
        content += _parse_node(node)
    content += "\n"

    output_names = _parse_onnx_outputs(model)
    content += _gen_forward_return(output_names)

    return content
