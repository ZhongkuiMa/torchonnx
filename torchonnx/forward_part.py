__docformat__ = "restructuredtext"
__all__ = ["gen_forward"]

import onnx

from .utils import *

_INDENT = "    "


def _gen_code_of_binary_op(node: onnx.NodeProto, op: str) -> str:
    input_names = parse_node_inputs(node)
    output_names = parse_node_outputs(node)
    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]} {op} {input_names[1]}\n"
    return code


def _gen_code_of_unary_func(node: onnx.NodeProto) -> str:
    input_names = parse_node_inputs(node)
    output_names = parse_node_outputs(node)
    code = _INDENT * 2 + f"{output_names[0]} = self.{node.name}({input_names[0]})\n"
    return code


def _gen_code_of_add(node: onnx.NodeProto) -> str:
    return _gen_code_of_binary_op(node, "+")


def _gen_code_of_div(node: onnx.NodeProto) -> str:
    return _gen_code_of_binary_op(node, "/")


def _gen_code_of_mul(node: onnx.NodeProto) -> str:
    return _gen_code_of_binary_op(node, "*")


def _gen_code_of_sub(node: onnx.NodeProto) -> str:
    return _gen_code_of_binary_op(node, "-")


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
    "Add": _gen_code_of_add,
    "AveragePool": _gen_code_of_unary_func,
    "BatchNormalization": _gen_code_of_unary_func,
    "Conv": _gen_code_of_unary_func,
    "ConvTranspose": _gen_code_of_unary_func,
    "Div": _gen_code_of_div,
    "Elu": _gen_code_of_unary_func,
    "Flatten": _gen_code_of_unary_func,
    "Gelu": _gen_code_of_unary_func,
    "Gemm": _gen_code_of_unary_func,
    "LeakyRelu": _gen_code_of_unary_func,
    "MaxPool": _gen_code_of_unary_func,
    "Mul": _gen_code_of_mul,
    "Relu": _gen_code_of_unary_func,
    "Sigmoid": _gen_code_of_unary_func,
    "Softmax": _gen_code_of_unary_func,
    "Sub": _gen_code_of_sub,
    "Tanh": _gen_code_of_unary_func,
    "Upsample": _gen_code_of_unary_func,
}


def gen_forward(model: onnx.ModelProto) -> str:
    input_names, input_shapes = _parse_onnx_inputs(onnx_model=model)
    content = _gen_forward_method(input_names)
    content += "\n"

    for node in model.graph.node:
        op_type = node.op_type
        _gen_code = _PARSE_NODE_MAP.get(op_type)
        if _gen_code is None:
            print(content)
            raise NotImplementedError(f"Invalid op_type: {op_type}")
        content += _gen_code(node)
    content += "\n"

    output_names = _parse_onnx_outputs(model)
    content += _gen_forward_return(output_names)

    return content
