__docformat__ = "restructuredtext"
__all__ = ["gen_forward_code"]

import onnx

from ._onnx_attrs import *
from ._torch_args import *
from ._utils import *

_INDENT = "    "


def _gen_code_of_add(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    return _gen_code_of_binary_op(node, initializer_shapes, "+")


def _gen_code_of_binary_op(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]], op: str
) -> str:
    input_names = parse_input_names(node, initializer_shapes)
    output_names = parse_output_names(node, initializer_shapes)
    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]} {op} {input_names[1]}\n"
    return code


def _gen_code_of_concat(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.cat.html
    input_names = parse_input_names(node, initializer_shapes)
    output_names = parse_output_names(node, initializer_shapes)
    axis = get_attrs_of_onnx_node(node)["axis"]
    code = (
        _INDENT * 2
        + f"{output_names[0]} = torch.cat([{', '.join(input_names)}], dim={axis})\n"
    )
    return code


def _gen_code_of_div(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    return _gen_code_of_binary_op(node, initializer_shapes, "/")


def _gen_code_of_matmul(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    input_names = parse_input_names(node, initializer_shapes)
    output_names = parse_output_names(node, initializer_shapes)
    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]} @ {input_names[1]}\n"
    return code


def _gen_code_of_mul(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    return _gen_code_of_binary_op(node, initializer_shapes, "*")


def _gen_code_of_reducemean(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.mean.html
    input_names = parse_input_names(node, initializer_shapes)
    output_names = parse_output_names(node, initializer_shapes)
    attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, attrs, initializer_shapes)
    keepdims = torch_args["keepdims"]
    code = _INDENT * 2 + (
        f"{output_names[0]} = "
        f"torch.mean({input_names[0]}, dim={input_names[1]}, keepdim={keepdims})\n"
    )
    return code


def _gen_code_of_reshape(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.reshape.html
    input_names = parse_input_names(node, initializer_shapes)
    output_names = parse_output_names(node, initializer_shapes)
    return _INDENT * 2 + (
        f"{output_names[0]} = {input_names[0]}.reshape({input_names[1]})\n"
    )


def _gen_code_of_sub(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    return _gen_code_of_binary_op(node, initializer_shapes, "-")


def _gen_code_of_transpose(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.permute.html
    input_names = parse_input_names(node, initializer_shapes)
    output_names = parse_output_names(node, initializer_shapes)
    perm = get_attrs_of_onnx_node(node)["perm"]
    return _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.permute({perm})\n"


def _gen_code_of_unary_func(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    input_names = parse_input_names(node, initializer_shapes)
    output_names = parse_output_names(node, initializer_shapes)
    code = _INDENT * 2 + f"{output_names[0]} = self.{node.name}({input_names[0]})\n"
    return code


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
    for input_node in onnx_model.graph.input:
        input_names.append(input_node.name)
        input_shape = []
        for dim in input_node.type.tensor_type.shape.dim:
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
    "Concat": _gen_code_of_concat,
    "Conv": _gen_code_of_unary_func,
    "ConvTranspose": _gen_code_of_unary_func,
    "Div": _gen_code_of_div,
    "Elu": _gen_code_of_unary_func,
    "Flatten": _gen_code_of_unary_func,
    "Gelu": _gen_code_of_unary_func,
    "Gemm": _gen_code_of_unary_func,
    "LeakyRelu": _gen_code_of_unary_func,
    "MaxPool": _gen_code_of_unary_func,
    "MatMul": _gen_code_of_matmul,
    "Mul": _gen_code_of_mul,
    "Relu": _gen_code_of_unary_func,
    "ReduceMean": _gen_code_of_reducemean,
    "Reshape": _gen_code_of_reshape,
    "Sigmoid": _gen_code_of_unary_func,
    "Softmax": _gen_code_of_unary_func,
    "Sub": _gen_code_of_sub,
    "Tanh": _gen_code_of_unary_func,
    "Transpose": _gen_code_of_transpose,
    "Upsample": _gen_code_of_unary_func,
}


def gen_forward_code(model: onnx.ModelProto) -> str:
    input_names, input_shapes = _parse_onnx_inputs(onnx_model=model)
    initializer_shapes = get_initializer_shapes(model)

    content = _gen_forward_method(input_names)
    content += "\n"

    for node in model.graph.node:
        op_type = node.op_type
        _gen_code = _PARSE_NODE_MAP.get(op_type)
        if _gen_code is None:
            raise NotImplementedError(f"Invalid op_type: {op_type}\n{node}")
        content += _gen_code(node, initializer_shapes)
    content += "\n"

    output_names = _parse_onnx_outputs(model)
    content += _gen_forward_return(output_names)

    return content
