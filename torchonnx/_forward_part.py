__docformat__ = "restructuredtext"
__all__ = ["gen_forward_code"]

from onnx import ModelProto, NodeProto, TensorProto

from ._onnx_attrs import *
from ._torch_args import *
from ._utils import *

_INDENT = "    "

# TODO: Try to only use torch args.


def _parse_onnx_inputs(onnx_model: ModelProto) -> tuple[list[str], list[tuple]]:
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


def _parse_onnx_outputs(model: ModelProto) -> list[str]:
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


def _gen_code_of_unary_func(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> str:
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    code = _INDENT * 2 + f"{output_names[0]} = self.{node.name}({input_names[0]})\n"
    return code


def _gen_code_of_binary_op(
    node: NodeProto, initializers: dict[str, TensorProto], op: str
) -> str:
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]} {op} {input_names[1]}\n"
    return code


def _gen_code_of_add(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.add.html
    return _gen_code_of_binary_op(node, initializers, "+")


def _gen_code_of_argmax(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.argmax.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    dim = attrs["dim"]
    keepdim = attrs["keepdim"]
    code = (
        _INDENT * 2 + f"{output_names[0]} = torch.argmax("
        f"{input_names[0]}, "
        f"dim={dim}"
    )
    if keepdim:
        code += ", keepdim=True"
    code += ")\n"

    return code


def _gen_code_of_avgpool(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_batchnorm(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    return _gen_code_of_unary_func(node, initializers)


def _gen_code_of_cast(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    # TODO: Support this operation
    raise NotImplementedError("This method has not been implemented yet.")


def _gen_code_of_concat(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.cat.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    dim = attrs["dim"]
    code = _INDENT * 2 + f"{output_names[0]} = torch.cat([{', '.join(input_names)}]"
    if dim != 0:
        code + ", dim={dim}"
    code += ")\n"

    return code


def _gen_code_of_conv(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_convtranspose(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_constantofshape(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.full.html
    raise RuntimeError("This method is unnecessary and slim it to an initializer.")


def _gen_code_of_div(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    return _gen_code_of_binary_op(node, initializers, "/")


def _gen_code_of_leakyrelu(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_matmul(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]} @ {input_names[1]}\n"
    return code


def _gen_code_of_elu(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_flatten(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_gather(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.gather.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    dim = attrs["axis"]
    index = attrs["axis"]
    code = (
        _INDENT * 2 + f"{output_names[0]} = torch.gather("
        f"{input_names[0]}, {dim}, {input_names[1]}"
        f")\n"
    )
    return code


def _gen_code_of_gelu(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_gemm(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    return _gen_code_of_unary_func(node, initializers)


def _gen_code_of_maxpool(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_mul(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    return _gen_code_of_binary_op(node, initializers, "*")


def _gen_code_of_pad(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    pad = attrs["pads"]
    mode = attrs["mode"]
    value = attrs["value"]
    code = _INDENT * 2 + f"{output_names[0]} = F.pad({input_names[0]}, {pad}"
    if mode != "constant":
        code += f", mode={mode}"
    if value != 0:
        code += f", value={value}"
    code += ")\n"

    return code


def _gen_code_of_reducemean(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.mean.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    dim = attrs["dim"]
    if len(dim) == 1:
        dim = dim[0]
    keepdim = attrs["keepdim"]
    code = _INDENT * 2 + f"{output_names[0]} = torch.mean({input_names[0]}, {dim}"
    if keepdim:
        code += ", keepdim=True"
    code += ")\n"

    return code


def _gen_code_of_reducesum(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.sum.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    dim = attrs["dim"]
    if len(dim) == 1:
        dim = dim[0]
    keepdim = attrs["keepdim"]
    code = _INDENT * 2 + f"{output_names[0]} = torch.sum({input_names[0]}, {dim}"
    if keepdim:
        code += ", keepdim=True"
    code += ")\n"

    return code


def _gen_code_of_relu(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_reshape(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.reshape.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    return _INDENT * 2 + (
        f"{output_names[0]} = {input_names[0]}.reshape({input_names[1]})\n"
    )


def _gen_code_of_resize(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    # TODO: Support this operation
    raise NotImplementedError("This method has not been implemented yet.")


def _gen_code_of_scatter(*args, **kwargs) -> str:
    return _gen_code_of_scatterelement(*args, **kwargs)


def _gen_code_of_scatterelement(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.scatter.html
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)

    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    dim = attrs["dim"]
    index = attrs["index"]
    src = attrs["src"]

    code = (
        _INDENT * 2
        + f"{output_names[0]} = {input_names[0]}.scatter({dim}, {index}, {src})\n"
    )
    return code


def _gen_code_of_scatternd(*args, **kwargs) -> str:
    return _gen_code_of_scatterelement(*args, **kwargs)


def _gen_code_of_sigmoid(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_slice(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    input_name = parse_input_names(node, initializers)[0]
    output_name = parse_output_names(node, initializers)[0]
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    starts = attrs["starts"]
    ends = attrs["ends"]
    axes = attrs["axes"]
    steps = attrs["steps"]
    code = _INDENT * 2 + f"{output_name} = {input_name}["
    for i in range(len(starts)):
        if i in axes:
            code += f"{starts[i]}:{ends[i]}"
            if steps[i] != 1:
                code += f":{steps[i]}"
        else:
            code += ":"
        code += ", "
    code = code[:-2] + "]\n"

    return code


def _gen_code_of_softmax(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    return _gen_code_of_unary_func(node, initializers)


def _gen_code_of_split(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.split.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    split = attrs["split_size_or_sections"]
    dim = attrs["dim"]
    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.split({split}"
    if dim != 0:
        code += f", dim={dim}"
    code += ")\n"

    return code


def _gen_code_of_sub(*args, **kwargs) -> str:
    return _gen_code_of_binary_op(*args, **kwargs)


def _gen_code_of_tanh(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_transpose(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.permute.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    dims = attrs["dims"]
    return _INDENT * 2 + (f"{output_names[0]} = {input_names[0]}.permute({dims}" f")\n")


def _gen_code_of_unsqueeze(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    attrs = get_onnx_attrs(node, initializers)
    attrs = get_torch_args(node, attrs, nodes, initializers)
    dim = attrs["dim"]
    return _INDENT * 2 + (
        f"{output_names[0]} = {input_names[0]}.unsqueeze(dim={dim})\n"
    )


def _gen_code_of_upsample(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # TODO: Support this operation
    raise NotImplementedError("Upsample operation is not supported yet.")


_PARSE_NODE_MAP = {
    "Add": _gen_code_of_add,
    "ArgMax": _gen_code_of_argmax,
    "AveragePool": _gen_code_of_avgpool,
    "BatchNormalization": _gen_code_of_batchnorm,
    "Cast": _gen_code_of_cast,
    "Concat": _gen_code_of_concat,
    "Conv": _gen_code_of_conv,
    "ConvTranspose": _gen_code_of_convtranspose,
    "ConstantOfShape": _gen_code_of_constantofshape,
    "Div": _gen_code_of_div,
    "Elu": _gen_code_of_elu,
    "Flatten": _gen_code_of_flatten,
    "Gather": _gen_code_of_gather,
    "Gelu": _gen_code_of_gelu,
    "Gemm": _gen_code_of_gemm,
    "LeakyRelu": _gen_code_of_leakyrelu,
    "MatMul": _gen_code_of_matmul,
    "MaxPool": _gen_code_of_maxpool,
    "Mul": _gen_code_of_mul,
    "Pad": _gen_code_of_pad,
    "ReduceMean": _gen_code_of_reducemean,
    "ReduceSum": _gen_code_of_reducesum,
    "Relu": _gen_code_of_relu,
    "Reshape": _gen_code_of_reshape,
    "Resize": _gen_code_of_resize,
    "Sigmoid": _gen_code_of_sigmoid,
    "Scatter": _gen_code_of_scatter,
    "ScatterElements": _gen_code_of_scatterelement,
    "ScatterND": _gen_code_of_scatternd,
    "Slice": _gen_code_of_slice,
    "Softmax": _gen_code_of_softmax,
    "Split": _gen_code_of_split,
    "Sub": _gen_code_of_sub,
    "Tanh": _gen_code_of_unary_func,
    "Transpose": _gen_code_of_transpose,
    "Unsqueeze": _gen_code_of_unsqueeze,
    "Upsample": _gen_code_of_upsample,
}


def gen_forward_code(model: ModelProto) -> str:
    input_names, input_shapes = _parse_onnx_inputs(model)
    initializers = get_initializers(model)
    nodes = {node.name: node for node in model.graph.node}

    content = _gen_forward_method(input_names)
    content += "\n"

    for node in model.graph.node:
        op_type = node.op_type
        _gen_code = _PARSE_NODE_MAP.get(op_type)
        if _gen_code is None:
            raise NotImplementedError(f"Invalid op_type: {op_type}\n{node}")
        content += _gen_code(node, nodes, initializers)
    content += "\n"

    output_names = _parse_onnx_outputs(model)
    content += _gen_forward_return(output_names)

    return content
