__docformat__ = "restructuredtext"
__all__ = ["gen_forward_code"]

from onnx import ModelProto, NodeProto, TensorProto

from ._torch_args import *
from ._utils import *

_INDENT = "    "


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
    s = _INDENT + "def forward(self, "
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
    s = 2 * _INDENT + "return "
    for output in outputs:
        s += f"{output}, "
    s = s[:-2]  # Remove the last comma and space
    s += "\n"

    return s


def _gen_code_of_unary_func(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + f"{output_names[0]} = self.{node.name}({input_names[0]})\n"

    return code


def _gen_code_of_equal(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.eq.html
    return _gen_code_of_binary_op(node, initializers, "==")


def _gen_code_of_expand(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    shape = input_names[1]
    if node.input[1] in initializers:
        shape = initializer_to_tuple(initializers[node.input[1]])

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.expand({shape})\n"

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
    return _gen_code_of_binary_op(node, initializers, "+")


def _gen_code_of_argmax(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.argmax.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    dim = torch_args["dim"]
    keepdim = torch_args["keepdim"]

    code = _INDENT * 2 + f"{output_names[0]} = torch.argmax({input_names[0]}, "
    if dim is not None:
        code += f"dim={dim}, "
    if keepdim:
        code += "keepdim=True, "
    code = code[:-2] + ")\n"

    return code


def _gen_code_of_cast(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    dtype = torch_args["dtype"]

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.to(dtype={dtype})\n"

    return code


def _gen_code_of_clip(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.clamp.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    min = input_names[1]
    if node.input[1] in initializers:
        min = initializer_to_int(initializers[node.input[1]])
    max = input_names[2]
    if node.input[2] in initializers:
        max = initializer_to_int(initializers[node.input[2]])

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.clamp("
    if min != "":
        code += f"min={min}, "
    if max != "":
        code += f"max={max}, "
    code = code[:-2] + ")\n"

    return code


def _gen_code_of_concat(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.cat.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    dim = torch_args["dim"]

    code = _INDENT * 2 + f"{output_names[0]} = torch.cat([{', '.join(input_names)}], "
    if dim != 0:
        code += f"dim={dim}, "
    code = code[:-2] + ")\n"

    return code


def _gen_code_of_constant(*args, **kwargs) -> str:
    raise RuntimeError(
        "You should use slimonnx to slim the Constant to reduce calculation. "
        "slimonnx will convert Constant to an initializer."
    )


def _gen_code_of_constantofshape(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.full.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    fill_value = torch_args["fill_value"]
    dtype = torch_args["dtype"]

    code = _INDENT * 2 + f"{output_names[0]} = torch.full({input_names[0]}, "
    if fill_value is not None:
        code += f"{fill_value}, "
    if dtype is not None:
        code += f"dtype={dtype}, "
    code = code[:-2] + ")\n"

    return code


def _gen_code_of_cos(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.cos.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + f"{output_names[0]} = torch.cos({input_names[0]})\n"

    return code


def _gen_code_of_gather(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.gather.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    dim = torch_args["dim"]
    index = input_names[1]
    if node.input[1] in initializers:
        index = initializer_to_tensor(initializers[node.input[1]])
        # if it is a single scalar
        if index.dim() == 0:
            index = "torch.tensor(" + str(index.item()) + ")"

    code = (
        _INDENT * 2 + f"{output_names[0]} = torch.gather("
        f"{input_names[0]}, {dim}, {index}"
        f")\n"
    )
    return code


def _gen_code_of_div(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    return _gen_code_of_binary_op(node, initializers, "/")


def _gen_code_of_matmul(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = (
        _INDENT * 2
        + f"{output_names[0]} = torch.matmul({input_names[0]}, {input_names[1]})\n"
    )

    return code


def _gen_code_of_max(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.max.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.max()\n"

    return code


def _gen_code_of_min(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.min.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.min()\n"

    return code


def _gen_code_of_mul(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    return _gen_code_of_binary_op(node, initializers, "*")


def _gen_code_of_neg(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.neg.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + f"{output_names[0]} = -{input_names[0]}\n"

    return code


def _gen_code_of_pad(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    pad = torch_args["pad"]
    mode = torch_args["mode"]
    value = torch_args["value"]

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

    torch_args = get_torch_args(node, nodes, initializers)
    dim = torch_args["dim"]
    keepdim = torch_args["keepdim"]
    if len(dim) == 1:
        dim = dim[0]

    code = _INDENT * 2 + f"{output_names[0]} = torch.mean({input_names[0]}, dim={dim}, "
    if keepdim:
        code += "keepdim=True, "
    code = code[:-2] + ")\n"

    return code


def _gen_code_of_reducesum(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.sum.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    dim = torch_args["dim"]
    keepdim = torch_args["keepdim"]
    if len(dim) == 1:
        dim = dim[0]

    code = _INDENT * 2 + f"{output_names[0]} = torch.sum({input_names[0]}, dim={dim}, "
    if keepdim:
        code += "keepdim=True, "
    code = code[:-2] + ")\n"

    return code


def _gen_code_of_reshape(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.reshape.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    shape = input_names[1]
    if node.input[1] in initializers:
        shape = initializer_to_tuple(initializers[node.input[1]])

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.reshape({shape})\n"

    return code


def _gen_code_of_resize(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    size = torch_args["size"]
    scale_factor = torch_args["scale_factor"]
    mode = torch_args["mode"]
    align_corners = torch_args["align_corners"]

    code = _INDENT * 2 + f"{output_names[0]} = F.interpolate({input_names[0]}, "
    if size is not None:
        code += f"size={size}, "
    if scale_factor is not None:
        code += f"scale_factor={scale_factor}, "
    if mode is not None:
        code += f"mode='{mode}', "
    if align_corners:
        code += f"align_corners={align_corners}, "
    code = code[:-2] + ")\n"

    return code


def _gen_code_of_scatter(*args, **kwargs) -> str:
    return _gen_code_of_scatterelement(*args, **kwargs)


def _gen_code_of_scatterelement(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.scatter.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    dim = torch_args["dim"]
    index = input_names[1]
    src = input_names[2]

    code = (
        _INDENT * 2
        + f"{output_names[0]} = {input_names[0]}.scatter({dim}, {index}, {src})\n"
    )

    return code


def _gen_code_of_scatternd(*args, **kwargs) -> str:
    return _gen_code_of_scatterelement(*args, **kwargs)


def _gen_code_of_shape(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.shape.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.shape\n"

    return code


def _gen_code_of_sin(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.sin.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + f"{output_names[0]} = torch.sin({input_names[0]})\n"

    return code


def _gen_code_of_slice(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)
    torch_args = get_torch_args(node, nodes, initializers)

    starts_is_initer = node.input[1] in initializers
    ends_is_initer = node.input[2] in initializers
    steps_is_initer = len(input_names) > 4 and node.input[4] in initializers
    starts = input_names[1]
    if starts_is_initer:
        starts = initializer_to_list(initializers[node.input[1]])
    ends = input_names[2]
    if ends_is_initer:
        ends = initializer_to_list(initializers[node.input[2]])

    axes = None
    if len(input_names) > 3:
        axes = torch_args["axes"]
    else:
        # TODO: Support slice without axes by data shape.
        raise NotImplementedError("Slice node without axes is not supported yet. ")

    steps = None
    if len(input_names) > 4:
        steps = input_names[4]
        if node.input[4] in initializers:
            steps = initializer_to_list(initializers[node.input[4]])

    dim = max(axes) + 1

    code = ""
    if not starts_is_initer:
        code += _INDENT * 2 + f"_start = {starts}\n"
    if not ends_is_initer:
        code += _INDENT * 2 + f"_end = {ends}\n"
    if steps is not None and steps_is_initer and steps != [1]:
        code += _INDENT * 2 + f"_step = {steps}\n"
    code += _INDENT * 2 + f"{output_names[0]} = {input_names[0]}["
    k = 0
    for i in range(dim):
        if i in axes:
            if not starts_is_initer:
                code += f"_start[{k}]"
            else:
                code += f"{starts[k]}"
            code += ":"
            if not ends_is_initer:
                code += f"_end[{k}]"
            else:
                code += f"{ends[k]}"
            if steps is not None:
                if not steps_is_initer:
                    code += f":_step[{k}]"
                else:
                    if steps[k] != 1:
                        code += f":{steps[k]}"
            k += 1
        else:
            code += ":"
        code += ", "
    code = code[:-2] + "]\n"

    return code


def _gen_code_of_split(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.split.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    split = torch_args["split_size_or_sections"]
    dim = torch_args["dim"]

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.split({split}"
    if dim != 0:
        code += f", dim={dim}"
    code += ")\n"

    return code


def _gen_code_of_squeeze(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.squeeze.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    dim = None
    if len(input_names) > 1:
        dim = input_names[1]
        if node.input[1] in initializers:
            dim = initializer_to_int(initializers[node.input[1]])

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.squeeze("
    if dim is not None:
        code += f"dim={dim}"
    code += ")\n"

    return code


def _gen_code_of_sub(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    return _gen_code_of_binary_op(node, initializers, "-")


def _gen_code_of_transpose(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.permute.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    torch_args = get_torch_args(node, nodes, initializers)
    dims = torch_args["dims"]

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.permute({dims})\n"

    return code


def _gen_code_of_unsqueeze(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    dim = input_names[1]
    if node.input[1] in initializers:
        dim = initializer_to_int(initializers[node.input[1]])

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.unsqueeze(dim={dim})\n"

    return code


def _gen_code_of_upsample(*args, **kwargs) -> str:
    return _gen_code_of_unary_func(*args, **kwargs)


def _gen_code_of_where(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.where.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    condition = input_names[0]
    x = input_names[1]
    y = input_names[2]

    code = _INDENT * 2 + f"{output_names[0]} = torch.where(" f"{condition}, {x}, {y})\n"

    return code


_PARSE_NODE_MAP = {
    "Add": _gen_code_of_add,
    "ArgMax": _gen_code_of_argmax,
    "AveragePool": _gen_code_of_unary_func,
    "BatchNormalization": _gen_code_of_unary_func,
    "Cast": _gen_code_of_cast,
    "Clip": _gen_code_of_clip,
    "Concat": _gen_code_of_concat,
    "Conv": _gen_code_of_unary_func,
    "ConvTranspose": _gen_code_of_unary_func,
    "Constant": _gen_code_of_constant,
    "ConstantOfShape": _gen_code_of_constantofshape,
    "Cos": _gen_code_of_cos,
    "Div": _gen_code_of_div,
    "Dropout": _gen_code_of_unary_func,
    "Elu": _gen_code_of_unary_func,
    "Equal": _gen_code_of_equal,
    "Expand": _gen_code_of_expand,
    "Flatten": _gen_code_of_unary_func,
    "Gather": _gen_code_of_gather,
    "Gelu": _gen_code_of_unary_func,
    "Gemm": _gen_code_of_unary_func,
    "LeakyRelu": _gen_code_of_unary_func,
    "Min": _gen_code_of_min,
    "MatMul": _gen_code_of_matmul,
    "Max": _gen_code_of_max,
    "MaxPool": _gen_code_of_unary_func,
    "Mul": _gen_code_of_mul,
    "Neg": _gen_code_of_neg,
    "Pad": _gen_code_of_pad,
    "ReduceMean": _gen_code_of_reducemean,
    "ReduceSum": _gen_code_of_reducesum,
    "Relu": _gen_code_of_unary_func,
    "Reshape": _gen_code_of_reshape,
    "Resize": _gen_code_of_resize,
    "Shape": _gen_code_of_shape,
    "Sigmoid": _gen_code_of_unary_func,
    "Sin": _gen_code_of_sin,
    "Scatter": _gen_code_of_scatter,
    "ScatterElements": _gen_code_of_scatterelement,
    "ScatterND": _gen_code_of_scatternd,
    "Slice": _gen_code_of_slice,
    "Softmax": _gen_code_of_unary_func,
    "Split": _gen_code_of_split,
    "Squeeze": _gen_code_of_squeeze,
    "Sub": _gen_code_of_sub,
    "Tanh": _gen_code_of_unary_func,
    "Transpose": _gen_code_of_transpose,
    "Unsqueeze": _gen_code_of_unsqueeze,
    "Upsample": _gen_code_of_unary_func,
    "Where": _gen_code_of_where,
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
