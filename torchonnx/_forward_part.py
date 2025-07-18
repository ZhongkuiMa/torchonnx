__docformat__ = "restructuredtext"
__all__ = ["gen_forward_code"]

import onnx
from onnx import ModelProto, NodeProto, TensorProto

from slimonnx import reformat_io_shape
from ._torch_args import *
from ._utils import *

_INDENT = "    "


def _parse_onnx_inputs(onnx_model: ModelProto) -> tuple[list[str], list[list[int]]]:
    input_names = []
    input_shapes = []
    for node in onnx_model.graph.input:
        input_names.append(node.name)
        input_shapes.append(reformat_io_shape(node))

    return input_names, input_shapes


def _gen_forward_method(inputs_name: list[str], input_shapes: list[list[int]]) -> str:
    s = _INDENT + "def forward(self, "
    for name in inputs_name:
        s += f"{name}: Tensor, "
    s = s[:-2]  # Remove the last comma and space
    s += ") -> Tensor:\n"
    for name, shape in zip(inputs_name, input_shapes):
        s += _INDENT * 2 + f"{name} = {name}.reshape(-1, "
        for dim in shape[1:]:
            s += f"{dim}, "
        s = s[:-2] + ")\n"

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

    # The following refers to the Expand operator in ONNX, which is different from
    # the expand function in PyTorch.
    # https://onnx.ai/onnx/operators/onnx__Expand.html
    code = (
        _INDENT * 2
        + f"shape = {shape}.tolist()\n"
        + _INDENT * 2
        + f"try:\n"
        + _INDENT * 3
        + f"{output_names[0]} = {input_names[0]}.expand(shape)\n"
        + _INDENT * 2
        + "except RuntimeError:\n"
        + _INDENT * 3
        + f"# Apply right-alignment and generate a valid shape for PyTorch\n"
        + _INDENT * 3
        + f"shape1, shape2 = list({input_names[0]}.shape), shape\n"
        + _INDENT * 3
        + f"# Pad with 1s to the left side to get the same length\n"
        + _INDENT * 3
        + f"shape2 = [1] * (len(shape1) - len(shape2)) + shape2\n"
        + _INDENT * 3
        + f"shape1 = [1] * (len(shape2) - len(shape1)) + shape1\n"
        + _INDENT * 3
        + f"shape = [max(a, b) for a, b in zip(shape1, shape2)]\n"
        + _INDENT * 3
        + f"{output_names[0]} = {input_names[0]}.expand(shape)\n"
    )

    return code


def _gen_code_of_floor(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.floor.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + f"{output_names[0]} = torch.floor({input_names[0]})\n"

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

    # Check if all inputs are initializers or variables
    all_initer = True
    all_var = True
    for i in range(len(input_names)):
        if node.input[i] in initializers:
            all_var = False
        else:
            all_initer = False
    mixed = not all_initer and not all_var
    if mixed:
        # Add a batch dimension for all initializers to make them the same shape as
        # variables.
        new_input_names = []
        for i in range(len(input_names)):
            input_name = input_names[i]
            # All initializers do not have batch dimension.
            if node.input[i] in initializers:
                initializer = initializers[node.input[i]]
                # Check if the initializer is empty.
                # Sometimes, the initializer is empty. Weired but true.
                array = onnx.numpy_helper.to_array(initializer)
                if array.size == 0:
                    continue
                input_name += ".unsqueeze(0)"
            new_input_names.append(input_name)
        input_names = new_input_names

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

    code = _INDENT * 2 + f"{output_names[0]} = torch.full({input_names[0]}.tolist(), "
    if fill_value is not None:
        code += f"{fill_value}, "
    if dtype is not None:
        code += f"dtype={dtype}, "
    code += f"device=self.device)\n"

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
        # If it is a single scalar
        index_data = initializer_to_tensor(initializers[node.input[1]])
        if index_data.dim() == 0:
            index = str(index_data.item())

    code = _INDENT * 2 + f"{output_names[0]} = {input_names[0]}["
    d = 0
    while True:
        if d == dim:
            code += f"{index}]\n"
            break
        code += " : , "
        d += 1

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

    code = _INDENT * 2 + f"{output_names[0]} = torch.matmul("
    # Handle left and right matmul
    if node.input[0] in initializers and node.input[1] not in nodes:
        # There is a special cases, some networks has an input without batch dimension.
        # We will add this batch dimension to the input.
        # Because we may use batched inputs to process adv attack.
        # And the first layer is a left matmul.
        # In such case, there will be an error.
        # Because the input is not in the nodes dict, we can check such case.
        code += f"{input_names[1]}, {input_names[0]}.t())\n"
    else:
        code += f"{input_names[0]}, {input_names[1]})\n"

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


def _gen_code_of_pow(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.pow.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = (
        _INDENT * 2
        + f"{output_names[0]} = torch.pow({input_names[0]}, {input_names[1]})\n"
    )

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
        # For multiple batch dimension, we need to reshape it to a valid shape.
        if len(shape) > 1:
            shape = shape[1:]  # Remove the first batch dimension
            code = _INDENT * 2 + f"temp_batch_size = {input_names[0]}.shape[0]\n"
            code = code + _INDENT * 2 + f"temp_shape = (temp_batch_size, *{shape})\n"
            code = (
                code
                + _INDENT * 2
                + (f"{output_names[0]} = {input_names[0]}.reshape(temp_shape)\n")
            )
        else:
            code = (
                _INDENT * 2 + f"{output_names[0]} = {input_names[0]}.reshape({shape})\n"
            )
        return code

    code = (
        _INDENT * 2
        + f"shape = {shape}.tolist()\n"
        + _INDENT * 2
        + f"try:\n"
        + _INDENT * 3
        + f"{output_names[0]} = {input_names[0]}.reshape(shape)\n"
        + _INDENT * 2
        + "except RuntimeError:\n"
        + _INDENT * 3
        + f"# Replace 0 with -1 to be a valid shape in PyTorch\n"
        + _INDENT * 3
        + f"shape = [-1 if dim == 0 else dim for dim in shape]\n"
        + _INDENT * 3
        + f"{output_names[0]} = {input_names[0]}.reshape(shape)\n"
    )

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
    assert dim is not None, "Dim must be specified for scatter operation"
    index = input_names[1]
    src = input_names[2]

    code = (
        _INDENT * 2
        + f"{output_names[0]} = {input_names[0]}.scatter({dim}, {index}, {src})\n"
    )

    return code


def _gen_code_of_scatternd(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.scatter.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    index = input_names[1]
    src = input_names[2]

    code = (
        _INDENT * 2
        + f"{output_names[0]} = {input_names[0]}.clone()\n"
        + _INDENT * 2
        + f"{output_names[0]}[{index}] = {src}\n"
    )

    return code


def _gen_code_of_shape(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.shape.html
    input_names = parse_input_names(node, initializers)
    output_names = parse_output_names(node, initializers)

    code = _INDENT * 2 + (
        f"{output_names[0]} = "
        f"torch.tensor({input_names[0]}.shape, device=self.device)\n"
    )

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

    code = _INDENT * 2
    for output_name in output_names:
        code += f"{output_name}, "
    code = code[:-2]
    code += f" = {input_names[0]}.split({split}"
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
    "Floor": _gen_code_of_floor,
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
    "Pow": _gen_code_of_pow,
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

    content = _gen_forward_method(input_names, input_shapes)
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
