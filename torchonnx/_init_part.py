__docformat__ = "restructuredtext"
__all__ = ["gen_init_code"]

from onnx import ModelProto, NodeProto, TensorProto

from ._onnx_attrs import get_onnx_attrs
from ._torch_args import get_torch_args
from ._utils import *

_INDENT = "    "

# TODO: Try to only use torch args.


def _gen_nothing(*args, **kwargs) -> str:
    return ""


def _gen_code_of_add(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.add.html
    return ""


def _gen_code_of_argmax(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.argmax.html
    return ""


def _gen_code_of_avgpool(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    dim = len(torch_args["kernel_size"])
    code = (
        _INDENT * 2 + f"self.{node.name} = "
        f"nn.AvgPool{dim}d("
        f'{torch_args["kernel_size"]}, '
    )

    if (
        torch_args["stride"] is not None
        or torch_args["stride"] != torch_args["kernel_size"]
    ):
        code += f"stride={torch_args['stride']}, "
    if torch_args["padding"] != 0:
        code += f"padding={torch_args['padding']}, "
    if torch_args["ceil_mode"]:
        code += f"ceil_mode={torch_args['ceil_mode']}, "
    if not torch_args["count_include_pad"]:
        code += f"count_include_pad={torch_args['count_include_pad']}, "

    code = code[:-2] + ")\n"

    return code


def _gen_code_of_batchnorm(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    dim = len(torch_args["num_features"])
    code = (
        _INDENT * 2 + f"self.{node.name} = nn.BatchNorm{dim}d("
        f'{torch_args["num_features"]}, '
    )
    if torch_args["eps"] != 1e-5:
        code += f"eps={torch_args['eps']}, "
    if torch_args["momentum"] != 0.1:
        code += f"momentum={torch_args['momentum']}, "
    if not torch_args["track_running_stats"]:
        code += f"track_running_stats={torch_args['track_running_stats']}, "
    code = code[:-2] + ")\n"

    # Set parameters
    input_names = parse_input_names(node, initializers.keys())  # noqa
    scale = torch_args["scale"]
    b = torch_args["bias"]
    mean = torch_args["running_mean"]
    variance = torch_args["running_var"]
    code += _INDENT * 2 + f"self.{node.name}.weight.data = {scale}\n"
    code += _INDENT * 2 + f"self.{node.name}.bias.data = {b}\n"
    code += _INDENT * 2 + f"self.{node.name}.running_mean.data = {mean}\n"
    code += _INDENT * 2 + f"self.{node.name}.running_var.data = {variance}\n"

    return code


def _gen_code_of_cast(*args, **kwargs) -> str:
    raise NotImplementedError("Cast is not supported yet.")


def _gen_code_of_concat(*args, **kwargs) -> str:
    return ""


def _gen_code_of_conv(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    dim = len(torch_args["kernel_size"])
    code = (
        _INDENT * 2 + f"self.{node.name} = nn.Conv{dim}d("
        f'{torch_args["in_channels"]}, '
        f'{torch_args["out_channels"]}, '
        f'{torch_args["kernel_size"]}, '
    )
    if torch_args["stride"] != 1:
        code += f"stride={torch_args['stride']}, "
    if torch_args["padding"] != 0:
        code += f"padding={torch_args['padding']}, "
    if torch_args["dilation"] != 1:
        code += f"dilation={torch_args['dilation']}, "
    if torch_args["groups"] != 1:
        code += f"groups={torch_args['groups']}, "
    code = code[:-2] + ")\n"

    # Set parameters
    input_names = parse_input_names(node, initializers.keys())  # noqa
    weight = input_names[1]
    code += _INDENT * 2 + f"self.{node.name}.weight.data = {weight}\n"
    if len(node.input) == 3:
        assert torch_args["bias"] is not None
        bias = input_names[2]
        code += _INDENT * 2 + f"self.{node.name}.bias.data = {bias}\n"
    else:
        assert torch_args["bias"] is None
    code += "\n"

    return code


def _gen_code_of_convtranspose(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    dim = len(torch_args["kernel_size"])
    code = (
        _INDENT * 2 + f"self.{node.name} = nn.ConvTranspose{dim}d("
        f'{torch_args["in_channels"]}, '
        f'{torch_args["out_channels"]}, '
        f'{torch_args["kernel_size"]}, '
    )
    if torch_args["stride"] != 1:
        code += f"stride={torch_args['stride']}, "
    if torch_args["padding"] != 0:
        code += f"padding={torch_args['padding']}, "
    if torch_args["output_padding"] != 0:
        code += f"output_padding={torch_args['output_padding']}, "
    if torch_args["groups"] != 1:
        code += f"groups={torch_args['groups']}, "
    if torch_args["dilation"] != 1:
        code += f"dilation={torch_args['dilation']}, "
    if not torch_args["bias"]:
        code += f"bias={torch_args['bias']}, "
    code = code[:-2] + ")\n"

    # Set parameters
    input_names = parse_input_names(node, initializers.keys())  # noqa
    weight = input_names[1]
    code += _INDENT * 2 + f"self.{node.name}.weight.data = {weight}\n"
    if len(node.input) == 3:
        assert torch_args["bias"] is not None
        bias = input_names[2]
        code += _INDENT * 2 + f"self.{node.name}.bias.data = {bias}\n"
    else:
        assert torch_args["bias"] is None
    code += "\n"

    return code


def _gen_code_of_constant(*args, **kwargs) -> str:
    raise RuntimeError(
        "You should use slimonnx to slim the Constant to reduce calculation. "
        "slimonnx will convert Constant to an initializer."
    )


def _gen_code_of_constantofshape(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.full.html
    raise RuntimeError(
        "You should use slimonnx to slim the ConstantOfShape to reduce calculation. "
        "slimonnx will convert ConstantOfShape to an initializer."
    )


def _gen_code_of_div(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.div.html
    return ""


def _gen_code_of_elu(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.ELU.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    code = _INDENT * 2 + f"self.{node.name} = nn.ELU("
    if torch_args["alpha"] != 1.0:
        code += f"alpha={torch_args['alpha']}"
    code += ")\n"
    return code


def _gen_code_of_flatten(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    code = _INDENT * 2 + f"self.{node.name} = nn.Flatten("
    if torch_args["start_dim"] != 1:
        code += f"start_dim={torch_args['start_dim']}"
    code += ")\n"

    return code


def _gen_code_of_gather(*args, **kwargs) -> str:
    return ""


def _gen_code_of_gelu(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    code = _INDENT * 2 + f"self.{node.name} = nn.GELU("
    if torch_args["approximation"] != "none":
        code += f"approximation={torch_args['approximation']}"
    code += ")\n"
    return code


def _gen_code_of_gemm(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    code = (
        _INDENT * 2 + f"self.{node.name} = nn.Linear("
        f"{torch_args['input_features']}, "
        f"{torch_args['output_features']}, "
    )
    if not torch_args["bias"]:
        code += f"bias={torch_args['bias']}, "
    code = code[:-2] + ")\n"

    # Set parameters
    input_names = parse_input_names(node, initializers.keys())  # noqa
    weight = input_names[1]
    code += _INDENT * 2 + f"self.{node.name}.weight.data = {weight}\n"
    if len(node.input) == 3:
        assert torch_args["bias"] is not None
        bias = input_names[2]
        code += _INDENT * 2 + f"self.{node.name}.bias.data = {bias}\n"
    else:
        assert torch_args["bias"] is None
    code += "\n"

    return code


def _gen_code_of_leakyrelu(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    code = _INDENT * 2 + f"self.{node.name} = nn.LeakyReLU("
    if torch_args["alpha"] != 0.01:
        code += f"{torch_args['alpha']}"
    code += ")\n"
    return code


def _gen_code_of_matmul(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.matmul.html
    return ""


def _gen_code_of_maxpool(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    dim = len(torch_args["kernel_size"])
    code = (
        _INDENT * 2 + f"self.{node.name} = nn.MaxPool{dim}d("
        f'{torch_args["kernel_size"]}, '
    )
    if (
        torch_args["stride"] is not None
        or torch_args["stride"] != torch_args["kernel_size"]
    ):
        code += f'stride={torch_args["stride"]}, '
    if torch_args["padding"] != 0:
        code += f'padding={torch_args["padding"]}, '
    if torch_args["dilation"] != 1:
        code += f'dilation={torch_args["dilation"]}, '
    if torch_args["ceil_mode"]:
        code += f'ceil_mode={torch_args["ceil_mode"]}, '
    code = code[:-2] + ")\n"
    return code


def _gen_code_of_mul(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.mul.html
    return ""


def _gen_code_of_pad(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.htm
    return ""


def _gen_code_of_reducemean(*args, **kwargs) -> str:
    return ""


def _gen_code_of_reducesum(*args, **kwargs) -> str:
    return ""


def _gen_code_of_relu(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    return _INDENT * 2 + f"self.{node.name} = nn.ReLU()\n"


def _gen_code_of_reshape(*args, **kwargs) -> str:
    return ""


def _gen_code_of_resize(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    return ""


def _gen_code_of_scatter(*args, **kwargs) -> str:

    return ""


def _gen_code_of_scatterelements(*args, **kwargs) -> str:
    return ""


def _gen_code_of_scatternd(*args, **kwargs) -> str:
    return ""


def _gen_code_of_sigmoid(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
    return _INDENT * 2 + f"self.{node.name} = nn.Sigmoid()\n"


def _gen_code_of_slice(*args, **kwargs) -> str:
    return ""


def _gen_code_of_softmax(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    code = _INDENT * 2 + f"self.{node.name} = nn.Softmax("
    if torch_args["dim"] is not None and torch_args["dim"] != -1:
        code += f'dim={torch_args["dim"]}'
    code += ")\n"
    return code


def _gen_code_of_split(*args, **kwargs) -> str:
    return ""


def _gen_code_of_sub(*args, **kwargs) -> str:
    # https://pytorch.org/docs/stable/generated/torch.sub.html
    return ""


def _gen_code_of_tanh(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
    return _INDENT * 2 + f"self.{node.name} = nn.Tanh()\n"


def _gen_code_of_transpose(*args, **kwargs) -> str:
    return ""


def _gen_code_of_unsqueeze(*args, **kwargs) -> str:
    return ""


def _gen_code_of_upsample(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = get_torch_args(node, attrs, nodes, initializers)
    code = _INDENT * 2 + f"self.{node.name} = nn.Upsample("
    if torch_args["mode"] != "nearest":
        code += f'mode="{torch_args["mode"]}", '
    if torch_args["scales"] is not None:
        code += f'scale_factor={torch_args["scales"]}, '
    code = code[:-2] + ")\n"
    return code


_GEN_CODE_MAP = {
    "Add": _gen_code_of_add,
    "ArgMax": _gen_code_of_argmax,
    "AveragePool": _gen_code_of_avgpool,
    "BatchNormalization": _gen_code_of_batchnorm,
    "Cast": _gen_code_of_cast,
    "Concat": _gen_code_of_concat,
    "Conv": _gen_code_of_conv,
    "ConvTranspose": _gen_code_of_convtranspose,
    "Constant": _gen_code_of_constant,
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
    "ScatterElements": _gen_code_of_scatterelements,
    "ScatterND": _gen_code_of_scatternd,
    "Slice": _gen_code_of_slice,
    "Softmax": _gen_code_of_softmax,
    "Split": _gen_code_of_split,
    "Sub": _gen_code_of_sub,
    "Tanh": _gen_code_of_tanh,
    "Transpose": _gen_code_of_transpose,
    "Unsqueeze": _gen_code_of_unsqueeze,
    "Upsample": _gen_code_of_upsample,
}


def _gen_init_header_code() -> str:
    return (
        _INDENT
        + "def __init__(self):\n"
        + _INDENT * 2
        + "super(Module, self).__init__()\n"
        + "\n"
    )


def _gen_load_pth_data_code(model: ModelProto, pth_path: str) -> str:
    return _INDENT * 2 + f"self.data = torch.load('{pth_path}')\n" + "\n"


def gen_init_code(model: ModelProto, pth_path: str) -> str:
    nodes = {node.name: node for node in model.graph.node}
    initializers = get_initializers(model)

    content = _gen_init_header_code()
    content += _gen_load_pth_data_code(model, pth_path)

    for node in model.graph.node:
        op_type = node.op_type
        _gen_node = _GEN_CODE_MAP.get(op_type)
        if _gen_node is None:
            raise NotImplementedError(f"Invalid op_type: {op_type}\n{node}")
        code = _gen_node(node, nodes, initializers)
        content += code
    content += "\n"

    return content
