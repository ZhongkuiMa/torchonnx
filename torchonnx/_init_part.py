__docformat__ = "restructuredtext"
__all__ = ["gen_init_code"]

import warnings

from onnx import ModelProto, NodeProto, TensorProto

from ._torch_args import get_torch_args
from ._utils import *

_INDENT = "    "


def _gen_nothing(*args, **kwargs) -> str:
    return ""


def _gen_code_of_avgpool(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    torch_args = get_torch_args(node, nodes, initializers)

    dim = len(torch_args["kernel_size"])
    code = (
        _INDENT * 2 + f"self.{node.name} = "
        f"nn.AvgPool{dim}d("
        f'{torch_args["kernel_size"]}, '
    )

    if (
        torch_args["stride"] is not None
        and torch_args["stride"] != torch_args["kernel_size"]
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
    torch_args = get_torch_args(node, nodes, initializers)

    # TODO: Use data shape to infer the dimension.
    code = (
        _INDENT * 2 + f"self.{node.name} = nn.BatchNorm2d("
        f'{torch_args["num_features"]}, '
    )
    eps = torch_args["eps"]
    if eps != 1e-5:
        new_eps = round(eps, 6)
        print(f"Round BatchNorm momentum: {eps} -> {new_eps}")
        if new_eps != 1e-5:
            code += f"eps={new_eps}, "
    momentum = torch_args["momentum"]
    if momentum != 0.1:
        new_momentum = round(momentum, 6)
        print(f"Round BatchNorm momentum: {momentum} -> {new_momentum}")
        if new_momentum != 0.1:
            code += f"momentum={new_momentum}, "
    if not torch_args["track_running_stats"]:
        code += f"track_running_stats={torch_args['track_running_stats']}, "
    code = code[:-2] + ")\n"

    # Set parameters
    input_names = parse_input_names(node, initializers.keys())  # noqa
    scale = input_names[1]
    b = input_names[2]
    mean = input_names[3]
    variance = input_names[4]
    code += _INDENT * 2 + f"self.{node.name}.weight.data = {scale}\n"
    code += _INDENT * 2 + f"self.{node.name}.bias.data = {b}\n"
    code += _INDENT * 2 + f"self.{node.name}.running_mean.data = {mean}\n"
    code += _INDENT * 2 + f"self.{node.name}.running_var.data = {variance}\n"
    code += "\n"

    return code


def _gen_code_of_conv(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    torch_args = get_torch_args(node, nodes, initializers)

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
    torch_args = get_torch_args(node, nodes, initializers)

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


def _gen_code_of_dropout(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
    torch_args = get_torch_args(node, nodes, initializers)

    code = _INDENT * 2 + f"self.{node.name} = nn.Dropout("
    if torch_args["p"] != 0.5:
        code += f"p={torch_args['p']}"
    code += ")\n"

    warnings.warn("Dropout needs a fixed seed to reproduce the result.")

    return code


def _gen_code_of_elu(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.ELU.html
    torch_args = get_torch_args(node, nodes, initializers)

    code = _INDENT * 2 + f"self.{node.name} = nn.ELU("
    if torch_args["alpha"] != 1.0:
        code += f"alpha={torch_args['alpha']}"
    code += ")\n"

    return code


def _gen_code_of_flatten(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    torch_args = get_torch_args(node, nodes, initializers)

    code = _INDENT * 2 + f"self.{node.name} = nn.Flatten("
    if torch_args["start_dim"] != 1:
        code += f"start_dim={torch_args['start_dim']}"
    code += ")\n"

    return code


def _gen_code_of_gelu(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    torch_args = get_torch_args(node, nodes, initializers)

    code = _INDENT * 2 + f"self.{node.name} = nn.GELU("
    if torch_args["approximation"] != "none":
        code += f"approximation={torch_args['approximation']}"
    code += ")\n"

    return code


def _gen_code_of_gemm(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    torch_args = get_torch_args(node, nodes, initializers)
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
    code += _INDENT * 2 + f"self.{node.name}.weight.data = {weight}.t()\n"
    if len(node.input) == 3:
        bias = input_names[2]
        code += _INDENT * 2 + f"self.{node.name}.bias.data = {bias}\n"
    code += "\n"

    return code


def _gen_code_of_leakyrelu(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    torch_args = get_torch_args(node, nodes, initializers)

    code = _INDENT * 2 + f"self.{node.name} = nn.LeakyReLU("
    if torch_args["negative_slope"] != 0.01:
        code += f"negative_slope={torch_args['negative_slope']}"
    code += ")\n"

    return code


def _gen_code_of_maxpool(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    torch_args = get_torch_args(node, nodes, initializers)

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


def _gen_code_of_relu(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    return _INDENT * 2 + f"self.{node.name} = nn.ReLU()\n"


def _gen_code_of_sigmoid(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
    return _INDENT * 2 + f"self.{node.name} = nn.Sigmoid()\n"


def _gen_code_of_softmax(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    torch_args = get_torch_args(node, nodes, initializers)
    code = _INDENT * 2 + f"self.{node.name} = nn.Softmax("
    if torch_args["dim"] is not None:
        code += f'dim={torch_args["dim"]}'
    code += ")\n"
    return code


def _gen_code_of_tanh(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
    return _INDENT * 2 + f"self.{node.name} = nn.Tanh()\n"


def _gen_code_of_upsample(
    node: NodeProto, nodes: dict[str, NodeProto], initializers: dict[str, TensorProto]
) -> str:
    # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    torch_args = get_torch_args(node, nodes, initializers)
    code = _INDENT * 2 + f"self.{node.name} = nn.Upsample("
    if torch_args["mode"] != "nearest":
        code += f'mode="{torch_args["mode"]}", '
    if torch_args["scales"] is not None:
        code += f'scale_factor={torch_args["scales"]}, '
    code = code[:-2] + ")\n"
    return code


_GEN_CODE_MAP = {
    "Add": _gen_nothing,
    "ArgMax": _gen_nothing,
    "AveragePool": _gen_code_of_avgpool,
    "BatchNormalization": _gen_code_of_batchnorm,
    "Cast": _gen_nothing,
    "Clip": _gen_nothing,
    "Concat": _gen_nothing,
    "Conv": _gen_code_of_conv,
    "ConvTranspose": _gen_code_of_convtranspose,
    "Constant": _gen_code_of_constant,
    "ConstantOfShape": _gen_nothing,
    "Cos": _gen_nothing,
    "Div": _gen_nothing,
    "Dropout": _gen_code_of_dropout,
    "Elu": _gen_code_of_elu,
    "Equal": _gen_nothing,
    "Expand": _gen_nothing,
    "Flatten": _gen_code_of_flatten,
    "Floor": _gen_nothing,
    "Gather": _gen_nothing,
    "Gelu": _gen_code_of_gelu,
    "Gemm": _gen_code_of_gemm,
    "LeakyRelu": _gen_code_of_leakyrelu,
    "Min": _gen_nothing,
    "MatMul": _gen_nothing,
    "Max": _gen_nothing,
    "MaxPool": _gen_code_of_maxpool,
    "Mul": _gen_nothing,
    "Neg": _gen_nothing,
    "Pad": _gen_nothing,
    "Pow": _gen_nothing,
    "ReduceMean": _gen_nothing,
    "ReduceSum": _gen_nothing,
    "Relu": _gen_code_of_relu,
    "Reshape": _gen_nothing,
    "Resize": _gen_nothing,
    "Shape": _gen_nothing,
    "Sigmoid": _gen_code_of_sigmoid,
    "Sin": _gen_nothing,
    "Scatter": _gen_nothing,
    "ScatterElements": _gen_nothing,
    "ScatterND": _gen_nothing,
    "Slice": _gen_nothing,
    "Softmax": _gen_code_of_softmax,
    "Split": _gen_nothing,
    "Squeeze": _gen_nothing,
    "Sub": _gen_nothing,
    "Tanh": _gen_code_of_tanh,
    "Transpose": _gen_nothing,
    "Unsqueeze": _gen_nothing,
    "Upsample": _gen_code_of_upsample,
    "Where": _gen_nothing,
}


def _gen_init_header_code() -> str:
    return (
        _INDENT
        + "def __init__(\n"
        + _INDENT * 2
        + "self,\n"
        + _INDENT * 2
        + "dtype: torch.dtype = torch.float32,\n"
        + _INDENT * 2
        + "device:torch.device = torch.device('cpu'),\n"
        + _INDENT
        + "):\n"
        + _INDENT * 2
        + "super().__init__()\n\n"
    )


def _gen_load_pth_data_code(pth_path: str, initializers: dict[str, TensorProto]) -> str:
    pth_path = pth_path.replace("\\", "\\\\")
    code = (
        _INDENT * 2
        + f"self.data = torch.load('{pth_path}', weights_only=True)\n"
        + _INDENT * 2
        + "for name in self.data:\n"
        + _INDENT * 3
        + f"if torch.is_floating_point(self.data[name]):\n"
        + _INDENT * 4
        + f"self.data[name] = self.data[name].to(dtype=dtype, device=device)\n\n"
        + _INDENT * 3
        + "else:\n"
        + _INDENT * 4
        + f"self.data[name] = self.data[name].to(device=device)\n\n"
    )

    return code


def gen_init_code(model: ModelProto, pth_path: str) -> str:
    nodes = {node.name: node for node in model.graph.node}
    initializers = get_initializers(model)

    content = _gen_init_header_code()
    content += _gen_load_pth_data_code(pth_path, initializers)

    for node in model.graph.node:
        op_type = node.op_type
        _gen_node = _GEN_CODE_MAP.get(op_type)
        if _gen_node is None:
            raise NotImplementedError(f"Invalid op_type: {op_type}\n{node}")
        code = _gen_node(node, nodes, initializers)
        content += code
    content += "\n"

    return content
