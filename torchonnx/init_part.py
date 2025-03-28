__docformat__ = "restructuredtext"
__all__ = ["gen_init_code"]

import onnx

from .onnx_attrs import get_attrs_of_onnx_node
from .torch_args import get_torch_args_of_onnx_attrs

_INDENT = "    "


def _gen_code_of_add(*args, **kwargs) -> str:
    return ""


def _gen_code_of_avgpool(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
    dim = len(torch_args["kernel_size"])
    code = (
        _INDENT * 2 + f"self.{node.name} = nn.AvgPool{dim}d("
        f'{torch_args["kernel_size"]}, '
    )
    if (
        torch_args["stride"] is not None
        or torch_args["stride"] == torch_args["kernel_size"]
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
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
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
    return code


def _gen_code_of_conv(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
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
    return code


def _gen_code_of_convtranspose(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
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
    return code


def _gen_code_of_div(*args, **kwargs) -> str:
    return ""


def _gen_code_of_elu(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
    code = _INDENT * 2 + f"self.{node.name} = nn.ELU("
    if torch_args["alpha"] != 1.0:
        code += f"alpha={torch_args['alpha']}"
    code += ")\n"
    return code


def _gen_code_of_flatten(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
    code = _INDENT * 2 + f"self.{node.name} = nn.Flatten("
    if torch_args["start_dim"] != 1:
        code += f"start_dim={torch_args['start_dim']}"
    code += ")\n"
    return code


def _gen_code_of_gelu(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
    code = _INDENT * 2 + f"self.{node.name} = nn.GELU("
    if torch_args["approximation"] != "none":
        code += f"approximation={torch_args['approximation']}"
    code += ")\n"
    return code


def _gen_code_of_gemm(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
    code = (
        _INDENT * 2 + f"self.{node.name} = nn.Linear("
        f"{torch_args['input_features']}, "
        f"{torch_args['output_features']}, "
    )
    if not torch_args["bias"]:
        code += f"bias={torch_args['bias']}, "
    code = code[:-2] + ")\n"
    return code


def _gen_code_of_leakyrelu(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
    code = _INDENT * 2 + f"self.{node.name} = nn.LeakyReLU("
    if torch_args["alpha"] != 0.01:
        code += f"{torch_args['alpha']}"
    code += ")\n"
    return code


def _gen_code_of_maxpool(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
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
    return ""


def _gen_code_of_relu(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    return _INDENT * 2 + f"self.{node.name} = nn.ReLU()\n"


def _gen_code_of_sigmoid(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    return _INDENT * 2 + f"self.{node.name} = nn.Sigmoid()\n"


def _gen_code_of_softmax(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
    code = _INDENT * 2 + f"self.{node.name} = nn.Softmax("
    if torch_args["axis"] is not None:
        code += f'dim={torch_args["axis"]})\n'
    code = ")\n"
    return code


def _gen_code_of_sub(*args, **kwargs) -> str:
    return ""


def _gen_code_of_tanh(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    return _INDENT * 2 + f"self.{node.name} = nn.Tanh()\n"


def _gen_code_of_upsample(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    onnx_attrs = get_attrs_of_onnx_node(node)
    torch_args = get_torch_args_of_onnx_attrs(node, onnx_attrs, initializer_shapes)
    code = _INDENT * 2 + f"self.{node.name} = nn.Upsample("
    if torch_args["mode"] != "nearest":
        code += f'mode="{torch_args["mode"]}", '
    if torch_args["scales"] is not None:
        code += f'scale_factor={torch_args["scales"]}, '
    code = code[:-2] + ")\n"
    return code


_GEN_CODE_MAP = {
    "Add": _gen_code_of_add,
    "AveragePool": _gen_code_of_avgpool,
    "BatchNormalization": _gen_code_of_batchnorm,
    "Conv": _gen_code_of_conv,
    "ConvTranspose": _gen_code_of_convtranspose,
    "Div": _gen_code_of_div,
    "Elu": _gen_code_of_elu,
    "Flatten": _gen_code_of_flatten,
    "Gelu": _gen_code_of_gelu,
    "Gemm": _gen_code_of_gemm,
    "LeakyRelu": _gen_code_of_leakyrelu,
    "MaxPool": _gen_code_of_maxpool,
    "Mul": _gen_code_of_mul,
    "Relu": _gen_code_of_relu,
    "Sigmoid": _gen_code_of_sigmoid,
    "Softmax": _gen_code_of_softmax,
    "Sub": _gen_code_of_sub,
    "Tanh": _gen_code_of_tanh,
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


def _get_initializer_shapes(model: onnx.ModelProto) -> dict[str, tuple]:
    initializer_shapes = {}
    for initializer in model.graph.initializer:
        shape = []
        for dim in initializer.dims:
            shape.append(dim)
        initializer_shapes[initializer.name] = tuple(shape)

    return initializer_shapes


def gen_init_code(model: onnx.ModelProto) -> str:
    initializer_shapes = _get_initializer_shapes(model)

    content = _gen_init_header_code()

    for node in model.graph.node:
        op_type = node.op_type
        _gen_node = _GEN_CODE_MAP.get(op_type)
        if _gen_node is None:
            raise NotImplementedError(f"Invalid op_type: {op_type}")
        content += _gen_node(node, initializer_shapes)
    content += "\n"

    return content
