__docformat__ = "restructuredtext"
__all__ = ["get_torch_args"]

import warnings
from typing import Any

import onnx
import torch
from torch import Tensor
from onnx import NodeProto, TensorProto

from ._utils import *


def _simplify_pool_args(arg: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(arg, tuple):
        if len(arg) == 1:  # (x,) -> x
            return arg[0]
        elif all(x == arg[0] for x in arg):  # all elements are the same
            return arg[0]
        else:
            raise ValueError(f"Unsupported pooling argument: {arg}")
    return arg


def _to_tensor(initializer: TensorProto) -> Tensor:
    return torch.tensor(onnx.numpy_helper.to_array(initializer))


def _to_list(initializer: TensorProto) -> list:
    return onnx.numpy_helper.to_array(initializer).tolist()


def _to_tuple(initializer: TensorProto) -> tuple:
    return tuple(onnx.numpy_helper.to_array(initializer).tolist())


def _torch_add(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.add.html
    return {}


def _torch_argmax(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.argmax.html
    torch_args = {
        "dim": None,
        "keepdim": False,
    }

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v
        elif k == "keepdims":
            torch_args["keepdim"] = bool(v)

    return torch_args


def _torch_avgpool(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    torch_attrs = {
        "kernel_size": None,
        "stride": None,
        "padding": 0,
        "ceil_mode": False,
        "count_include_pad": True,
    }

    for k, v in attrs.items():
        if k == "kernel_shape":
            torch_attrs["kernel_size"] = v  # Do not simplify to indicate the dimension
        elif k == "strides":
            torch_attrs["stride"] = _simplify_pool_args(v)
        elif k == "pads":
            torch_attrs["padding"] = _simplify_pool_args(v[:2])
        elif k == "ceil_mode":
            torch_attrs["ceil_mode"] = bool(v)
        elif k == "count_include_pad":
            torch_attrs["count_include_pad"] = bool(v)

    return torch_attrs


def _torch_batchnorm(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    warnings.warn(
        "You may use slimonnx to slim the BatchNormalization to reduce calculation. "
        "slimonnx will fuse BatchNormalization with its neighbor Conv or Gemm layers."
    )
    torch_args = {
        "num_features": None,
        "eps": 1e-5,
        "momentum": 0.1,
        "track_running_stats": True,
    }

    for k, v in attrs.items():
        if k == "epsilon":
            torch_args["eps"] = v
        elif k == "momentum":
            torch_args["momentum"] = 1.0 - v
        elif k == "training_mode":
            torch_args["track_running_stats"] = bool(v)

    bias = onnx.numpy_helper.to_array(initializers[node.input[2]])
    torch_args["num_features"] = bias.shape[0]

    return torch_args


def _torch_cast(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    # TODO: Support cast
    raise NotImplementedError("This method has not been implemented yet.")


def _torch_concat(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.cat.html
    torch_args = {"dim": 0}
    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v

    return torch_args


def _torch_conv(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    torch_args = {
        "in_channels": None,
        "out_channels": None,
        "kernel_size": None,
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
    }

    inputs = parse_input_names(node, initializers)
    weight_shape = tuple(initializers[node.input[1]].dims)
    torch_args["in_channels"] = weight_shape[1]
    torch_args["out_channels"] = weight_shape[0]
    torch_args["bias"] = bool(len(inputs) == 3)

    for k, v in attrs.items():
        if k == "kernel_shape":
            torch_args["kernel_size"] = v  # Do not simplify to indicate the dimension
        elif k == "strides":
            torch_args["stride"] = _simplify_pool_args(v)
        elif k == "pads":
            torch_args["padding"] = _simplify_pool_args(v[:2])
        elif k == "dilations":
            torch_args["dilation"] = _simplify_pool_args(v)
        elif k == "group":
            torch_args["groups"] = _simplify_pool_args(v)

    return torch_args


def _torch_convtranspose(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    torch_args = {
        "in_channels": None,
        "out_channels": None,
        "kernel_size": None,
        "stride": 1,
        "padding": 0,
        "output_padding": 0,
        "dilation": 1,
        "groups": 1,
        "bias": True,
    }

    inputs = parse_input_names(node, initializers)
    weight_shape = tuple(initializers[node.input[1]].dims)
    torch_args["in_channels"] = weight_shape[1]
    torch_args["out_channels"] = weight_shape[0]
    torch_args["bias"] = bool(len(inputs) == 3)

    for k, v in attrs.items():
        if k == "kernel_shape":
            torch_args["kernel_size"] = v  # Do not simplify to indicate the dimension
        elif k == "strides":
            torch_args["stride"] = _simplify_pool_args(v)
        elif k == "pads":
            torch_args["padding"] = _simplify_pool_args(v[:2])
        elif k == "dilations":
            torch_args["dilation"] = _simplify_pool_args(v)
        elif k == "group":
            torch_args["groups"] = _simplify_pool_args(v)
        elif k == "output_padding":
            torch_args["output_padding"] = _simplify_pool_args(v)

    return torch_args


def _torch_constant(*args, **kwargs) -> dict[str, Any]:
    raise RuntimeError(
        "You should use slimonnx to slim the Constant to reduce calculation. "
        "slimonnx will convert Constant to an initializer."
    )


def _torch_constantofshape(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.full.html
    raise RuntimeError(
        "You should use slimonnx to slim the ConstantOfShape to reduce calculation. "
        "slimonnx will convert ConstantOfShape to an initializer."
    )


def _torch_div(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.div.html
    return {}


def _torch_elu(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.ELU.html
    torch_args = {"alpha": 1.0}
    for k, v in attrs.items():
        if k == "alpha":
            torch_args["alpha"] = v
    return torch_args


def _torch_flatten(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    torch_args = {"start_dim": 1}
    for k, v in attrs.items():
        if k == "axis":
            torch_args["start_dim"] = v

    return torch_args


def _torch_gather(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.gather.html
    torch_args = {"dim": None, "index": None}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v
        elif k == "index":
            index = _to_tensor(initializers[node.input[1]])
            torch_args["index"] = index

    return torch_args


def _torch_gelu(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    torch_args = {"approximation": "none"}
    for k, v in attrs.items():
        if k == "approximation":
            torch_args["approximation"] = v

    return torch_args


def _torch_gemm(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    torch_args = {  # We need those to set transpose in code generation
        "transA": 0,
        "transB": 0,
        "alpha": 1.0,
        "beta": 1.0,
    }

    for k, v in attrs.items():
        if k == "transA":
            torch_args["transA"] = v
        elif k == "transB":
            torch_args["transB"] = v
        elif k == "alpha":
            torch_args["alpha"] = v
        elif k == "beta":
            torch_args["beta"] = v

    input_names = parse_input_names(node, initializers)
    weight_shape = tuple(initializers[node.input[1]].dims)
    if attrs["transB"] == 1:
        torch_args["input_features"] = weight_shape[1]
        torch_args["output_features"] = weight_shape[0]
    else:
        torch_args["input_features"] = weight_shape[0]
        torch_args["output_features"] = weight_shape[1]
    torch_args["bias"] = bool(len(input_names) == 3)

    return torch_args


def _torch_leakyrelu(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    torch_args = {"negative_slope": 0.01}
    for k, v in attrs.items():
        if k == "alpha":
            torch_args["negative_slope"] = v
    return torch_args


def _torch_matmul(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.matmul.html
    return {}


def _torch_maxpool(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    torch_args = {
        "kernel_size": None,
        "stride": None,  # Default value is kernel_size in torch.nn.MaxPool
        "padding": 0,
        "dilation": 1,
        "ceil_mode": False,
    }

    for k, v in attrs.items():
        if k == "kernel_shape":
            torch_args["kernel_size"] = v  # Do not simplify to indicate the dimension
        elif k == "strides":
            torch_args["stride"] = _simplify_pool_args(v)
        elif k == "pads":
            torch_args["padding"] = _simplify_pool_args(v[:2])
        elif k == "dilations":
            torch_args["dilation"] = _simplify_pool_args(v)
        elif k == "ceil_mode":
            torch_args["ceil_mode"] = bool(v)  # 0 or 1 => False or True

    return torch_args


def _torch_mul(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.mul.html
    return {}


def _torch_pad(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    torch_args = {
        "pad": None,
        "mode": "constant",
        "value": 0.0,
    }
    for k, v in attrs.items():
        if k == "mode":
            torch_args["mode"] = v

    """
    Note:
    Convert ONNX pad [start_1, start_2, start_3, ..., end_1, end_2, end_3, ...]
    to PyTorch pad [..., start_3, end_3, start_2, end_2, start_1, end_1]
    """
    pads = _to_list(initializers[node.input[1]])
    pads = [int(x) for x in pads]
    reversed_onnx_pad = list(pads[::-1])
    torch_pad = [0] * len(reversed_onnx_pad)
    dims = len(reversed_onnx_pad) // 2
    for i in range(0, len(reversed_onnx_pad), 2):
        torch_pad[i] = reversed_onnx_pad[i // 2]
        torch_pad[i + 1] = reversed_onnx_pad[i // 2 + dims]
    torch_args["pad"] = tuple(torch_pad)

    constant_value = _to_tensor(initializers[node.input[2]])
    torch_args["value"] = constant_value

    if len(node.input) > 3:
        axes = _to_list(initializers[node.input[3]])
        raise ValueError(f"We haven't support partial axes yet with axes={axes}")

    return torch_args


def _torch_reducemean(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.mean.html
    torch_args = {"keepdims": False}
    for k, v in attrs.items():
        if k == "keepdims":
            torch_args["keepdim"] = bool(v)

    axes = initializers[node.input[1]]
    dim = _to_tuple(axes)
    torch_args["dim"] = dim

    return torch_args


def _torch_reducesum(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.sum.html
    torch_args = {"keepdims": False}
    for k, v in attrs.items():
        if k == "keepdims":
            torch_args["keepdim"] = bool(v)

    axes = initializers[node.input[1]]
    dim = _to_tuple(axes)
    torch_args["dim"] = dim

    return torch_args


def _torch_relu(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
    return {}


def _torch_reshape(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.reshape.html
    torch_args = {"shape": None}
    shape = initializers[node.input[1]]
    shape = _to_tuple(shape)
    torch_args["shape"] = shape

    return torch_args


def _torch_resize(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    # TODO: Support resize
    raise NotImplementedError("This method has not been implemented yet.")


def _torch_scatter(*args, **kwargs) -> dict[str, Any]:
    return _torch_scatterelement(*args, **kwargs)


def _torch_scatterelement(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.scatter.html
    torch_args = {"dim": None, "index": None, "src": None}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v

    torch_args["index"] = _to_tensor(initializers[node.input[1]])
    torch_args["src"] = _to_tensor(initializers[node.input[2]])

    return torch_args


def _torch_scatternd(*args, **kwargs) -> dict[str, Any]:
    return _torch_scatterelement(*args, **kwargs)


def _torch_shape(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.shape.html
    raise RuntimeError(
        "You should use slimonnx to slim the Shape to reduce calculation. "
        "slimonnx will convert Shape to an initializer."
    )


def _torch_sigmoid(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.sigmoid.html
    return {}


def _torch_slice(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://onnx.ai/onnx/operators/onnx__Slice.html
    torch_args = {
        "starts": None,
        "ends": None,
        "axes": None,
        "steps": None,
    }

    starts = initializers[node.input[1]]
    starts = _to_list(starts)
    torch_args["starts"] = starts

    ends = initializers[node.input[2]]
    ends = _to_list(ends)
    torch_args["ends"] = ends

    if len(node.input) < 4:
        torch_args["axes"] = list(range(len(starts)))
    else:
        axes = initializers[node.input[3]]
        axes = _to_list(axes)
        torch_args["axes"] = axes

    if len(node.input) < 5:
        torch_args["steps"] = [1] * len(starts)
    else:
        steps = initializers[node.input[4]]
        steps = _to_list(steps)
        torch_args["steps"] = steps

    return torch_args


def _torch_softmax(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    torch_args = {"dim": None}
    for k, v in attrs.items():
        if k == "axis":
            assert v >= 0 or v == -1
            if v == -1:
                v = None
            torch_args["dim"] = v
    return torch_args


def _torch_split(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.split.html
    torch_args = {
        "split_size_or_sections": None,
        "dim": 0,
    }

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v

    if len(node.input) > 1:
        split = initializers[node.input[1]]
        split = int(onnx.numpy_helper.to_array(split))
        torch.args["split_size_or_sections"] = split
    else:
        # TODO: We need the node shapes to infer the split size.
        raise NotImplementedError(
            "We only support split with split_size_or_sections is given in ONNX."
        )

    return torch_args


def _torch_sub(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.sub.html
    return {}


def _torch_tanh(*args, **kwargs) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.tanh.html
    return {}


def _torch_transpose(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.permute.html
    torch_args = {"dims": None}
    for k, v in attrs.items():
        if k == "perm":
            torch_args["dims"] = v

    return torch_args


def _torch_unsqueeze(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
    torch_args = {"dim": None}
    for k, v in attrs.items():
        if k == "axes":
            torch_args["dim"] = v

    return torch_args


def _torch_upsample(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    # TODO: Support upsample
    raise NotImplementedError("This method has not been implemented yet.")
    torch_args = {
        # "size":None,
        "scale_factor": None,
        "mode": "nearest",
        # "align_corners": None,
        # "recompute_scale_factor": None,
    }
    for k, v in attrs.items():
        if k == "mode":
            torch_args["mode"] = v

    scales = _to_tuple(initializers[node.input[1]])
    torch_args["scale_factor"] = scales

    return torch_args


_TORCH_ATTRS_MAP = {
    "Add": _torch_add,
    "ArgMax": _torch_argmax,
    "AveragePool": _torch_avgpool,
    "BatchNormalization": _torch_batchnorm,
    "Cast": _torch_cast,
    "Concat": _torch_concat,
    "Conv": _torch_conv,
    "ConvTranspose": _torch_convtranspose,
    "Constant": _torch_constant,
    "ConstantOfShape": _torch_constantofshape,
    "Div": _torch_div,
    "Elu": _torch_elu,
    "Flatten": _torch_flatten,
    "Gather": _torch_gather,
    "Gelu": _torch_gelu,
    "Gemm": _torch_gemm,
    "LeakyRelu": _torch_leakyrelu,
    "MatMul": _torch_matmul,
    "MaxPool": _torch_maxpool,
    "Mul": _torch_mul,
    "Pad": _torch_pad,
    "ReduceMean": _torch_reducemean,
    "ReduceSum": _torch_reducesum,
    "Relu": _torch_relu,
    "Reshape": _torch_reshape,
    "Resize": _torch_resize,
    "Shape": _torch_shape,
    "Sigmoid": _torch_sigmoid,
    "Scatter": _torch_scatter,
    "ScatterElements": _torch_scatterelement,
    "ScatterND": _torch_scatternd,
    "Slice": _torch_slice,
    "Softmax": _torch_softmax,
    "Split": _torch_split,
    "Sub": _torch_sub,
    "Tanh": _torch_tanh,
    "Transpose": _torch_transpose,
    "Unsqueeze": _torch_unsqueeze,
    "Upsample": _torch_upsample,
}


def get_torch_args(
    node: NodeProto,
    attrs: dict[str, Any],
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    _torch = _TORCH_ATTRS_MAP.get(node.op_type)
    if _torch is None:
        raise ValueError(f"Invalid op_type: {node.op_type}")
    torch_args = _torch(node, attrs, nodes, initializers)

    return torch_args
