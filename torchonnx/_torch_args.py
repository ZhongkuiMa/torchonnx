__docformat__ = "restructuredtext"
__all__ = ["get_torch_args"]

import warnings
from typing import Any

import onnx
import torch
from onnx import NodeProto, TensorProto

from ._utils import *
from ._onnx_attrs import get_onnx_attrs


def _simplify_pool_args(arg: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(arg, tuple):
        if len(arg) == 1:  # (x,) -> x
            return arg[0]
        elif all(x == arg[0] for x in arg):  # all elements are the same
            return arg[0]
        else:
            raise ValueError(f"Unsupported pooling argument: {arg}")
    return arg


def _torch_nothing(*args, **kwargs) -> dict[str, Any]:
    """This function does nothing and is used as a placeholder."""
    return {}


def _torch_argmax(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.argmax.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dim": None, "keepdim": False}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v
        elif k == "keepdims":
            torch_args["keepdim"] = bool(v)

    return torch_args


def _torch_avgpool(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    attrs = get_onnx_attrs(node, initializers)
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
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
    attrs = get_onnx_attrs(node, initializers)
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
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.Tensor.to.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dtype": None}

    to = attrs["to"]
    if to == 1:
        dtype = torch.float32
    elif to == 2:
        dtype = torch.uint8
    elif to == 3:
        dtype = torch.int8
    elif to == 4:
        dtype = torch.uint16
    elif to == 5:
        dtype = torch.int16
    elif to == 6:
        dtype = torch.int32
    elif to == 7:
        dtype = torch.int64
    elif to == 8:
        dtype = torch.str
    elif to == 9:
        dtype = torch.bool
    elif to == 10:
        dtype = torch.float16
    elif to == 11:
        dtype = torch.double
    elif to == 12:
        dtype = torch.uint32
    elif to == 13:
        dtype = torch.uint64
    elif to == 14:
        dtype = torch.complex64
    elif to == 15:
        dtype = torch.complex128
    else:
        raise NotImplementedError(f"Cast node with to={to} is not supported.")

    torch_args["dtype"] = dtype

    return torch_args


def _torch_concat(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.cat.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dim": 0}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v

    return torch_args


def _torch_conv(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    attrs = get_onnx_attrs(node, initializers)
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
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    attrs = get_onnx_attrs(node, initializers)
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


def _torch_constantofshape(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.full.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"fill_value": 0.0, "dtype": torch.float32}

    for k, v in attrs.items():
        if k == "value":
            torch_args["fill_value"] = v[0]
            torch_args["dtype"] = v.dtype

    return torch_args


def _torch_elu(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.ELU.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"alpha": 1.0}

    for k, v in attrs.items():
        if k == "alpha":
            torch_args["alpha"] = v
    return torch_args


def _torch_flatten(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"start_dim": 1}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["start_dim"] = v

    return torch_args


def _torch_gather(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.gather.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dim": None}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v

    return torch_args


def _torch_gelu(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"approximation": "none"}

    for k, v in attrs.items():
        if k == "approximation":
            torch_args["approximation"] = v

    return torch_args


def _torch_gemm(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    attrs = get_onnx_attrs(node, initializers)
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
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"negative_slope": 0.01}

    for k, v in attrs.items():
        if k == "alpha":
            torch_args["negative_slope"] = v
    return torch_args


def _torch_maxpool(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    attrs = get_onnx_attrs(node, initializers)
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


def _torch_pad(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    attrs = get_onnx_attrs(node, initializers)
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
    pads = initializers.get(node.input[1])
    if pads is None:
        raise RuntimeError("Pad node only supports pads as an initializer. ")
    pads = initializer_to_list(pads)
    pads = [int(x) for x in pads]
    reversed_onnx_pad = list(pads[::-1])
    torch_pad = [0] * len(reversed_onnx_pad)
    dims = len(reversed_onnx_pad) // 2
    for i in range(0, len(reversed_onnx_pad), 2):
        torch_pad[i] = reversed_onnx_pad[i // 2]
        torch_pad[i + 1] = reversed_onnx_pad[i // 2 + dims]
    torch_args["pad"] = tuple(torch_pad)  # noqa

    if len(node.input) > 2:
        constant_value = initializers.get(node.input[2])
        if constant_value is None:
            raise RuntimeError(
                "Pad node only supports constant_value as an initializer."
            )
        constant_value = initializer_to_tensor(constant_value)
        torch_args["value"] = constant_value  # noqa

    if len(node.input) > 3:
        axes = initializers.get(node.input[3])
        if axes is None:
            raise RuntimeError("Pad node only supports axes as an initializer.")
        axes = initializer_to_list(axes)
        raise NotImplementedError(f"Pad node with axes={axes} is not supported.")

    return torch_args


def _torch_reducemean(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.mean.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dim": None, "keepdim": False}
    for k, v in attrs.items():
        if k == "keepdims":
            torch_args["keepdim"] = bool(v)

    axes = initializers.get(node.input[1])
    if axes is None:
        raise RuntimeError("ReduceMean node only supports axes as an initializer.")
    dim = initializer_to_tuple(axes)
    torch_args["dim"] = dim  # noqa

    return torch_args


def _torch_reducesum(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.sum.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dim": None, "keepdim": False}

    for k, v in attrs.items():
        if k == "keepdims":
            torch_args["keepdim"] = bool(v)

    if len(node.input) > 1:
        axes = initializers.get(node.input[1])
        if axes is None:
            raise RuntimeError("ReduceSum node only supports axes as an initializer.")
        dim = initializer_to_tuple(axes)
        torch_args["dim"] = dim  # noqa

    return torch_args


def _torch_resize(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {
        "size": None,
        "scale_factor": None,
        "mode": "nearest",
        "align_corners": False,
    }

    for k, v in attrs.items():
        if k == "antialias":
            if v != 0:
                raise NotImplementedError(
                    f"Resize node with antialias={v} is not supported."
                )
        elif k == "axes":
            if v is not None:
                raise NotImplementedError("Resize node with axes is not supported.")
        elif k == "coordinate_transformation_mode":
            if v not in {"asymmetric", "half_pixel"}:
                raise NotImplementedError(
                    f"Resize node with coordinate_transformation_mode={v} is not supported."
                )
            torch_args["align_corners"] = False
            # TODO: There is some inconsistency between ONNX and PyTorch. We need check.
        elif k == "cubic_coeff_a":
            if v != -0.75:
                raise NotImplementedError(
                    f"Resize node with cubic_coeff_a={v} is not supported."
                )
        elif k == "exclude_outside":
            if v != 0:
                raise NotImplementedError(
                    f"Resize node with exclude_outside={v} is not supported."
                )
        elif k == "extrapolation_value":
            if v != 0.0:
                raise NotImplementedError(
                    f"Resize node with extrapolation_value={v} is not supported."
                )
        elif k == "keep_aspect_ratio_policy":
            if v != "stretch":
                raise NotImplementedError(
                    f"Resize node with keep_aspect_ratio_policy={v} is not supported."
                )
        elif k == "nearest_mode":
            if v not in {"floor", "round_prefer_floor"}:
                warnings.warn(f"Resize node with nearest_mode={v} is not supported.")
        elif k == "mode":
            if v != "nearest":
                raise NotImplementedError("Resize node with mode={v} is not supported.")
            torch_args["mode"] = v

    scale_factor = initializers.get(node.input[2])
    if scale_factor is None:
        raise RuntimeError("Resize node only supports scale_factor as an initializer.")
    scale_factor = initializer_to_list(scale_factor)
    torch_args["scale_factor"] = scale_factor  # noqa

    if len(node.input) > 3:
        size = initializers.get(node.input[3])
        if size is None:
            raise RuntimeError("Resize node only supports size as an initializer.")
        size = initializer_to_list(size)
        torch_args["size"] = size  # noqa

    return torch_args


def _torch_scatter(*args, **kwargs) -> dict[str, Any]:
    return _torch_scatterelement(*args, **kwargs)


def _torch_scatterelement(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.scatter.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dim": None, "index": None, "src": None}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v

    return torch_args


def _torch_scatternd(*args, **kwargs) -> dict[str, Any]:
    return _torch_scatterelement(*args, **kwargs)


def _torch_softmax(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dim": None}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v
    return torch_args


def _torch_slice(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.narrow.html
    torch_args = {"axes": None}

    if len(node.input) > 3:
        axes = initializers.get(node.input[3])
        if axes is None:
            raise RuntimeError(
                "Slice node only supports constant axes as an initializer."
            )
        axes = onnx.numpy_helper.to_array(axes).tolist()
        torch_args["axes"] = axes

    return torch_args


def _torch_split(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.split.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"split_size_or_sections": None, "dim": 0}

    for k, v in attrs.items():
        if k == "axis":
            torch_args["dim"] = v

    if len(node.input) > 1:
        split = initializers.get(node.input[1])
        if split is None:
            raise RuntimeError(
                "Split node only supports split_size_or_sections as an initializer."
            )
        split = initializer_to_list(split)
        torch_args["split_size_or_sections"] = split  # noqa
    else:
        # TODO: We need the node shapes to infer the split size.
        raise NotImplementedError(
            "Split node without split_size_or_sections is not supported."
        )

    return torch_args


def _torch_transpose(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.permute.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"dims": None}

    for k, v in attrs.items():
        if k == "perm":
            torch_args["dims"] = v

    return torch_args


def _torch_upsample(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    # https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html
    attrs = get_onnx_attrs(node, initializers)
    torch_args = {"scale_factor": None, "mode": "nearest"}

    for k, v in attrs.items():
        if k == "mode":
            torch_args["mode"] = v

    scales = initializers.get(node.input[2])
    if scales is None:
        raise RuntimeError("Upsample node only supports scales as an initializer.")
    scales = initializer_to_tuple(scales)
    torch_args["scale_factor"] = scales  # noqa

    return torch_args


_TORCH_ATTRS_MAP = {
    "Add": _torch_nothing,
    "ArgMax": _torch_argmax,
    "AveragePool": _torch_avgpool,
    "BatchNormalization": _torch_batchnorm,
    "Cast": _torch_cast,
    "Concat": _torch_concat,
    "Conv": _torch_conv,
    "ConvTranspose": _torch_convtranspose,
    "Constant": _torch_constant,
    "ConstantOfShape": _torch_constantofshape,
    "Div": _torch_nothing,
    "Elu": _torch_elu,
    "Flatten": _torch_flatten,
    "Gather": _torch_gather,
    "Gelu": _torch_gelu,
    "Gemm": _torch_gemm,
    "LeakyRelu": _torch_leakyrelu,
    "MatMul": _torch_nothing,
    "MaxPool": _torch_maxpool,
    "Mul": _torch_nothing,
    "Pad": _torch_pad,
    "ReduceMean": _torch_reducemean,
    "ReduceSum": _torch_reducesum,
    "Relu": _torch_nothing,
    "Reshape": _torch_nothing,
    "Resize": _torch_resize,
    "Shape": _torch_nothing,
    "Sigmoid": _torch_nothing,
    "Scatter": _torch_scatter,
    "ScatterElements": _torch_scatterelement,
    "ScatterND": _torch_scatternd,
    "Slice": _torch_slice,
    "Softmax": _torch_softmax,
    "Split": _torch_split,
    "Sub": _torch_nothing,
    "Tanh": _torch_nothing,
    "Transpose": _torch_transpose,
    "Unsqueeze": _torch_nothing,
    "Upsample": _torch_upsample,
}


def get_torch_args(
    node: NodeProto,
    nodes: dict[str, NodeProto],
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    _torch = _TORCH_ATTRS_MAP.get(node.op_type)
    if _torch is None:
        raise ValueError(f"Op type {node.op_type} is not supported.")
    torch_args = _torch(node, nodes, initializers)

    return torch_args
