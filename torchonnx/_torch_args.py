__docformat__ = "restructuredtext"
__all__ = ["get_torch_args_of_onnx_attrs"]

from typing import Any

import onnx

from ._utils import *


def _simplify_pool_args(arg: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(arg, tuple):
        if len(arg) == 1:  # (x,) -> x
            return arg[0]
        elif all(x == arg[0] for x in arg):  # all elements are the same
            return arg[0]
    return arg


def _torch_attrs_avgpool(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
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
            if len(v) == 4:
                if v[0] != v[2] or v[1] != v[3]:
                    raise ValueError(f"Unsupported padding: {v}")
                torch_attrs["padding"] = _simplify_pool_args(v[:2])
            else:
                raise NotImplementedError
        elif k == "ceil_mode":
            torch_attrs["ceil_mode"] = bool(v)
        elif k == "count_include_pad":
            torch_attrs["count_include_pad"] = bool(v)

    return torch_attrs


def _torch_attrs_batchnorm(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {
        "num_features": None,
        "eps": 1e-5,
        "momentum": 0.1,
        "track_running_stats": True,
    }
    raise NotImplementedError("The num_feature needs shape inference.")
    # torch_args["num_features"] = input_size[1] # Channel size

    for k, v in attrs.items():
        if k == "epsilon":
            torch_args["eps"] = v
        elif k == "momentum":
            torch_args["momentum"] = 1.0 - v
        elif k == "training_mode":
            torch_args["track_running_stats"] = v

    return torch_args


def _torch_attrs_conv(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
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

    inputs = parse_input_names(node, initializer_shapes.keys())  # type: ignore
    weight_shape = initializer_shapes[node.input[1]]
    torch_args["in_channels"] = weight_shape[1]
    torch_args["out_channels"] = weight_shape[0]
    torch_args["bias"] = bool(len(inputs) == 3)

    for k, v in attrs.items():
        if k == "kernel_shape":
            torch_args["kernel_size"] = v  # Do not simplify to indicate the dimension
        elif k == "strides":
            torch_args["stride"] = _simplify_pool_args(v)
        elif k == "pads":
            if len(v) == 4:
                if v[0] != v[2] or v[1] != v[3]:
                    raise ValueError(f"Unsupported padding: {v}")
                torch_args["padding"] = _simplify_pool_args(v[:2])
            else:
                raise NotImplementedError
        elif k == "dilations":
            torch_args["dilation"] = _simplify_pool_args(v)
        elif k == "group":
            torch_args["groups"] = _simplify_pool_args(v)

    return torch_args


def _torch_attrs_convtranspose(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
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

    inputs = parse_input_names(node, initializer_shapes.keys())  # type: ignore
    weight_shape = initializer_shapes[inputs[1]]
    torch_args["in_channels"] = weight_shape[1]
    torch_args["out_channels"] = weight_shape[0]
    torch_args["bias"] = bool(len(inputs) == 3)

    for k, v in attrs.items():
        if k == "kernel_shape":
            torch_args["kernel_size"] = v  # Do not simplify to indicate the dimension
        elif k == "strides":
            torch_args["stride"] = _simplify_pool_args(v)
        elif k == "pads":
            if len(v) == 4:
                if v[0] != v[2] or v[1] != v[3]:
                    raise ValueError(f"Unsupported padding: {v}")
                torch_args["padding"] = _simplify_pool_args(v[:2])
            else:
                raise NotImplementedError
        elif k == "dilations":
            torch_args["dilation"] = _simplify_pool_args(v)
        elif k == "group":
            torch_args["groups"] = _simplify_pool_args(v)
        elif k == "output_padding":
            torch_args["output_padding"] = _simplify_pool_args(v)

    return torch_args


def _torch_attrs_elu(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {"alpha": 1.0}
    for k, v in attrs.items():
        if k == "alpha":
            torch_args["alpha"] = v
    return torch_args


def _torch_attrs_flatten(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {"start_dim": 1}
    for k, v in attrs.items():
        if k == "axis":
            torch_args["start_dim"] = v

    return torch_args


def _torch_attrs_gelu(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {"approximation": "none"}
    for k, v in attrs.items():
        if k == "approximation":
            torch_args["approximation"] = v

    return torch_args


def _torch_attrs_gemm(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {}

    input_names = parse_input_names(node, initializer_shapes.keys())  # type: ignore
    weight_shape = initializer_shapes[node.input[1]]
    if attrs["transB"] == 1:
        torch_args["input_features"] = weight_shape[1]
        torch_args["output_features"] = weight_shape[0]
    else:
        torch_args["input_features"] = weight_shape[0]
        torch_args["output_features"] = weight_shape[1]
    torch_args["bias"] = bool(len(input_names) == 3)

    if attrs["alpha"] != 1.0:
        raise ValueError(f"Invalid alpha: {attrs['alpha']}")
    if attrs["beta"] != 1.0:
        raise ValueError(f"Invalid beta: {attrs['beta']}")
    if attrs["transA"] != 0:
        raise ValueError(f"Invalid transA: {attrs['transA']}")

    return torch_args


def _torch_attrs_leakyrelu(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {"negative_slope": 0.01}
    for k, v in attrs.items():
        if k == "alpha":
            torch_args["negative_slope"] = v
    return torch_args


def _torch_attrs_maxpool(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
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
            if len(v) == 4:
                if v[0] != v[2] or v[1] != v[3]:
                    raise ValueError(f"Unsupported padding: {v}")
                torch_args["padding"] = _simplify_pool_args(v[:2])
            else:
                raise NotImplementedError
        elif k == "dilations":
            torch_args["dilation"] = _simplify_pool_args(v)
        elif k == "ceil_mode":
            torch_args["ceil_mode"] = bool(v)  # 0 or 1 => False or True

    return torch_args


def _torch_attrs_softmax(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {"dim": None}
    for k, v in attrs.items():
        if k == "axis":
            if v not in {0, -1}:
                raise NotImplementedError(f"Unsupported axis: {v}")
            torch_args["dim"] = v
    return torch_args


def _torch_attrs_upsample(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {
        # "size":None,
        "scale_factor": None,
        "mode": "nearest",
        # "align_corners": None,
        # "recompute_scale_factor": None,
    }
    for k, v in attrs.items():
        if k == "scales":
            torch_args["scale_factor"] = v
        elif k == "mode":
            torch_args["mode"] = v
    raise NotImplementedError("This method needs confirmation.")
    return torch_args


def _torch_attrs_reducemean(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    torch_args = {"keepdims": True}
    for k, v in attrs.items():
        if k == "keepdims":
            torch_args["keepdim"] = bool(v)

    return torch_args


_TORCH_ATTRS_MAP = {
    "AveragePool": _torch_attrs_avgpool,
    "BatchNormalization": _torch_attrs_batchnorm,
    "Conv": _torch_attrs_conv,
    "ConvTranspose": _torch_attrs_convtranspose,
    "Elu": _torch_attrs_elu,
    "Flatten": _torch_attrs_flatten,
    "GELU": _torch_attrs_gelu,
    "Gemm": _torch_attrs_gemm,
    "LeakyRelu": _torch_attrs_leakyrelu,
    "MaxPool": _torch_attrs_maxpool,
    "Softmax": _torch_attrs_softmax,
    "Upsample": _torch_attrs_upsample,
    "ReduceMean": _torch_attrs_reducemean,
}


def get_torch_args_of_onnx_attrs(
    node: onnx.NodeProto,
    attrs: dict[str, Any],
    initializer_shapes: dict[str, tuple[int, ...]],
) -> dict[str, Any]:
    _torch_attrs = _TORCH_ATTRS_MAP.get(node.op_type)
    if _torch_attrs is None:
        raise ValueError(f"Invalid onnx.op_type: {node.op_type}")
    torch_args = _torch_attrs(node, attrs, initializer_shapes)

    return torch_args
