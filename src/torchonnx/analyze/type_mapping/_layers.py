"""ONNX to PyTorch operator type mapping.

This module provides utilities for inferring PyTorch layer types from ONNX operators.
Reference: torchonnx/_torch_args.py _TORCH_ATTRS_MAP
"""

__docformat__ = "restructuredtext"
__all__ = ["ONNX_TO_PYTORCH_LAYERS", "extract_layer_args", "is_layer_with_args"]

from typing import Any

from onnx import NodeProto, TensorProto

from torchonnx.analyze.attr_extractor import extract_onnx_attrs

ONNX_TO_PYTORCH_LAYERS: dict[str, str] = {
    # Convolution - Note: Conv/ConvTranspose require weight shape inspection
    # "Conv": "Conv2d",  # Determined dynamically based on weight shape
    # "ConvTranspose": "ConvTranspose2d",  # Determined dynamically based on weight
    # shape
    # Pooling
    "MaxPool": "MaxPool2d",
    "AveragePool": "AvgPool2d",
    "GlobalAveragePool": "AdaptiveAvgPool2d",
    # Linear and matrix operations
    "Gemm": "Linear",
    # Normalization
    "BatchNormalization": "BatchNorm2d",
    # Activation functions
    "Relu": "ReLU",
    "LeakyRelu": "LeakyReLU",
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "Softmax": "Softmax",
    "Elu": "ELU",
    "Gelu": "GELU",
    # Dropout and regularization
    "Dropout": "Dropout",
    # Upsampling
    "Resize": "Upsample",
    "Upsample": "Upsample",
    # Shape operations
    "Flatten": "Flatten",
}


LAYERS_WITH_ARGS: set[str] = {
    "Conv1d",
    "Conv2d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "Linear",
    "BatchNorm2d",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    "Dropout",
    "ReLU",
    "LeakyReLU",
    "ELU",
    "GELU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "Flatten",
    "Upsample",
}


def is_layer_with_args(layer_type: str) -> bool:
    """Check if layer type has learnable parameters.

    :param layer_type: PyTorch layer type (e.g., "nn.Conv2d" or "Conv2d")
    :return: True if layer has learnable parameters
    """
    # Strip nn. prefix if present
    layer_type = layer_type.removeprefix("nn.")
    return layer_type in LAYERS_WITH_ARGS


def _simplify_tuple(arg: tuple[int, ...] | int) -> tuple[int, ...] | int:
    """Simplify tuple argument if all elements are equal.

    :param arg: Tuple or int argument
    :return: Simplified argument (int if all elements equal, otherwise tuple)
    """
    if isinstance(arg, tuple):
        if len(arg) == 0:
            return arg
        if len(arg) == 1 or all(x == arg[0] for x in arg):
            return arg[0]
    return arg


def _check_symmetric_padding(pads: tuple[int, ...]) -> None:
    """Check that padding is symmetric.

    :param pads: ONNX padding in format [start_h, start_w, end_h, end_w]
    """
    length = len(pads)
    dims = length // 2
    for i in range(dims):
        if pads[i] != pads[i + dims]:
            raise ValueError(
                f"Asymmetric padding {pads} not supported. Start and end padding must be equal."
            )


def _extract_conv_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Conv attributes to PyTorch Conv2d constructor arguments.

    :param node: ONNX Conv node
    :param initializers: All ONNX initializers
    :return: PyTorch Conv2d constructor arguments (empty dict for dynamic Conv)
    """
    if len(node.input) < 2 or node.input[1] not in initializers:
        return {}

    attrs = extract_onnx_attrs(node, initializers)

    weight_shape = tuple(initializers[node.input[1]].dims)
    has_bias = len(node.input) >= 3 and node.input[2] in initializers

    kernel_shape = attrs.get("kernel_shape")
    if kernel_shape is None:
        kernel_shape = weight_shape[2:]

    strides = attrs.get("strides")
    if strides is None:
        strides = tuple([1] * len(kernel_shape))

    pads = attrs.get("pads")
    if pads is None:
        pads = tuple([0] * len(kernel_shape) * 2)
    _check_symmetric_padding(pads)

    dilations = attrs.get("dilations")
    if dilations is None:
        dilations = tuple([1] * len(kernel_shape))

    groups = attrs.get("group", 1)

    torch_args = {
        "in_channels": weight_shape[1] * groups,
        "out_channels": weight_shape[0],
        "kernel_size": _simplify_tuple(kernel_shape),
        "stride": _simplify_tuple(strides),
        "padding": _simplify_tuple(pads[: len(kernel_shape)]),
        "dilation": _simplify_tuple(dilations),
        "groups": groups,
        "bias": has_bias,
    }

    return torch_args


def _extract_batchnorm_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX BatchNormalization attributes to PyTorch BatchNorm2d arguments.

    :param node: ONNX BatchNormalization node
    :param initializers: All ONNX initializers
    :return: PyTorch BatchNorm2d constructor arguments
    """
    attrs = extract_onnx_attrs(node, initializers)

    bias_tensor = initializers[node.input[2]]
    num_features = bias_tensor.dims[0]

    eps = attrs.get("epsilon", 1e-5)

    onnx_momentum = attrs.get("momentum", 0.9)
    pytorch_momentum = 1.0 - onnx_momentum

    torch_args = {
        "num_features": num_features,
        "eps": eps,
        "momentum": pytorch_momentum,
        "track_running_stats": True,
    }

    return torch_args


def _extract_gemm_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Gemm attributes to PyTorch Linear constructor arguments.

    :param node: ONNX Gemm node
    :param initializers: All ONNX initializers
    :return: PyTorch Linear constructor arguments (empty dict for dynamic Gemm)
    """
    if any(inp not in initializers for inp in node.input[1:]):
        return {}

    attrs = extract_onnx_attrs(node, initializers)

    weight_shape = tuple(initializers[node.input[1]].dims)

    trans_b = attrs.get("transB", 0)
    if trans_b == 1:
        in_features = weight_shape[1]
        out_features = weight_shape[0]
    else:
        in_features = weight_shape[0]
        out_features = weight_shape[1]

    has_bias = len(node.input) >= 3 and node.input[2] in initializers

    torch_args = {
        "in_features": in_features,
        "out_features": out_features,
        "bias": has_bias,
    }

    return torch_args


def _extract_convtranspose_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX ConvTranspose attributes to PyTorch ConvTranspose2d arguments.

    :param node: ONNX ConvTranspose node
    :param initializers: All ONNX initializers
    :return: PyTorch ConvTranspose2d constructor arguments
    """
    if any(inp not in initializers for inp in node.input[1:]):
        return {}

    attrs = extract_onnx_attrs(node, initializers)

    weight_shape = tuple(initializers[node.input[1]].dims)
    has_bias = len(node.input) >= 3 and node.input[2] in initializers

    kernel_shape = attrs.get("kernel_shape")
    if kernel_shape is None:
        kernel_shape = weight_shape[2:]

    strides = attrs.get("strides")
    if strides is None:
        strides = tuple([1] * len(kernel_shape))

    pads = attrs.get("pads")
    if pads is None:
        pads = tuple([0] * len(kernel_shape) * 2)
    _check_symmetric_padding(pads)

    dilations = attrs.get("dilations")
    if dilations is None:
        dilations = tuple([1] * len(kernel_shape))

    groups = attrs.get("group", 1)

    output_padding = attrs.get("output_padding")
    if output_padding is None:
        output_padding = tuple([0] * len(kernel_shape))

    torch_args = {
        "in_channels": weight_shape[0],
        "out_channels": weight_shape[1] * groups,
        "kernel_size": _simplify_tuple(kernel_shape),
        "stride": _simplify_tuple(strides),
        "padding": _simplify_tuple(pads[: len(kernel_shape)]),
        "output_padding": _simplify_tuple(output_padding),
        "dilation": _simplify_tuple(dilations),
        "groups": groups,
        "bias": has_bias,
    }

    return torch_args


def _extract_relu_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Relu to PyTorch ReLU arguments.

    :param node: ONNX Relu node
    :param initializers: All ONNX initializers
    :return: PyTorch ReLU constructor arguments (empty dict)
    """
    return {}


def _extract_averagepool_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX AveragePool attributes to PyTorch AvgPool2d arguments.

    :param node: ONNX AveragePool node
    :param initializers: All ONNX initializers
    :return: PyTorch AvgPool2d constructor arguments
    """
    attrs = extract_onnx_attrs(node, initializers)

    kernel_shape = attrs.get("kernel_shape")
    if kernel_shape is None:
        raise ValueError("AveragePool requires kernel_shape attribute")

    strides = attrs.get("strides")
    if strides is None:
        strides = kernel_shape

    pads = attrs.get("pads")
    if pads is None:
        pads = tuple([0] * len(kernel_shape) * 2)
    _check_symmetric_padding(pads)

    ceil_mode = attrs.get("ceil_mode", 0)
    count_include_pad = attrs.get("count_include_pad", 1)

    torch_args = {
        "kernel_size": kernel_shape,
    }

    # Only add non-default arguments
    # PyTorch defaults: stride=None (defaults to kernel_size), padding=0,
    # ceil_mode=False, count_include_pad=True
    simplified_strides = _simplify_tuple(strides)
    simplified_kernel = (
        kernel_shape if isinstance(kernel_shape, int) else _simplify_tuple(kernel_shape)
    )
    if simplified_strides != simplified_kernel:
        torch_args["stride"] = simplified_strides

    simplified_pads = _simplify_tuple(pads[: len(kernel_shape)])
    if simplified_pads != 0:
        torch_args["padding"] = simplified_pads

    if bool(ceil_mode):
        torch_args["ceil_mode"] = True

    if not bool(count_include_pad):
        torch_args["count_include_pad"] = False

    return torch_args


def _extract_maxpool_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX MaxPool attributes to PyTorch MaxPool2d arguments.

    :param node: ONNX MaxPool node
    :param initializers: All ONNX initializers
    :return: PyTorch MaxPool2d constructor arguments
    """
    attrs = extract_onnx_attrs(node, initializers)

    kernel_shape = attrs.get("kernel_shape")
    if kernel_shape is None:
        raise ValueError("MaxPool requires kernel_shape attribute")

    strides = attrs.get("strides")
    pads = attrs.get("pads")
    if pads is None:
        pads = tuple([0] * len(kernel_shape) * 2)
    _check_symmetric_padding(pads)

    dilations = attrs.get("dilations")
    if dilations is None:
        dilations = tuple([1] * len(kernel_shape))

    ceil_mode = attrs.get("ceil_mode", 0)

    torch_args = {
        "kernel_size": kernel_shape,
        "stride": _simplify_tuple(strides) if strides else kernel_shape,
        "padding": _simplify_tuple(pads[: len(kernel_shape)]),
        "dilation": _simplify_tuple(dilations),
        "ceil_mode": bool(ceil_mode),
    }

    return torch_args


def _extract_dropout_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Dropout attributes to PyTorch Dropout arguments.

    Handles both opset < 12 (ratio as attribute) and opset >= 12 (ratio as input).

    :param node: ONNX Dropout node
    :param initializers: All ONNX initializers
    :return: PyTorch Dropout constructor arguments
    """
    from onnx import numpy_helper

    if len(node.input) > 1 and node.input[1] in initializers:
        ratio = float(numpy_helper.to_array(initializers[node.input[1]]))
        return {"p": ratio}

    attrs = extract_onnx_attrs(node, initializers)
    ratio = attrs.get("ratio", 0.5)
    return {"p": ratio}


def _extract_elu_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Elu attributes to PyTorch ELU arguments.

    :param node: ONNX Elu node
    :param initializers: All ONNX initializers
    :return: PyTorch ELU constructor arguments
    """
    attrs = extract_onnx_attrs(node, initializers)
    alpha = attrs.get("alpha", 1.0)
    return {"alpha": alpha}


def _extract_leakyrelu_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX LeakyRelu attributes to PyTorch LeakyReLU arguments.

    :param node: ONNX LeakyRelu node
    :param initializers: All ONNX initializers
    :return: PyTorch LeakyReLU constructor arguments
    """
    attrs = extract_onnx_attrs(node, initializers)
    alpha = attrs.get("alpha", 0.01)
    return {"negative_slope": alpha}


def _extract_softmax_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Softmax attributes to PyTorch Softmax arguments.

    :param node: ONNX Softmax node
    :param initializers: All ONNX initializers
    :return: PyTorch Softmax constructor arguments
    """
    attrs = extract_onnx_attrs(node, initializers)
    axis = attrs.get("axis")
    return {"dim": axis} if axis is not None else {}


def _extract_gelu_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Gelu attributes to PyTorch GELU arguments.

    :param node: ONNX Gelu node
    :param initializers: All ONNX initializers
    :return: PyTorch GELU constructor arguments
    """
    attrs = extract_onnx_attrs(node, initializers)
    approximation = attrs.get("approximation", "none")
    return {"approximate": approximation} if approximation != "none" else {}


def _extract_upsample_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Upsample attributes to PyTorch Upsample arguments.

    :param node: ONNX Upsample node
    :param initializers: All ONNX initializers
    :return: PyTorch Upsample constructor arguments
    """
    from onnx import numpy_helper

    attrs = extract_onnx_attrs(node, initializers)
    mode = attrs.get("mode", "nearest")

    # ONNX Resize has inputs: X, roi, scales, [sizes]
    # roi at input[1] is often empty, scales at input[2] or input[3]
    # Check input[2] first (scales), then input[1] as fallback

    # Check input[2] for scales (typical for opset 10+)
    if len(node.input) > 2 and node.input[2] in initializers:
        scales = numpy_helper.to_array(initializers[node.input[2]])
        if len(scales) > 0:  # Make sure scales is not empty
            if len(scales) > 2:
                scales = scales[2:]  # Skip batch and channel dims
            return {"scale_factor": tuple(scales.tolist()), "mode": mode}

    # Fallback: check input[1] for older opset versions
    if len(node.input) > 1 and node.input[1] in initializers:
        scales = numpy_helper.to_array(initializers[node.input[1]])
        if len(scales) > 0:  # Make sure scales is not empty
            if len(scales) > 2:
                scales = scales[2:]
            return {"scale_factor": tuple(scales.tolist()), "mode": mode}

    return {"mode": mode}


def _extract_sigmoid_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Sigmoid to PyTorch Sigmoid arguments.

    :param node: ONNX Sigmoid node
    :param initializers: All ONNX initializers
    :return: PyTorch Sigmoid constructor arguments (empty dict)
    """
    return {}


def _extract_tanh_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Tanh to PyTorch Tanh arguments.

    :param node: ONNX Tanh node
    :param initializers: All ONNX initializers
    :return: PyTorch Tanh constructor arguments (empty dict)
    """
    return {}


def _extract_globalavgpool_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX GlobalAveragePool to PyTorch AdaptiveAvgPool2d arguments.

    :param node: ONNX GlobalAveragePool node
    :param initializers: All ONNX initializers
    :return: PyTorch AdaptiveAvgPool2d constructor arguments
    """
    return {"output_size": (1, 1)}


def _extract_flatten_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Flatten attributes to PyTorch Flatten arguments.

    :param node: ONNX Flatten node
    :param initializers: All ONNX initializers
    :return: PyTorch Flatten constructor arguments
    """
    attrs = extract_onnx_attrs(node, initializers)
    axis = attrs.get("axis", 1)

    return {
        "start_dim": axis,
    }


_ONNX_TO_PYTORCH_ARGS: dict[str, Any] = {
    "Conv": _extract_conv_args,
    "ConvTranspose": _extract_convtranspose_args,
    "Gemm": _extract_gemm_args,
    "BatchNormalization": _extract_batchnorm_args,
    "MaxPool": _extract_maxpool_args,
    "AveragePool": _extract_averagepool_args,
    "Dropout": _extract_dropout_args,
    "Relu": _extract_relu_args,  # ONNX uses "Relu" (lowercase u)
    "ReLU": _extract_relu_args,  # Some models use "ReLU" (uppercase U)
    "LeakyRelu": _extract_leakyrelu_args,
    "Elu": _extract_elu_args,
    "Gelu": _extract_gelu_args,
    "Sigmoid": _extract_sigmoid_args,
    "Tanh": _extract_tanh_args,
    "Softmax": _extract_softmax_args,
    "Upsample": _extract_upsample_args,
    "Resize": _extract_upsample_args,  # Resize also maps to nn.Upsample
    "GlobalAveragePool": _extract_globalavgpool_args,
    "Flatten": _extract_flatten_args,
}


def extract_layer_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX node attributes to PyTorch constructor arguments.

    :param node: ONNX node
    :param initializers: All ONNX initializers
    :return: PyTorch layer constructor arguments
    """
    mapping_func = _ONNX_TO_PYTORCH_ARGS.get(node.op_type)

    if mapping_func is None:
        raise ValueError(
            f"Unsupported ONNX operator: {node.op_type}. "
            f"Phase 1 supports: {sorted(_ONNX_TO_PYTORCH_ARGS.keys())}"
        )

    return mapping_func(node, initializers)  # type: ignore[no-any-return]
