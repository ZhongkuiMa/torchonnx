"""ONNX to PyTorch constructor argument mapping.

This module maps ONNX node attributes to PyTorch layer constructor arguments.
Phase 1 implements Conv2d, BatchNorm2d, and ReLU only.
"""

__docformat__ = "restructuredtext"
__all__ = ["map_onnx_to_pytorch_args"]

from typing import Any

from onnx import NodeProto, TensorProto


def simplify_tuple(arg: tuple[int, ...] | int) -> tuple[int, ...] | int:
    """Simplify tuple argument if all elements are equal.

    :param arg: Tuple or int argument
    :return: Simplified argument (int if all elements equal, otherwise tuple)
    """
    if isinstance(arg, tuple):
        if len(arg) == 0:
            return arg
        elif len(arg) == 1:
            return arg[0]
        elif all(x == arg[0] for x in arg):
            return arg[0]
    return arg


def _extract_onnx_attributes(node: NodeProto) -> dict[str, Any]:
    """Extract ONNX attributes from node.

    :param node: ONNX node
    :return: Dictionary of attribute name to value
    """
    attrs: dict[str, Any] = {}

    for attr in node.attribute:
        if attr.type == 1:
            attrs[attr.name] = float(attr.f)
        elif attr.type == 2:
            attrs[attr.name] = int(attr.i)
        elif attr.type == 3:
            attrs[attr.name] = str(attr.s, encoding="utf-8")
        elif attr.type == 6:
            attrs[attr.name] = tuple(list(attr.floats))
        elif attr.type == 7:
            attrs[attr.name] = tuple(list(attr.ints))

    return attrs


def _check_symmetric_padding(pads: tuple[int, ...]) -> None:
    """Check that padding is symmetric.

    :param pads: ONNX padding in format [start_h, start_w, end_h, end_w]
    """
    length = len(pads)
    dims = length // 2
    for i in range(dims):
        if pads[i] != pads[i + dims]:
            raise ValueError(
                f"Asymmetric padding {pads} not supported. "
                f"Start and end padding must be equal."
            )


def map_conv_args(
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

    attrs = _extract_onnx_attributes(node)

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
        "kernel_size": kernel_shape,
        "stride": simplify_tuple(strides),
        "padding": simplify_tuple(pads[: len(kernel_shape)]),
        "dilation": simplify_tuple(dilations),
        "groups": groups,
        "bias": has_bias,
    }

    return torch_args


def map_batchnorm_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX BatchNormalization attributes to PyTorch BatchNorm2d arguments.

    :param node: ONNX BatchNormalization node
    :param initializers: All ONNX initializers
    :return: PyTorch BatchNorm2d constructor arguments
    """
    attrs = _extract_onnx_attributes(node)

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


def map_gemm_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Gemm attributes to PyTorch Linear constructor arguments.

    :param node: ONNX Gemm node
    :param initializers: All ONNX initializers
    :return: PyTorch Linear constructor arguments (empty dict for dynamic Gemm)
    """
    if len(node.input) < 2 or node.input[1] not in initializers:
        return {}

    attrs = _extract_onnx_attributes(node)

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


def map_convtranspose_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX ConvTranspose attributes to PyTorch ConvTranspose2d arguments.

    :param node: ONNX ConvTranspose node
    :param initializers: All ONNX initializers
    :return: PyTorch ConvTranspose2d constructor arguments
    """
    attrs = _extract_onnx_attributes(node)

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
        "kernel_size": kernel_shape,
        "stride": simplify_tuple(strides),
        "padding": simplify_tuple(pads[: len(kernel_shape)]),
        "output_padding": simplify_tuple(output_padding),
        "dilation": simplify_tuple(dilations),
        "groups": groups,
        "bias": has_bias,
    }

    return torch_args


def map_relu_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Relu to PyTorch ReLU arguments.

    :param node: ONNX Relu node
    :param initializers: All ONNX initializers
    :return: PyTorch ReLU constructor arguments (empty dict)
    """
    return {}


def map_averagepool_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX AveragePool attributes to PyTorch AvgPool2d arguments.

    :param node: ONNX AveragePool node
    :param initializers: All ONNX initializers
    :return: PyTorch AvgPool2d constructor arguments
    """
    attrs = _extract_onnx_attributes(node)

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
        "stride": simplify_tuple(strides),
        "padding": simplify_tuple(pads[: len(kernel_shape)]),
        "ceil_mode": bool(ceil_mode),
        "count_include_pad": bool(count_include_pad),
    }

    return torch_args


def map_maxpool_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX MaxPool attributes to PyTorch MaxPool2d arguments.

    :param node: ONNX MaxPool node
    :param initializers: All ONNX initializers
    :return: PyTorch MaxPool2d constructor arguments
    """
    attrs = _extract_onnx_attributes(node)

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
        "stride": simplify_tuple(strides) if strides else kernel_shape,
        "padding": simplify_tuple(pads[: len(kernel_shape)]),
        "dilation": simplify_tuple(dilations),
        "ceil_mode": bool(ceil_mode),
    }

    return torch_args


def map_dropout_args(
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

    attrs = _extract_onnx_attributes(node)
    ratio = attrs.get("ratio", 0.5)
    return {"p": ratio}


def map_elu_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Elu attributes to PyTorch ELU arguments.

    :param node: ONNX Elu node
    :param initializers: All ONNX initializers
    :return: PyTorch ELU constructor arguments
    """
    attrs = _extract_onnx_attributes(node)
    alpha = attrs.get("alpha", 1.0)
    return {"alpha": alpha}


def map_leakyrelu_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX LeakyRelu attributes to PyTorch LeakyReLU arguments.

    :param node: ONNX LeakyRelu node
    :param initializers: All ONNX initializers
    :return: PyTorch LeakyReLU constructor arguments
    """
    attrs = _extract_onnx_attributes(node)
    alpha = attrs.get("alpha", 0.01)
    return {"negative_slope": alpha}


def map_softmax_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Softmax attributes to PyTorch Softmax arguments.

    :param node: ONNX Softmax node
    :param initializers: All ONNX initializers
    :return: PyTorch Softmax constructor arguments
    """
    attrs = _extract_onnx_attributes(node)
    axis = attrs.get("axis")
    return {"dim": axis} if axis is not None else {}


def map_gelu_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Gelu attributes to PyTorch GELU arguments.

    :param node: ONNX Gelu node
    :param initializers: All ONNX initializers
    :return: PyTorch GELU constructor arguments
    """
    attrs = _extract_onnx_attributes(node)
    approximation = attrs.get("approximation", "none")
    return {"approximate": approximation} if approximation != "none" else {}


def map_upsample_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Upsample attributes to PyTorch Upsample arguments.

    :param node: ONNX Upsample node
    :param initializers: All ONNX initializers
    :return: PyTorch Upsample constructor arguments
    """
    from onnx import numpy_helper

    attrs = _extract_onnx_attributes(node)
    mode = attrs.get("mode", "nearest")

    if len(node.input) > 1 and node.input[1] in initializers:
        scales = numpy_helper.to_array(initializers[node.input[1]])
        if len(scales) > 2:
            scales = scales[2:]
        return {"scale_factor": tuple(scales.tolist()), "mode": mode}

    if len(node.input) > 2 and node.input[2] in initializers:
        scales = numpy_helper.to_array(initializers[node.input[2]])
        if len(scales) > 2:
            scales = scales[2:]
        return {"scale_factor": tuple(scales.tolist()), "mode": mode}

    return {"mode": mode}


def map_sigmoid_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Sigmoid to PyTorch Sigmoid arguments.

    :param node: ONNX Sigmoid node
    :param initializers: All ONNX initializers
    :return: PyTorch Sigmoid constructor arguments (empty dict)
    """
    return {}


def map_tanh_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX Tanh to PyTorch Tanh arguments.

    :param node: ONNX Tanh node
    :param initializers: All ONNX initializers
    :return: PyTorch Tanh constructor arguments (empty dict)
    """
    return {}


def map_globalavgpool_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX GlobalAveragePool to PyTorch AdaptiveAvgPool2d arguments.

    :param node: ONNX GlobalAveragePool node
    :param initializers: All ONNX initializers
    :return: PyTorch AdaptiveAvgPool2d constructor arguments
    """
    return {"output_size": (1, 1)}


_ARGUMENT_MAPPING_FUNCTIONS: dict[str, Any] = {
    "Conv": map_conv_args,
    "ConvTranspose": map_convtranspose_args,
    "Gemm": map_gemm_args,
    "BatchNormalization": map_batchnorm_args,
    "Relu": map_relu_args,
    "AveragePool": map_averagepool_args,
    "MaxPool": map_maxpool_args,
    "Dropout": map_dropout_args,
    "Elu": map_elu_args,
    "LeakyRelu": map_leakyrelu_args,
    "Softmax": map_softmax_args,
    "Gelu": map_gelu_args,
    "Upsample": map_upsample_args,
    "Sigmoid": map_sigmoid_args,
    "Tanh": map_tanh_args,
    "GlobalAveragePool": map_globalavgpool_args,
}


def map_onnx_to_pytorch_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
) -> dict[str, Any]:
    """Map ONNX node attributes to PyTorch constructor arguments.

    :param node: ONNX node
    :param initializers: All ONNX initializers
    :return: PyTorch layer constructor arguments
    """
    mapping_func = _ARGUMENT_MAPPING_FUNCTIONS.get(node.op_type)

    if mapping_func is None:
        raise ValueError(
            f"Unsupported ONNX operator: {node.op_type}. "
            f"Phase 1 supports: {sorted(_ARGUMENT_MAPPING_FUNCTIONS.keys())}"
        )

    return mapping_func(node, initializers)
