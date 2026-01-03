"""Functional operation mappings for ONNX to PyTorch.

This module defines operations that should be implemented as functional calls
rather than as nn.Module layers.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ONNX_TO_PYTORCH_OPERATIONS",
    "ONNX_TO_PYTORCH_OPERATORS",
    "convert_to_operation",
    "convert_to_operator",
    "extract_operation_args",
    "is_operation",
    "is_operator",
]

from typing import Any

import numpy as np
from onnx import NodeProto, TensorProto, numpy_helper

from torchonnx.analyze.attr_extractor import extract_onnx_attrs

ONNX_TO_PYTORCH_OPERATORS: dict[str, str] = {
    # Unary operators
    "Neg": "-",
    # Binary operators
    "MatMul": "@",
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
    "Pow": "**",
    "Equal": "==",
}

ONNX_TO_PYTORCH_OPERATIONS: dict[str, str] = {
    # Fallback for Conv when constructor args fail
    "Conv": "F.conv",
    # Fallback for ConvTranspose when constructor args fail
    "ConvTranspose": "F.conv_transpose",
    # Fallback for Gemm when constructor args fail
    "Gemm": "F.linear",
    "Pad": "F.pad",
    "Reshape": "reshape",
    "Transpose": "permute",
    "Squeeze": "squeeze",
    "Unsqueeze": "unsqueeze",
    "Shape": "shape",
    "Gather": "torch.gather",
    "Cast": "cast",
    "Concat": "torch.cat",
    "Slice": "slice",
    "Sign": "sign",
    "Split": "split",
    "ConstantOfShape": "torch.full",
    "ReduceMean": "torch.mean",
    "ReduceSum": "torch.sum",
    "Cos": "cos",
    "Sin": "sin",
    "Floor": "floor",
    "Expand": "expand",
    "Range": "torch.arange",
    "Where": "torch.where",
    "ScatterND": "scatter_nd",
    "ArgMax": "torch.argmax",
    "Min": "torch.min",
    "Max": "torch.max",
    "Clip": "torch.clamp",
}


def is_operator(layer_type: str) -> bool:
    """Check if operation is a functional operation (not a layer).

    :param layer_type: Operation type
    :return: True if operation is functional
    """
    return layer_type in ONNX_TO_PYTORCH_OPERATORS


def is_operation(layer_type: str) -> bool:
    """Check if operation should be implemented as functional with arguments.

    :param layer_type: PyTorch layer type
    :return: True if operation is functional with arguments
    """
    return layer_type in ONNX_TO_PYTORCH_OPERATIONS


def convert_to_operator(layer_type: str) -> str:
    """Get Python operator for functional operation.

    :param layer_type: Operation type
    :return: Python operator string (e.g., "+", "-", "@")
    """
    if layer_type not in ONNX_TO_PYTORCH_OPERATORS:
        raise ValueError(f"Not a functional operation: {layer_type}")
    return ONNX_TO_PYTORCH_OPERATORS[layer_type]


def convert_to_operation(layer_type: str) -> str:
    """Get the template for generating functional operation code.

    :param layer_type: PyTorch layer type
    :return: Template string for code generation
    """
    if layer_type not in ONNX_TO_PYTORCH_OPERATIONS:
        raise ValueError(f"Not a functional operation with args: {layer_type}")
    return ONNX_TO_PYTORCH_OPERATIONS[layer_type]


def _extract_pad_args(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract arguments for Pad operation.

    :param node: ONNX Pad node
    :param initializers: All ONNX initializers
    :return: Arguments for F.pad call
    """
    attrs = extract_onnx_attrs(node, initializers)
    mode = attrs.get("mode", "constant")
    value = attrs.get("value")
    pads = attrs.get("pads")

    if not pads and len(node.input) >= 2 and node.input[1] in initializers:
        pads_array = numpy_helper.to_array(initializers[node.input[1]])
        pads = pads_array.tolist()

    return {"pad": pads, "mode": mode, "value": value}


def _extract_squeeze_unsqueeze_args(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Extract arguments for Squeeze/Unsqueeze operations."""
    attrs = extract_onnx_attrs(node, initializers)
    axes = attrs.get("axes", [])

    # In opset 13+, axes moved from attribute to input (second input)
    if not axes and len(node.input) >= 2:
        axes_name = node.input[1]
        if axes_name in initializers:
            axes_array = numpy_helper.to_array(initializers[axes_name])
            axes = tuple(axes_array.tolist())

    return {"dim": axes[0] if len(axes) == 1 else tuple(axes)} if axes else {}


def _extract_constant_of_shape_args(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Extract arguments for ConstantOfShape operation."""
    attrs = extract_onnx_attrs(node, initializers)
    value = attrs.get("value")
    if value is not None:
        # Keep the numpy array to preserve dtype information
        # The codegen phase will extract the scalar value and dtype
        if isinstance(value, np.ndarray):
            return {"value": value}
        # Fallback for TensorProto
        value_array = numpy_helper.to_array(value)
        return {"value": value_array}
    # Return 0-dimensional array to preserve type information
    return {"value": np.array([0.0])}


def _simplify_homogeneous_values(
    values: list[int], skip_default: int | None = None
) -> int | tuple[int, ...]:
    """Simplify homogeneous values into scalar or tuple.

    :param values: List of values
    :param skip_default: Default value to skip if not needed
    :return: Scalar if all same, tuple otherwise
    """
    if not values:
        return ()
    if all(v == values[0] for v in values):
        return values[0]
    return tuple(values)


def _process_conv_padding(pads: list[int], operation: str) -> int | tuple[int, ...]:
    """Convert and validate ONNX padding to PyTorch format.

    :param pads: ONNX padding values
    :param operation: Operation name (Conv or ConvTranspose)
    :return: PyTorch padding format
    :raises ValueError: If padding is asymmetric
    """
    ndims = len(pads) // 2
    symmetric = all(pads[i] == pads[i + ndims] for i in range(ndims))

    if not symmetric:
        raise ValueError(
            f"Asymmetric padding {pads} not supported for F.{operation.lower()}. "
            f"PyTorch F.{operation.lower()} only supports symmetric padding."
        )

    pad_values = pads[:ndims]
    return _simplify_homogeneous_values(pad_values)


def _extract_conv_args(node: NodeProto, initializers: dict[str, TensorProto]) -> dict[str, Any]:
    """Extract arguments for Conv operation."""
    attrs = extract_onnx_attrs(node, initializers)
    strides = attrs.get("strides")
    pads = attrs.get("pads")
    dilations = attrs.get("dilations")
    groups = attrs.get("group", 1)

    pytorch_args: dict[str, Any] = {}

    if strides:
        pytorch_args["stride"] = _simplify_homogeneous_values(strides)

    if pads:
        pytorch_args["padding"] = _process_conv_padding(pads, "Conv")

    if dilations:
        simplified = _simplify_homogeneous_values(dilations)
        if simplified != 1:
            pytorch_args["dilation"] = simplified

    if groups != 1:
        pytorch_args["groups"] = groups

    return pytorch_args


def _extract_conv_transpose_args(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Extract arguments for ConvTranspose operation."""
    attrs = extract_onnx_attrs(node, initializers)
    strides = attrs.get("strides")
    pads = attrs.get("pads")
    dilations = attrs.get("dilations")
    groups = attrs.get("group", 1)
    output_padding = attrs.get("output_padding")

    pytorch_args: dict[str, Any] = {}

    if strides:
        pytorch_args["stride"] = _simplify_homogeneous_values(strides)

    if pads:
        pytorch_args["padding"] = _process_conv_padding(pads, "ConvTranspose")

    if output_padding:
        simplified = _simplify_homogeneous_values(output_padding)
        if simplified != 0:
            pytorch_args["output_padding"] = simplified

    if dilations:
        simplified = _simplify_homogeneous_values(dilations)
        if simplified != 1:
            pytorch_args["dilation"] = simplified

    if groups != 1:
        pytorch_args["groups"] = groups

    return pytorch_args


# Dispatch dictionary for operation argument extraction
_OPERATION_ARGS_EXTRACTORS: dict[str, Any] = {
    "Pad": _extract_pad_args,
    "Reshape": lambda node, initializers: {},
    "Transpose": lambda node, initializers: {
        "perm": extract_onnx_attrs(node, initializers).get("perm", [])
    },
    "Squeeze": _extract_squeeze_unsqueeze_args,
    "Unsqueeze": _extract_squeeze_unsqueeze_args,
    "Shape": lambda node, initializers: {},
    "Gather": lambda node, initializers: {
        "axis": extract_onnx_attrs(node, initializers).get("axis", 0)
    },
    "Cast": lambda node, initializers: {"to": extract_onnx_attrs(node, initializers).get("to", 1)},
    "Concat": lambda node, initializers: {
        "axis": extract_onnx_attrs(node, initializers).get("axis", 0)
    },
    "Gemm": lambda node, initializers: {
        "alpha": extract_onnx_attrs(node, initializers).get("alpha", 1.0),
        "beta": extract_onnx_attrs(node, initializers).get("beta", 1.0),
        "transA": extract_onnx_attrs(node, initializers).get("transA", 0),
        "transB": extract_onnx_attrs(node, initializers).get("transB", 0),
    },
    "Slice": lambda node, initializers: {
        "starts": extract_onnx_attrs(node, initializers).get("starts", []),
        "ends": extract_onnx_attrs(node, initializers).get("ends", []),
        "axes": extract_onnx_attrs(node, initializers).get("axes", None),
    },
    "Sign": lambda node, initializers: {},
    "Split": lambda node, initializers: {
        "axis": extract_onnx_attrs(node, initializers).get("axis", 0)
    },
    "ConstantOfShape": _extract_constant_of_shape_args,
    "ReduceMean": lambda node, initializers: {
        "axes": extract_onnx_attrs(node, initializers).get("axes", None),
        "keepdims": bool(extract_onnx_attrs(node, initializers).get("keepdims", 1)),
    },
    "ReduceSum": lambda node, initializers: {
        "axes": extract_onnx_attrs(node, initializers).get("axes", None),
        "keepdims": bool(extract_onnx_attrs(node, initializers).get("keepdims", 1)),
    },
    "Cos": lambda node, initializers: {},
    "Sin": lambda node, initializers: {},
    "Floor": lambda node, initializers: {},
    "Neg": lambda node, initializers: {},
    "Expand": lambda node, initializers: {},
    "Range": lambda node, initializers: {},
    "Where": lambda node, initializers: {},
    "ScatterND": lambda node, initializers: {
        "reduction": extract_onnx_attrs(node, initializers).get("reduction", "none")
    },
    "ArgMax": lambda node, initializers: {
        "dim": extract_onnx_attrs(node, initializers).get("axis", 0),
        "keepdim": bool(extract_onnx_attrs(node, initializers).get("keepdims", 1)),
    },
    "Min": lambda node, initializers: {},
    "Max": lambda node, initializers: {},
    "Clip": lambda node, initializers: {
        "min": extract_onnx_attrs(node, initializers).get("min"),
        "max": extract_onnx_attrs(node, initializers).get("max"),
    },
    "Conv": _extract_conv_args,
    "ConvTranspose": _extract_conv_transpose_args,
}


def extract_operation_args(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    layer_type: str,
) -> dict[str, Any]:
    """Extract arguments for functional operation.

    :param node: ONNX node
    :param initializers: All ONNX initializers
    :param layer_type: PyTorch layer type
    :return: Arguments for functional call
    """
    extractor = _OPERATION_ARGS_EXTRACTORS.get(layer_type)
    if extractor is None:
        return {}
    return extractor(node, initializers)  # type: ignore[no-any-return]
