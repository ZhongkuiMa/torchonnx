"""Functional operation mappings for ONNX to PyTorch.

This module defines operations that should be implemented as functional calls
rather than as nn.Module layers.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ONNX_TO_PYTORCH_OPERATORS",
    "ONNX_TO_PYTORCH_OPERATIONS",
    "is_operator",
    "convert_to_operator",
    "is_operation",
    "convert_to_operation",
    "extract_operation_args",
]

from typing import Any

from onnx import numpy_helper, NodeProto, TensorProto

from ..attr_extractor import extract_onnx_attrs

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


def _extract_pad_args(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
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

    if layer_type == "Pad":
        return _extract_pad_args(node, initializers)
    elif layer_type == "Reshape":
        return {}
    elif layer_type == "Transpose":
        attrs = extract_onnx_attrs(node, initializers)
        return {"perm": attrs.get("perm", [])}
    elif layer_type == "Squeeze":
        attrs = extract_onnx_attrs(node, initializers)
        axes = attrs.get("axes", [])

        # In opset 13+, axes moved from attribute to input (second input)
        if not axes and len(node.input) >= 2:
            axes_name = node.input[1]
            if axes_name in initializers:
                axes_array = numpy_helper.to_array(initializers[axes_name])
                axes = tuple(axes_array.tolist())

        return {"dim": axes[0] if len(axes) == 1 else tuple(axes)} if axes else {}
    elif layer_type == "Unsqueeze":
        attrs = extract_onnx_attrs(node, initializers)
        axes = attrs.get("axes", [])

        # In opset 13+, axes moved from attribute to input (second input)
        if not axes and len(node.input) >= 2:
            axes_name = node.input[1]
            if axes_name in initializers:
                axes_array = numpy_helper.to_array(initializers[axes_name])
                axes = tuple(axes_array.tolist())

        return {"dim": axes[0] if len(axes) == 1 else tuple(axes)} if axes else {}
    elif layer_type == "Shape":
        return {}
    elif layer_type == "Gather":
        attrs = extract_onnx_attrs(node, initializers)
        return {"axis": attrs.get("axis", 0)}
    elif layer_type == "Cast":
        attrs = extract_onnx_attrs(node, initializers)
        return {"to": attrs.get("to", 1)}
    elif layer_type == "Concat":
        attrs = extract_onnx_attrs(node, initializers)
        return {"axis": attrs.get("axis", 0)}
    elif layer_type == "Gemm":
        attrs = extract_onnx_attrs(node, initializers)
        return {
            "alpha": attrs.get("alpha", 1.0),
            "beta": attrs.get("beta", 1.0),
            "transA": attrs.get("transA", 0),
            "transB": attrs.get("transB", 0),
        }
    elif layer_type == "Slice":
        attrs = extract_onnx_attrs(node, initializers)
        return {
            "starts": attrs.get("starts", []),
            "ends": attrs.get("ends", []),
            "axes": attrs.get("axes", None),
        }
    elif layer_type == "Sign":
        return {}
    elif layer_type == "Split":
        attrs = extract_onnx_attrs(node, initializers)
        return {"axis": attrs.get("axis", 0)}
    elif layer_type == "ConstantOfShape":
        attrs = extract_onnx_attrs(node, initializers)
        value = attrs.get("value")
        if value is not None:
            import numpy as np

            # Keep the numpy array to preserve dtype information
            # The codegen phase will extract the scalar value and dtype
            if isinstance(value, np.ndarray):
                return {"value": value}
            else:
                # Fallback for TensorProto
                # numpy_helper is already imported at module level
                value_array = numpy_helper.to_array(value)
                return {"value": value_array}
        # Return 0-dimensional array to preserve type information
        import numpy as np

        return {"value": np.array([0.0])}
    elif layer_type == "ReduceMean":
        attrs = extract_onnx_attrs(node, initializers)
        axes = attrs.get("axes", None)
        keepdims = attrs.get("keepdims", 1)
        return {"axes": axes, "keepdims": bool(keepdims)}
    elif layer_type == "ReduceSum":
        attrs = extract_onnx_attrs(node, initializers)
        axes = attrs.get("axes", None)
        keepdims = attrs.get("keepdims", 1)
        return {"axes": axes, "keepdims": bool(keepdims)}
    elif layer_type == "Cos":
        return {}
    elif layer_type == "Sin":
        return {}
    elif layer_type == "Floor":
        return {}
    elif layer_type == "Neg":
        return {}
    elif layer_type == "Expand":
        return {}
    elif layer_type == "Range":
        return {}
    elif layer_type == "Where":
        return {}
    elif layer_type == "ScatterND":
        attrs = extract_onnx_attrs(node, initializers)
        return {"reduction": attrs.get("reduction", "none")}
    elif layer_type == "ArgMax":
        attrs = extract_onnx_attrs(node, initializers)
        return {
            "dim": attrs.get("axis", 0),
            "keepdim": bool(attrs.get("keepdims", 1)),
        }
    elif layer_type == "Min":
        return {}
    elif layer_type == "Max":
        return {}
    elif layer_type == "Clip":
        attrs = extract_onnx_attrs(node, initializers)
        return {"min": attrs.get("min"), "max": attrs.get("max")}
    elif layer_type == "Conv":
        attrs = extract_onnx_attrs(node, initializers)

        # Get ONNX attributes with defaults
        strides = attrs.get("strides")
        pads = attrs.get("pads")
        dilations = attrs.get("dilations")
        groups = attrs.get("group", 1)

        # Convert ONNX padding format to PyTorch format
        # ONNX: [top, left, bottom, right] for 2D
        # PyTorch: (pad_h, pad_w) - single value for symmetric padding
        pytorch_args = {}

        if strides:
            # Simplify if all values are the same
            if len(strides) > 0 and all(s == strides[0] for s in strides):
                pytorch_args["stride"] = strides[0]
            else:
                pytorch_args["stride"] = tuple(strides)

        if pads:
            # Check if padding is symmetric
            ndims = len(pads) // 2
            symmetric = all(pads[i] == pads[i + ndims] for i in range(ndims))

            if symmetric:
                # Extract first half: [top, left, ...]
                pad_values = pads[:ndims]
                # Simplify if all values are the same
                if all(p == pad_values[0] for p in pad_values):
                    pytorch_args["padding"] = pad_values[0]
                else:
                    pytorch_args["padding"] = tuple(pad_values)
            else:
                raise ValueError(
                    f"Asymmetric padding {pads} not supported for F.conv. "
                    "PyTorch F.conv only supports symmetric padding."
                )

        if dilations:
            # Simplify if all values are the same
            if len(dilations) > 0 and all(d == dilations[0] for d in dilations):
                if dilations[0] != 1:  # Only add if non-default
                    pytorch_args["dilation"] = dilations[0]
            else:
                pytorch_args["dilation"] = tuple(dilations)

        if groups != 1:
            pytorch_args["groups"] = groups

        return pytorch_args
    elif layer_type == "ConvTranspose":
        attrs = extract_onnx_attrs(node, initializers)

        # Get ONNX attributes with defaults
        strides = attrs.get("strides")
        pads = attrs.get("pads")
        dilations = attrs.get("dilations")
        groups = attrs.get("group", 1)
        output_padding = attrs.get("output_padding")

        # Convert ONNX padding format to PyTorch format
        pytorch_args = {}

        if strides:
            # Simplify if all values are the same
            if len(strides) > 0 and all(s == strides[0] for s in strides):
                pytorch_args["stride"] = strides[0]
            else:
                pytorch_args["stride"] = tuple(strides)

        if pads:
            # Check if padding is symmetric
            ndims = len(pads) // 2
            symmetric = all(pads[i] == pads[i + ndims] for i in range(ndims))

            if symmetric:
                # Extract first half: [top, left, ...]
                pad_values = pads[:ndims]
                # Simplify if all values are the same
                if all(p == pad_values[0] for p in pad_values):
                    pytorch_args["padding"] = pad_values[0]
                else:
                    pytorch_args["padding"] = tuple(pad_values)
            else:
                raise ValueError(
                    f"Asymmetric padding {pads} not supported for F.conv_transpose. "
                    "PyTorch F.conv_transpose only supports symmetric padding."
                )

        if output_padding:
            # Simplify if all values are the same
            if len(output_padding) > 0 and all(
                p == output_padding[0] for p in output_padding
            ):
                if output_padding[0] != 0:  # Only add if non-default
                    pytorch_args["output_padding"] = output_padding[0]
            else:
                pytorch_args["output_padding"] = tuple(output_padding)

        if dilations:
            # Simplify if all values are the same
            if len(dilations) > 0 and all(d == dilations[0] for d in dilations):
                if dilations[0] != 1:  # Only add if non-default
                    pytorch_args["dilation"] = dilations[0]
            else:
                pytorch_args["dilation"] = tuple(dilations)

        if groups != 1:
            pytorch_args["groups"] = groups

        return pytorch_args
    else:
        return {}
