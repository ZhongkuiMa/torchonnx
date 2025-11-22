"""Functional operation mappings for ONNX to PyTorch.

This module defines operations that should be implemented as functional calls
rather than as nn.Module layers.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "is_functional_operation_with_args",
    "get_functional_operation_template",
    "extract_functional_args",
]

from typing import Any

from onnx import NodeProto, TensorProto


FUNCTIONAL_OPERATIONS_WITH_ARGS: dict[str, str] = {
    "Pad": "F.pad",
    "Flatten": "torch.flatten",
    "Reshape": "reshape",
    "Transpose": "permute",
    "Squeeze": "squeeze",
    "Unsqueeze": "unsqueeze",
    "Shape": "shape",
    "Gather": "gather",
    "Cast": "cast",
    "Concat": "torch.cat",
    "Gemm": "gemm",
    "Conv": "F.conv2d",
    "ConvTranspose": "F.conv_transpose2d",
    "Slice": "slice",
    "Sign": "sign",
    "Split": "split",
    "ConstantOfShape": "constant_of_shape",
    "ReduceMean": "reduce_mean",
    "ReduceSum": "reduce_sum",
    "Cos": "cos",
    "Sin": "sin",
    "Floor": "floor",
    "Neg": "neg",
    "Expand": "expand",
    "Range": "range",
    "Where": "where",
    "ScatterND": "scatter_nd",
    "ArgMax": "torch.argmax",
    "Min": "torch.min",
    "Max": "torch.max",
    "Clip": "torch.clamp",
    "Upsample": "F.interpolate",
    "Resize": "F.interpolate",
}


def is_functional_operation_with_args(layer_type: str) -> bool:
    """Check if operation should be implemented as functional with arguments.

    :param layer_type: PyTorch layer type
    :return: True if operation is functional with arguments
    """
    return layer_type in FUNCTIONAL_OPERATIONS_WITH_ARGS


def get_functional_operation_template(layer_type: str) -> str:
    """Get the template for generating functional operation code.

    :param layer_type: PyTorch layer type
    :return: Template string for code generation
    """
    if layer_type not in FUNCTIONAL_OPERATIONS_WITH_ARGS:
        raise ValueError(f"Not a functional operation with args: {layer_type}")
    return FUNCTIONAL_OPERATIONS_WITH_ARGS[layer_type]


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
        elif attr.type == 4:
            attrs[attr.name] = attr.t
        elif attr.type == 6:
            attrs[attr.name] = list(attr.floats)
        elif attr.type == 7:
            attrs[attr.name] = list(attr.ints)

    return attrs


def extract_pad_args(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Extract arguments for Pad operation.

    :param node: ONNX Pad node
    :param initializers: All ONNX initializers
    :return: Arguments for F.pad call
    """
    attrs = _extract_onnx_attributes(node)

    mode = attrs.get("mode", "constant")
    if mode == b"constant":
        mode = "constant"
    elif isinstance(mode, bytes):
        mode = mode.decode("utf-8")

    value = attrs.get("value", 0.0)

    pads = attrs.get("pads", [])
    if not pads and len(node.input) >= 2 and node.input[1] in initializers:
        from onnx import numpy_helper

        pads_array = numpy_helper.to_array(initializers[node.input[1]])
        pads = pads_array.tolist()

    return {
        "pad": pads,
        "mode": mode,
        "value": value,
    }


def extract_flatten_args(
    node: NodeProto, initializers: dict[str, TensorProto]
) -> dict[str, Any]:
    """Extract arguments for Flatten operation.

    :param node: ONNX Flatten node
    :param initializers: All ONNX initializers
    :return: Arguments for torch.flatten call
    """
    attrs = _extract_onnx_attributes(node)
    axis = attrs.get("axis", 1)

    return {
        "start_dim": axis,
    }


def extract_functional_args(
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
        return extract_pad_args(node, initializers)
    elif layer_type == "Flatten":
        return extract_flatten_args(node, initializers)
    elif layer_type == "Reshape":
        return {}
    elif layer_type == "Transpose":
        attrs = _extract_onnx_attributes(node)
        return {"perm": attrs.get("perm", [])}
    elif layer_type == "Squeeze":
        attrs = _extract_onnx_attributes(node)
        axes = attrs.get("axes", [])
        return {"dim": axes[0] if len(axes) == 1 else tuple(axes)} if axes else {}
    elif layer_type == "Unsqueeze":
        attrs = _extract_onnx_attributes(node)
        axes = attrs.get("axes", [])
        return {"dim": axes[0] if len(axes) == 1 else tuple(axes)} if axes else {}
    elif layer_type == "Shape":
        return {}
    elif layer_type == "Gather":
        attrs = _extract_onnx_attributes(node)
        return {"axis": attrs.get("axis", 0)}
    elif layer_type == "Cast":
        attrs = _extract_onnx_attributes(node)
        return {"to": attrs.get("to", 1)}
    elif layer_type == "Concat":
        attrs = _extract_onnx_attributes(node)
        return {"axis": attrs.get("axis", 0)}
    elif layer_type == "Gemm":
        attrs = _extract_onnx_attributes(node)
        return {
            "alpha": attrs.get("alpha", 1.0),
            "beta": attrs.get("beta", 1.0),
            "transA": attrs.get("transA", 0),
            "transB": attrs.get("transB", 0),
        }
    elif layer_type == "Conv":
        attrs = _extract_onnx_attributes(node)
        return {
            "stride": attrs.get("strides", [1, 1]),
            "padding": attrs.get("pads", [0, 0, 0, 0]),
            "dilation": attrs.get("dilations", [1, 1]),
            "groups": attrs.get("group", 1),
        }
    elif layer_type == "ConvTranspose":
        attrs = _extract_onnx_attributes(node)
        return {
            "stride": attrs.get("strides", [1, 1]),
            "padding": attrs.get("pads", [0, 0, 0, 0]),
            "dilation": attrs.get("dilations", [1, 1]),
            "groups": attrs.get("group", 1),
            "output_padding": attrs.get("output_padding", [0, 0]),
        }
    elif layer_type == "Slice":
        attrs = _extract_onnx_attributes(node)
        return {
            "starts": attrs.get("starts", []),
            "ends": attrs.get("ends", []),
            "axes": attrs.get("axes", None),
        }
    elif layer_type == "Sign":
        return {}
    elif layer_type == "Split":
        attrs = _extract_onnx_attributes(node)
        return {"axis": attrs.get("axis", 0)}
    elif layer_type == "ConstantOfShape":
        attrs = _extract_onnx_attributes(node)
        value = attrs.get("value")
        if value is not None:
            from onnx import numpy_helper

            value_array = numpy_helper.to_array(value)
            return {"value": float(value_array.flatten()[0])}
        return {"value": 0.0}
    elif layer_type == "ReduceMean":
        attrs = _extract_onnx_attributes(node)
        axes = attrs.get("axes", None)
        keepdims = attrs.get("keepdims", 1)
        return {"axes": axes, "keepdims": bool(keepdims)}
    elif layer_type == "ReduceSum":
        attrs = _extract_onnx_attributes(node)
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
        attrs = _extract_onnx_attributes(node)
        return {"reduction": attrs.get("reduction", "none")}
    elif layer_type == "ArgMax":
        attrs = _extract_onnx_attributes(node)
        return {
            "axis": attrs.get("axis", 0),
            "keepdims": bool(attrs.get("keepdims", 1)),
        }
    elif layer_type == "Min":
        return {}
    elif layer_type == "Max":
        return {}
    elif layer_type == "Clip":
        attrs = _extract_onnx_attributes(node)
        return {"min": attrs.get("min"), "max": attrs.get("max")}
    else:
        return {}
