__docformat__ = "restructuredtext"
__all__ = ["parse_conv"]

from typing import Any

import onnx

from torchonnx.onnx_parser import *


def _to_torch_args(attrs: dict[str, Any]) -> dict[str, Any]:
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
    for k, v in attrs.items():
        if k == "kernel_shape":
            torch_args["kernel_size"] = v
        elif k == "pads":
            if len(v) == 4:
                if v[0] != v[2] or v[1] != v[3]:
                    raise ValueError(f"Unsupported padding: {v}")
                torch_args["padding"] = v[:2]
            else:
                raise NotImplementedError
        elif k == "dilations":
            torch_args["dilation"] = v
        elif k == "strides":
            torch_args["stride"] = v
        elif k == "group":
            torch_args["groups"] = v
        elif k == "auto_pad":
            assert v == "NOTSET", f"Unsupported auto_pad value: {v}"

    return torch_args


def parse_conv(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    inputs = parse_node_inputs(node)
    attrs = parse_node_attributes(node)

    torch_args = _to_torch_args(attrs)
    weight_shape = initializer_shapes[inputs[1]]
    torch_args["in_channels"] = weight_shape[1]
    torch_args["out_channels"] = weight_shape[0]
    torch_args["bias"] = bool(len(inputs) == 3)

    dim = len(torch_args["kernel_size"])

    if dim == 2:
        code = (
            f"        self.{node.name} = "
            f"nn.Conv2d("
            f'{torch_args["in_channels"]}, '  # in_channels
            f'{torch_args["out_channels"]}, '  # out_channels
            f'{torch_args["kernel_size"]}, '  # kernel_size
            f'stride={torch_args["stride"]}, '
            f'padding={torch_args["padding"]}, '
            f'dilation={torch_args["dilation"]}, '
            f'groups={torch_args["groups"]}'
            ")\n"
        )
    else:
        raise NotImplementedError(f"Unsupported dimension: {dim}")

    return code
