__docformat__ = "restructuredtext"
__all__ = ["parse_flatten"]

from typing import Any

import onnx

from torchonnx.onnx_parser import *


def _to_torch_args(attrs: dict[str, Any]) -> dict[str, Any]:
    torch_args = {"start_dim": 1}
    for k, v in attrs.items():
        if k == "axis":
            assert v == 1
            torch_args["start_dim"] = v
        else:
            raise ValueError(f"Invalid attribute: {k}")

    return torch_args


def parse_flatten(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    attrs = parse_node_attributes(node)
    torch_args = _to_torch_args(attrs)
    s = f"        self.{node.name} = nn.Flatten(start_dim={torch_args['start_dim']})\n"

    return s
