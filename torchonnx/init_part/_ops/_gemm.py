__docformat__ = "restructuredtext"
__all__ = ["parse_gemm"]

from typing import Any

import onnx

from torchonnx.onnx_parser import *


def _to_torch_args(attrs: dict[str, Any]) -> dict[str, Any]:
    torch_args = {}
    for k, v in attrs.items():
        if k == "transA":
            torch_args["transA"] = v
        elif k == "transB":
            torch_args["transB"] = v
        elif k == "alpha":
            torch_args["alpha"] = v
        elif k == "beta":
            torch_args["beta"] = v
        else:
            raise ValueError(f"Invalid attribute: {k}")

    return torch_args


def parse_gemm(
    node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    input_names = parse_node_inputs(node)
    attrs = parse_node_attributes(node)
    torch_args = _to_torch_args(attrs)

    weight_shape = initializer_shapes[input_names[1]]
    if attrs["transB"] == 1:
        torch_args["input_features"] = weight_shape[1]
        torch_args["output_features"] = weight_shape[0]
    else:
        torch_args["input_features"] = weight_shape[0]
        torch_args["output_features"] = weight_shape[1]

    torch_args["bias"] = bool(len(input_names) == 3)

    code = (
        f"        self.{node.name} = nn.Linear("
        f"{torch_args['input_features']}, "
        f"{torch_args['output_features']}, "
        f"bias={torch_args['bias']}"
        f")\n"
    )

    return code
