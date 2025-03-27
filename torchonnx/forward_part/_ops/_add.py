__docformat__ = "restructuredtext"
__all__ = ["parse_add"]

import onnx

from torchonnx.onnx_parser import *


def parse_add(node: onnx.NodeProto) -> str:
    input_names = parse_node_inputs(node)
    output_names = parse_node_outputs(node)
    code = f"        {output_names[0]} = {input_names[0]} + {input_names[1]}\n"
    return code
