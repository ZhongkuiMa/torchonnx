__docformat__ = "restructuredtext"
__all__ = ["parse_conv"]

import onnx

from torchonnx.onnx_parser import *


def parse_conv(node: onnx.NodeProto) -> str:
    input_names = parse_node_inputs(node)
    output_names = parse_node_outputs(node)
    code = f"        {output_names[0]} = self.{node.name}({input_names[0]})\n"
    return code
