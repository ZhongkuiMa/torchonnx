__docformat__ = "restructuredtext"
__all__ = ["parse_node_inputs", "parse_node_outputs"]

import onnx

def parse_node_inputs(node: onnx.NodeProto) -> list[str]:
    input_names = []
    for input in node.input:
        input_names.append(input)

    return input_names


def parse_node_outputs(node: onnx.NodeProto) -> list[str]:
    output_names = []
    for output in node.output:
        output_names.append(output)

    return output_names