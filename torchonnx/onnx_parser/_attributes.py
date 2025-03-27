__docformat__ = "restructuredtext"
__all__ = ["parse_node_inputs", "parse_node_outputs", "parse_node_attributes"]

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


def _parse_float_attr(attribute: onnx.AttributeProto) -> float:
    return attribute.f


def _parse_int_attr(attribute: onnx.AttributeProto) -> int:
    return attribute.i


def _parse_string_attr(attribute: onnx.AttributeProto) -> str:
    return attribute.s.decode("utf-8")


def _parse_ints_attr(attribute: onnx.AttributeProto) -> tuple[int, ...]:
    return tuple(int(v) for v in attribute.ints)


ATTR2VALUE_MAP = {
    1: _parse_float_attr,  # "FLOAT"
    2: _parse_int_attr,  # "INT"
    3: _parse_string_attr,  # "STRING"
    7: _parse_ints_attr,  # "INTS"
}


def parse_node_attributes(node: onnx.NodeProto) -> dict[str, any]:
    attributes = {}

    for attribute in node.attribute:
        value = ATTR2VALUE_MAP[attribute.type](attribute)
        attributes[attribute.name] = value

    return attributes
