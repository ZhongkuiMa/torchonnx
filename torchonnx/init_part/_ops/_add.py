__docformat__ = "restructuredtext"
__all__ = ["parse_add"]

import onnx


def parse_add(
        node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    return ""
