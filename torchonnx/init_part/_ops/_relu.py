__docformat__ = "restructuredtext"
__all__ = ["parse_relu"]

import onnx


def parse_relu(
        node: onnx.NodeProto, initializer_shapes: dict[str, tuple[int, ...]]
) -> str:
    s = f"        self.{node.name} = nn.ReLU()\n"
    return s
