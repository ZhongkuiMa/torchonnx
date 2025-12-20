"""Stage 2: Intermediate Representation (IR) Type Definitions.

Defines NodeIR and ModelIR dataclasses for pure structural IR.
Stage 2 extracts ONNX graph structure only - no semantic interpretation.
"""

__docformat__ = "restructuredtext"
__all__ = ["NodeIR", "ModelIR"]

from dataclasses import dataclass
from typing import Any

from onnx import ModelProto, NodeProto, TensorProto


@dataclass(frozen=True)
class NodeIR:
    """Pure structural IR for a single ONNX node.

    Stage 2 captures ONNX graph topology only, without semantic interpretation.
    All semantic analysis (PyTorch types, parameter classification, etc.)
    happens in Stage 3.

    :param name: Node name (from ONNX node.name or generated)
    :param onnx_op_type: ONNX operator type (e.g., "Conv", "Add", "BatchNormalization")
    :param raw_attributes: Raw ONNX attributes (unparsed dict of AttributeProto)
    :param input_names: ALL input tensor names (no filtering/classification)
    :param output_names: ALL output tensor names
    :param input_shapes: Input tensor shapes
    :param output_shapes: Output tensor shapes
    :param node: Original ONNX NodeProto
    """

    name: str
    onnx_op_type: str
    raw_attributes: dict[str, Any]
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, tuple[int, ...] | None]
    output_shapes: dict[str, tuple[int, ...] | None]
    node: NodeProto


@dataclass(frozen=True)
class ModelIR:
    """Pure structural IR for complete ONNX model.

    Stage 2 captures ONNX model structure only - graph topology, tensor shapes,
    and initializers. No semantic classification happens here.

    :param layers: List of node IRs (structural only)
    :param input_names: Model input tensor names
    :param output_names: Model output tensor names
    :param shapes: All tensor shapes in the model
    :param initializers: All ONNX initializers (unclassified)
    :param model: Original ONNX ModelProto
    """

    layers: list[NodeIR]
    input_names: list[str]
    output_names: list[str]
    shapes: dict[str, tuple[int | str, ...] | None]
    initializers: dict[str, TensorProto]
    model: ModelProto
