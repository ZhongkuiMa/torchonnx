"""Stage 2: IR Builder.

Builds pure structural intermediate representation (IR) from normalized ONNX models.
Stage 2 extracts ONNX graph topology only - no semantic interpretation.
"""

__docformat__ = "restructuredtext"
__all__ = ["build_model_ir"]

from onnx import ModelProto, NodeProto

from ..normalize import (
    extract_onnx_opset_version,
    get_onnx_initializers,
    get_onnx_model_input_names,
    get_onnx_model_output_names,
    get_onnx_model_shapes,
    get_onnx_nodes,
)
from .types import ModelIR, NodeIR


def _clean_shape(shape: tuple[int | str, ...] | None) -> tuple[int, ...] | None:
    """Clean shape by filtering out unknown/symbolic dimensions.

    Batch dimension (first dim) can be symbolic, but if any OTHER dimension
    is symbolic (e.g., 'unk__35'), return None to indicate unknown shape.

    :param shape: Shape tuple that may contain strings
    :return: Shape with only integers, or None if any non-batch dimension is unknown
    """
    if shape is None:
        return None
    # If any NON-BATCH dimension (dims after first) is a string, the shape is unknown
    if len(shape) > 1 and any(isinstance(dim, str) for dim in shape[1:]):
        return None
    # All non-batch dimensions are integers (batch dim can be symbolic)
    # Convert any remaining strings to ints (this handles the batch dim case)
    return tuple(1 if isinstance(dim, str) else dim for dim in shape)


def _build_node_ir(
    node: NodeProto,
    shapes: dict[str, tuple[int | str, ...] | None],
    node_counter: int,
) -> NodeIR:
    """Build pure structural NodeIR for a single ONNX node.

    No semantic interpretation - just captures ONNX structure.

    :param node: ONNX node
    :param shapes: Tensor shape information
    :param node_counter: Current node index (for name generation)
    :return: NodeIR representation
    """
    # Extract shapes for all inputs/outputs, clean them to remove symbolic dims
    input_shapes = {name: _clean_shape(shapes.get(name)) for name in node.input}
    output_shapes = {name: _clean_shape(shapes.get(name)) for name in node.output}

    # Generate node name from ONNX node name, or use first output name as fallback
    name = (
        node.name
        if node.name
        else (node.output[0] if node.output else f"node_{node_counter}")
    )

    # Store raw ONNX attributes (unparsed)
    raw_attributes = {attr.name: attr for attr in node.attribute}

    return NodeIR(
        name=name,
        onnx_op_type=node.op_type,  # Just ONNX type, no PyTorch conversion
        raw_attributes=raw_attributes,
        input_names=list(node.input),  # ALL inputs, no filtering
        output_names=list(node.output),  # ALL outputs
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        node=node,
    )


def build_model_ir(model: ModelProto) -> ModelIR:
    """Build pure structural IR from ONNX model.

    Stage 2 extracts only structural information:
    - ONNX operator types (not PyTorch types)
    - Graph topology (connections between nodes)
    - Tensor shapes
    - Raw attributes (unparsed)

    No semantic analysis happens here. All semantic interpretation
    (PyTorch type mapping, parameter classification, argument extraction)
    is deferred to Stage 3.

    :param model: ONNX model
    :return: ModelIR representation (structural only)
    """
    extract_onnx_opset_version(model)  # For checking supported opset versions

    initializers = get_onnx_initializers(model)
    nodes = get_onnx_nodes(model)
    # Get shapes - batch dim can be 1, but preserve other symbolic dims as strings
    shapes = get_onnx_model_shapes(model, set_batch_to_one=False)

    # Build structural IR for each node
    layers = [_build_node_ir(node, shapes, idx) for idx, node in enumerate(nodes)]

    input_names = get_onnx_model_input_names(model)
    output_names = get_onnx_model_output_names(model)

    return ModelIR(
        layers=layers,
        input_names=input_names,
        output_names=output_names,
        shapes=shapes,
        initializers=initializers,  # Unclassified - Stage 3 will classify them
        model=model,
    )
