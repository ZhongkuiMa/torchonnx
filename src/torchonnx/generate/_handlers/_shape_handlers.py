"""Shape-oriented OPERATION handlers (reshape, transpose, squeeze, unsqueeze).

Extracted from the ``_operations.py`` monolith. The carve-out keeps the
rank-manipulation handlers together because they share a tight dependency
on the reshape inference helpers in ``_operations_utils`` and they evolve
together when ONNX opset changes shift the input vs attribute layout.

``_operations.py`` re-imports these handler names so existing call sites
and the test fixtures in ``test_handlers_direct.py`` continue to work
unchanged.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "_handle_reshape",
    "_handle_squeeze",
    "_handle_transpose",
    "_handle_unsqueeze",
]

from torchonnx.analyze import ConstantInfo, SemanticLayerIR
from torchonnx.generate._handlers._operations_utils import (
    _can_infer_reshape_statically,
    _compute_inferred_dim,
    _get_input_code_name_selective,
    _get_input_code_names,
    _require_min_inputs,
)
from torchonnx.generate._utils import format_argument


def _handle_reshape(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle reshape operation.

    Detects common patterns:

    - Flatten pattern (batch, -1): generates ``x.flatten(1)`` for
      batch-aware flattening;
    - Other reshapes: generates ``x.reshape(shape)`` with the batch
      dimension preserved when the input also has a leading 1 (so a
      ``[1, k, k]`` inner tensor is NOT mistaken for a batched tensor).

    :param layer: Semantic layer IR.
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name.

    :return: Generated code line.
    """
    output = layer.outputs[0].code_name

    _require_min_inputs(layer, 2, "Reshape")

    data = _get_input_code_name_selective(layer.inputs[0])
    shape_input = layer.inputs[1]

    # If shape is a constant, use Python literal directly
    if isinstance(shape_input, ConstantInfo):
        shape_data = shape_input.data
        shape_list: list[int] = shape_data.tolist()  # pyright: ignore[reportAssignmentType]
        if not isinstance(shape_list, list):
            shape_list = [shape_list]

        # Detect flatten pattern: reshape(batch_size, -1) -> flatten(1)
        if len(shape_list) == 2 and shape_list[1] == -1:
            return f"{output} = {data}.flatten(1)"

        # Batch-aware reshaping. The "shape starts with 1" rewrite assumes the
        # leading 1 is the batch dimension; replacing it with -1 enables
        # dynamic batching downstream. But a `[1, k, k]` shape inside a
        # transformer block is just a rank-fixing constant, not a batch hint.
        # We only apply the rewrite when the input tensor also has a leading
        # dim of 1 -- meaning the layer is consuming something that already
        # looks batched (typically the model input or a tensor in the batch-
        # flow path). Inner tensors that happen to produce a leading-1 shape
        # without a leading-1 source keep their literal shape.
        input_info = layer.inputs[0]
        input_shape = input_info.shape if hasattr(input_info, "shape") else None
        input_has_batch_leading_one = bool(
            input_shape and len(input_shape) >= 1 and input_shape[0] == 1
        )
        if len(shape_list) >= 1 and shape_list[0] == 1 and input_has_batch_leading_one:
            if -1 in shape_list:
                if _can_infer_reshape_statically(input_shape, shape_list):
                    assert input_shape is not None
                    inferred_dim = _compute_inferred_dim(input_shape, shape_list)
                    if inferred_dim is not None:
                        minus_one_idx = shape_list.index(-1)
                        shape_list[minus_one_idx] = inferred_dim
                        shape_list[0] = -1
            else:
                shape_list[0] = -1

        shape_literal = tuple(shape_list)
        return f"{output} = {data}.reshape{shape_literal}"

    # Dynamic shape - use tolist() at runtime
    shape_code = _get_input_code_name_selective(shape_input)
    return f"{output} = {data}.reshape([int(x) for x in {shape_code}.tolist()])"


def _handle_transpose(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle transpose / permute operation.

    Generates ``output = input.permute(dims)`` when an explicit ``perm``
    attribute is present, otherwise the default ``transpose(-2, -1)``.

    :param layer: Semantic layer IR.
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name.

    :return: Generated code line.
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    # Get perm argument
    perm_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "perm"), None)

    if perm_arg:
        perm = format_argument(perm_arg.value)
        return f"{output} = {inputs[0]}.permute({perm})"
    # Default transpose (swap last two dims)
    return f"{output} = {inputs[0]}.transpose(-2, -1)"


def _handle_squeeze(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle squeeze operation.

    :param layer: Semantic layer IR.
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name.

    :return: Generated code line.
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    # Get dim argument
    dim_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "dim"), None)

    if dim_arg:
        dim = format_argument(dim_arg.value)
        return f"{output} = {inputs[0]}.squeeze({dim})"
    return f"{output} = {inputs[0]}.squeeze()"


def _handle_unsqueeze(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle unsqueeze operation.

    In ONNX opset 13+, ``axes`` is the second input tensor.
    In older opsets, ``axes`` is an attribute.

    :param layer: Semantic layer IR.
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name.

    :return: Generated code line.
    """
    output = layer.outputs[0].code_name

    # Get dim argument (from attributes for older opsets)
    dim_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "dim"), None)

    # Process data input
    data = _get_input_code_name_selective(layer.inputs[0])

    if dim_arg and dim_arg.value is not None:
        dim = format_argument(dim_arg.value)
        return f"{output} = {data}.unsqueeze({dim})"
    if len(layer.inputs) >= 2:
        # ONNX opset 13+: axes is the second input
        axes_input = layer.inputs[1]
        if isinstance(axes_input, ConstantInfo):
            # Use literal (don't mark as used)
            axes_value = int(axes_input.data.item())
            return f"{output} = {data}.unsqueeze({axes_value})"
        # Dynamic axes (mark as used)
        axes_code = _get_input_code_name_selective(axes_input)
        return f"{output} = {data}.unsqueeze({axes_code}.item())"
    # Fallback: unsqueeze at dim 0
    return f"{output} = {data}.unsqueeze(0)"
