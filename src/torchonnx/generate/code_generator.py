"""Main PyTorch code generation orchestrator.

Assembles complete PyTorch module from semantic IR.
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_pytorch_module"]

import inspect

from torch import Tensor

from torchonnx.analyze import ConstantInfo, SemanticModelIR, VariableInfo
from torchonnx.generate._forward_gen import (
    ForwardGenContext,
    generate_forward_method,
    set_forward_gen_context,
)
from torchonnx.generate._init_gen import build_layer_name_mapping, generate_init_method
from torchonnx.generate._runtime_helpers import _standard, _vmap
from torchonnx.generate._state_dict_gen import build_state_dict
from torchonnx.generate._templates import MODULE_TEMPLATE
from torchonnx.generate._utils import sanitize_identifier


def _helper_source(fn) -> str:
    """Return the source text of a runtime-helper function for inlining.

    ``inspect.getsource`` keeps the leading column-0 indent, the docstring,
    and the body verbatim, so the inlined text is exactly what the unit
    suite in ``_runtime_helpers/`` covers. We strip the trailing newline
    so the helper block in the generated module is consistent with the
    prior hand-written triple-quoted templates.
    """
    return inspect.getsource(fn).rstrip()


def generate_pytorch_module(
    semantic_ir: SemanticModelIR,
    module_name: str = "ONNXModel",
    vmap_mode: bool = True,
) -> tuple[str, dict[str, Tensor]]:
    """Generate complete PyTorch module from semantic IR.

    Creates a complete PyTorch nn.Module class with:
    - Imports
    - Class definition
    - __init__ method
    - forward() method
    - state_dict

    :param semantic_ir: Semantic IR from Stage 3/4.
    :param module_name: Name for the generated class.
    :param vmap_mode: If True (default), generate vmap-compatible helper
        functions that avoid ``.item()`` calls and in-place operations,
        enabling compatibility with ``torch.vmap`` and functorch transforms.

    :return: Tuple of (module_code_string, state_dict).
    """
    # Sanitize class name
    class_name = sanitize_identifier(module_name)

    # Build layer name mapping (shared between init, forward, and state_dict)
    layer_name_mapping = build_layer_name_mapping(semantic_ir)

    # Generate imports
    imports = _generate_imports(semantic_ir)

    # Generate forward() method FIRST - this populates the context with helper usage info
    forward_method, forward_context = _generate_forward_with_context(
        semantic_ir, layer_name_mapping, vmap_mode=vmap_mode
    )

    # Generate helper functions based on actual usage (not just op type existence)
    helpers = _generate_helpers_from_context(forward_context, vmap_mode=vmap_mode)

    # Generate __init__ method, emitting only the constants the forward
    # actually references. The IR-level set is the source of truth; the
    # simplify-stage `_remove_unused_buffers` regex-rewriter remains as a
    # safety net for any constants introduced after this point.
    code_to_onnx = {c.code_name: c.onnx_name for c in semantic_ir.constants}
    used_constant_onnx_names = {
        code_to_onnx[code_name]
        for code_name in forward_context.used_constants
        if code_name in code_to_onnx
    }
    init_method = generate_init_method(
        semantic_ir,
        layer_name_mapping,
        used_constant_onnx_names=used_constant_onnx_names,
    )

    # Assemble complete module
    module_code = MODULE_TEMPLATE.format(
        imports=imports,
        helpers=helpers,
        class_name=class_name,
        init_method=init_method,
        forward_method=forward_method,
    )

    # Build state_dict (with layer name mapping for correct keys, all constants included)
    state_dict = build_state_dict(semantic_ir, layer_name_mapping)

    return module_code, state_dict


def _generate_forward_with_context(
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str],
    vmap_mode: bool = True,
) -> tuple[str, ForwardGenContext]:
    """Generate forward method and return the context with helper usage info.

    :param semantic_ir: Semantic IR.
    :param layer_name_mapping: Layer name mapping.
    :param vmap_mode: If True, analyze for vmap-compatible code generation.

    :return: Tuple of (forward_method_code, context)
    """
    # Analyze the IR to determine which helpers are actually needed
    # and compute slice length hints for vmap mode
    helper_context = _get_helper_needs_from_ir(semantic_ir, vmap_mode=vmap_mode)

    # Set context globally before forward generation so handlers can access it
    set_forward_gen_context(helper_context)

    # Generate forward method
    forward_method = generate_forward_method(semantic_ir, layer_name_mapping)

    return forward_method, helper_context


def _check_slice_needs_helper(
    layer, producer_map: dict[str, tuple], vmap_mode: bool, ctx: "ForwardGenContext"
) -> bool:
    """Check if Slice operation needs dynamic_slice helper.

    :param layer: Slice layer.
    :param producer_map: Producer mapping for slice length detection.
    :param vmap_mode: If True, detect static slice lengths.
    :param ctx: Context to update with helper needs.

    :return: True if helper is needed, False otherwise
    """
    if len(layer.inputs) < 3:
        return False

    starts_input = layer.inputs[1]
    ends_input = layer.inputs[2]
    axes_input = layer.inputs[3] if len(layer.inputs) > 3 else None
    steps_input = layer.inputs[4] if len(layer.inputs) > 4 else None

    # All constants -> native slicing, no helper needed
    all_constants = (
        isinstance(starts_input, ConstantInfo)
        and isinstance(ends_input, ConstantInfo)
        and (axes_input is None or isinstance(axes_input, ConstantInfo))
        and (steps_input is None or isinstance(steps_input, ConstantInfo))
    )
    if all_constants:
        return False

    # Check if can use torch.narrow (all params constant, single axis, step=1)
    axes_constant = axes_input is None or isinstance(axes_input, ConstantInfo)
    steps_constant = steps_input is None or isinstance(steps_input, ConstantInfo)
    starts_constant = isinstance(starts_input, ConstantInfo)
    ends_constant = isinstance(ends_input, ConstantInfo)

    if axes_constant and steps_constant and starts_constant and ends_constant:
        axes_list = _extract_axes_list(axes_input)
        steps_list = _extract_steps_list(steps_input, len(axes_list))

        if len(axes_list) == 1 and steps_list[0] == 1:
            assert isinstance(starts_input, ConstantInfo)
            assert isinstance(ends_input, ConstantInfo)
            start_val = int(starts_input.data.item())
            end_val = int(ends_input.data.item())
            if end_val - start_val > 0:
                return False

    # Helper is needed - detect static lengths if vmap_mode
    if vmap_mode:
        static_lengths = _detect_static_slice_lengths(
            layer, starts_input, ends_input, axes_input, steps_input, producer_map
        )
        if static_lengths is not None:
            ctx.slice_length_hints[layer.name] = static_lengths

    return True


def _extract_axes_list(axes_input) -> list:
    """Extract axes list from axes input."""
    if axes_input is None:
        return [0]
    assert isinstance(axes_input, ConstantInfo)
    axes_data = axes_input.data.tolist()
    return axes_data if isinstance(axes_data, list) else [axes_data]


def _extract_steps_list(steps_input, axes_len: int) -> list:
    """Extract steps list from steps input."""
    if steps_input is None:
        return [1] * axes_len
    assert isinstance(steps_input, ConstantInfo)
    steps_data = steps_input.data.tolist()
    return steps_data if isinstance(steps_data, list) else [steps_data]


def _check_expand_needs_helper(layer) -> bool:
    """Check if Expand operation needs helper.

    :param layer: Expand layer.

    :return: True if helper is needed
    """
    if len(layer.inputs) < 2:
        return False

    shape_input = layer.inputs[1]
    output_info = layer.outputs[0]
    output_shape = output_info.shape if output_info else None

    # Known output shape with all integers -> no helper
    if output_shape and all(isinstance(dim, int) for dim in output_shape):
        return False

    # Constant shape with known data shape -> no helper
    if isinstance(shape_input, ConstantInfo):
        data_input = layer.inputs[0]
        if hasattr(data_input, "shape") and data_input.shape:
            data_shape = data_input.shape
            if all(isinstance(d, int) for d in data_shape):
                return False

    return True


def _get_helper_needs_from_ir(
    semantic_ir: SemanticModelIR, vmap_mode: bool = True
) -> ForwardGenContext:
    """Determine helper needs by analyzing the IR.

    This analyzes the IR directly to determine which helpers are actually
    needed after code generation optimizations. In vmap_mode, it also
    analyzes Slice operations to detect static slice lengths.

    :param semantic_ir: Semantic IR.
    :param vmap_mode: If True, detect static slice lengths for vmap compatibility.

    :return: Context with helper needs flags
    """
    from torchonnx.generate._forward_gen import ForwardGenContext

    ctx = ForwardGenContext()
    ctx.vmap_mode = vmap_mode

    # Build producer mapping for vmap slice length detection
    producer_map: dict[str, tuple] = {}
    if vmap_mode:
        for layer in semantic_ir.layers:
            for idx, output in enumerate(layer.outputs):
                producer_map[output.onnx_name] = (layer, idx)

    for layer in semantic_ir.layers:
        if layer.onnx_op_type == "Slice":
            if _check_slice_needs_helper(layer, producer_map, vmap_mode, ctx):
                ctx.needs_dynamic_slice = True
        elif layer.onnx_op_type == "ScatterND":
            ctx.needs_scatter_nd = True
        elif layer.onnx_op_type == "Expand" and _check_expand_needs_helper(layer):
            ctx.needs_dynamic_expand = True

    return ctx


def _detect_static_slice_lengths(
    layer, starts_input, ends_input, axes_input, steps_input, producer_map
) -> list[int] | None:
    """Detect static slice lengths for vmap compatibility.

    Analyzes the computation graph to find patterns where slice length
    is constant even if starts/ends are dynamic. Common pattern:
    ends = starts + constant -> slice_length = constant

    :param layer: Slice layer.
    :param starts_input: Starts tensor info.
    :param ends_input: Ends tensor info.
    :param axes_input: Axes tensor info (may be None or ConstantInfo).
    :param steps_input: Steps tensor info (may be None or ConstantInfo).
    :param producer_map: Maps onnx_name -> (producer_layer, output_index).

    :return: List of static slice lengths, or None if can't determine
    """
    # Get axes list
    if axes_input is None:
        # Default: slice all axes from 0
        num_axes = 1  # Assume single axis if starts is 1D
        if hasattr(starts_input, "shape") and starts_input.shape:
            num_axes = starts_input.shape[0] if len(starts_input.shape) > 0 else 1
        axes_list = list(range(num_axes))
    elif isinstance(axes_input, ConstantInfo):
        axes_data = axes_input.data.tolist()
        axes_list = axes_data if isinstance(axes_data, list) else [axes_data]
    else:
        return None  # Dynamic axes - can't determine static lengths

    # Get steps list
    if steps_input is None:
        steps_list = [1] * len(axes_list)
    elif isinstance(steps_input, ConstantInfo):
        steps_data = steps_input.data.tolist()
        steps_list = steps_data if isinstance(steps_data, list) else [steps_data]
    else:
        return None  # Dynamic steps - can't determine static lengths

    static_lengths = []

    for i in range(len(axes_list)):
        step = steps_list[i] if i < len(steps_list) else 1

        # Try to determine if (ends - starts) is constant for this axis
        slice_len = _get_static_slice_length(starts_input, ends_input, i, step, producer_map)

        if slice_len is None:
            return None  # Can't determine static length for this axis

        static_lengths.append(slice_len)

    return static_lengths


def _extract_value_at_index(data, axis_idx: int):
    """Extract scalar value from data at given axis index.

    :param data: Converted data (list or scalar).
    :param axis_idx: Index into array.

    :return: Scalar value
    """
    if isinstance(data, list) and axis_idx < len(data):
        return data[axis_idx]
    return data if not isinstance(data, list) else data[0]


def _try_constant_case(starts_input, ends_input, axis_idx: int, step: int) -> int | None:
    """Try to determine length when both starts and ends are constants.

    :param starts_input: Starts input (should be ConstantInfo).
    :param ends_input: Ends input (should be ConstantInfo).
    :param axis_idx: Axis index.
    :param step: Step size.

    :return: Slice length or None
    """
    if not (isinstance(starts_input, ConstantInfo) and isinstance(ends_input, ConstantInfo)):
        return None

    starts_data = starts_input.data.tolist()
    ends_data = ends_input.data.tolist()
    start_val = _extract_value_at_index(starts_data, axis_idx)
    end_val = _extract_value_at_index(ends_data, axis_idx)

    length = (end_val - start_val + step - 1) // step
    return max(0, int(length))


def _try_add_pattern_case(
    starts_input, ends_input, axis_idx: int, step: int, producer_map
) -> int | None:
    """Try to detect ends = starts + constant pattern.

    :param starts_input: Starts input.
    :param ends_input: Ends input.
    :param axis_idx: Axis index (unused but kept for symmetry).
    :param step: Step size.
    :param producer_map: Producer mapping.

    :return: Slice length or None
    """
    if not (isinstance(ends_input, VariableInfo) and ends_input.onnx_name in producer_map):
        return None

    ends_producer = _find_producer_through_shape_ops(ends_input.onnx_name, producer_map)
    if ends_producer is None or ends_producer.onnx_op_type != "Add":
        return None

    if len(ends_producer.inputs) != 2:
        return None

    add_input0 = ends_producer.inputs[0]
    add_input1 = ends_producer.inputs[1]

    # Find constant and variable operands
    const_operand = None
    var_operand = None
    if isinstance(add_input0, ConstantInfo):
        const_operand = add_input0
        var_operand = add_input1
    elif isinstance(add_input1, ConstantInfo):
        const_operand = add_input1
        var_operand = add_input0

    if const_operand is None or var_operand is None:
        return None

    # Check if var_operand and starts_input share the same source
    if not (isinstance(var_operand, VariableInfo) and isinstance(starts_input, VariableInfo)):
        return None

    if not _are_from_same_source(var_operand, starts_input, producer_map):
        return None

    # Pattern matched: slice_length = const_val / step
    const_val = int(const_operand.data.item())
    length = (const_val + step - 1) // step
    return max(0, length)


def _get_static_slice_length(
    starts_input, ends_input, axis_idx: int, step: int, producer_map
) -> int | None:
    """Get static slice length for a single axis.

    :param starts_input: Starts tensor info.
    :param ends_input: Ends tensor info.
    :param axis_idx: Index into starts/ends arrays.
    :param step: Step size (assumed constant).
    :param producer_map: Maps onnx_name -> (producer_layer, output_index).

    :return: Static slice length, or None if can't determine
    """
    # Case 1: Both are constants
    length = _try_constant_case(starts_input, ends_input, axis_idx, step)
    if length is not None:
        return length

    # Case 2: ends = starts + constant pattern
    length = _try_add_pattern_case(starts_input, ends_input, axis_idx, step, producer_map)
    if length is not None:
        return length

    return None


def _find_producer_through_shape_ops(onnx_name: str, producer_map, depth: int = 20):
    """Find the actual producer by tracing through shape-preserving operations.

    :param onnx_name: Starting variable onnx_name.
    :param producer_map: Producer mapping.
    :param depth: Max depth to trace.

    :return: Producer layer, or None if can't find
    """
    shape_ops = {"Unsqueeze", "Squeeze", "Reshape", "Flatten"}
    current = onnx_name

    for _ in range(depth):
        if current not in producer_map:
            return None

        producer, _ = producer_map[current]

        # If not a shape-preserving op, return this producer
        if producer.onnx_op_type not in shape_ops:
            return producer

        # Trace through the first input
        if producer.inputs and isinstance(producer.inputs[0], VariableInfo):
            current = producer.inputs[0].onnx_name
        else:
            return None

    return None


def _are_from_same_source(var1, var2, producer_map) -> bool:
    """Check if two variables are derived from the same source tensor.

    Handles patterns like:
    - x4 = some_tensor
    - x12 = x4 + 1
    - x14 = x4.unsqueeze(0)  (starts)
    - x15 = x12.unsqueeze(0) (ends)

    Here x14 and x15 are from the same source (x4) with a constant offset.
    """
    if not isinstance(var1, VariableInfo) or not isinstance(var2, VariableInfo):
        return False

    # Get the source of each by tracing through Unsqueeze/Reshape
    source1 = _trace_to_source(var1, producer_map)
    source2 = _trace_to_source(var2, producer_map)

    # If both trace to the same source, they're related
    return bool(source1 and source2 and source1 == source2)


def _trace_to_source(var, producer_map, depth: int = 20) -> str | None:
    """Trace a variable back to its source, skipping shape-preserving ops.

    :param var: Variable to trace.
    :param producer_map: Producer mapping.
    :param depth: Max depth to trace.

    :return: Source onnx_name, or None if can't trace
    """
    if not isinstance(var, VariableInfo):
        return None

    current = var.onnx_name
    shape_ops = {"Unsqueeze", "Squeeze", "Reshape", "Flatten"}

    for _ in range(depth):
        if current not in producer_map:
            return current

        producer, _ = producer_map[current]

        # Stop at non-shape-preserving ops
        if producer.onnx_op_type not in shape_ops:
            # For Add, trace through the non-constant input
            if producer.onnx_op_type == "Add" and len(producer.inputs) == 2:
                for inp in producer.inputs:
                    if not isinstance(inp, ConstantInfo) and isinstance(inp, VariableInfo):
                        current = inp.onnx_name
                        break
                else:
                    return current
            else:
                return current
        else:
            # Shape-preserving op: trace through first input
            if producer.inputs and isinstance(producer.inputs[0], VariableInfo):
                current = producer.inputs[0].onnx_name
            else:
                return current

    return current


def _generate_imports(semantic_ir: SemanticModelIR) -> str:
    """Generate import statements based on operations used.

    :param semantic_ir: Semantic IR.

    :return: Import statements string
    """
    imports = [
        "import torch",
        "import torch.nn as nn",
    ]

    # Check if F.* operations are used
    needs_functional = any(layer.pytorch_type.startswith("F.") for layer in semantic_ir.layers)

    if needs_functional:
        imports.append("import torch.nn.functional as F")

    return "\n".join(imports)


def _generate_helpers_from_context(ctx: ForwardGenContext, vmap_mode: bool = True) -> str:
    """Generate helper functions based on actual usage tracked in context.

    Only emits helpers that are actually called in the generated code,
    not just based on ONNX op type existence.

    :param ctx: Forward generation context with helper usage flags.
    :param vmap_mode: If True, use vmap-compatible helpers that avoid .item().

        and in-place operations.
    :return: Helper function definitions string
    """
    helpers = []

    if ctx.needs_dynamic_slice:
        helpers.append(DYNAMIC_SLICE_VMAP_HELPER if vmap_mode else DYNAMIC_SLICE_HELPER)

    if ctx.needs_scatter_nd:
        helpers.append(SCATTER_ND_VMAP_HELPER if vmap_mode else SCATTER_ND_HELPER)

    if ctx.needs_dynamic_expand:
        helpers.append(EXPAND_VMAP_HELPER if vmap_mode else EXPAND_HELPER)

    return "\n\n".join(helpers)


# Helper templates -- extracted at import time from the linted, unit-tested
# real functions in _runtime_helpers/. The standard variants are emitted when
# vmap_mode=False, the VMAP variants when vmap_mode=True (the default).
DYNAMIC_SLICE_HELPER = _helper_source(_standard.dynamic_slice)


SCATTER_ND_HELPER = _helper_source(_standard.scatter_nd)
EXPAND_HELPER = _helper_source(_standard.dynamic_expand)

DYNAMIC_SLICE_VMAP_HELPER = _helper_source(_vmap.dynamic_slice)
SCATTER_ND_VMAP_HELPER = _helper_source(_vmap.scatter_nd)
EXPAND_VMAP_HELPER = _helper_source(_vmap.dynamic_expand)
