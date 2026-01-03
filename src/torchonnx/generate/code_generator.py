"""Main PyTorch code generation orchestrator.

Assembles complete PyTorch module from semantic IR.
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_pytorch_module"]

import torch

from torchonnx.analyze import SemanticModelIR
from torchonnx.generate._forward_gen import (
    ForwardGenContext,
    generate_forward_method,
    set_forward_gen_context,
)
from torchonnx.generate._init_gen import build_layer_name_mapping, generate_init_method
from torchonnx.generate._state_dict_gen import build_state_dict
from torchonnx.generate._templates import MODULE_TEMPLATE
from torchonnx.generate._utils import sanitize_identifier


def generate_pytorch_module(
    semantic_ir: SemanticModelIR,
    module_name: str = "ONNXModel",
    vmap_mode: bool = True,
) -> tuple[str, dict[str, torch.Tensor]]:
    """Generate complete PyTorch module from semantic IR.

    Creates a complete PyTorch nn.Module class with:
    - Imports
    - Class definition
    - __init__ method
    - forward() method
    - state_dict

    :param semantic_ir: Semantic IR from Stage 3/4
    :param module_name: Name for the generated class
    :param vmap_mode: If True, generate vmap-compatible helper functions that
        avoid .item() calls and in-place operations. Default False preserves
        the standard helpers for backward compatibility.
    :return: Tuple of (module_code_string, state_dict)
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

    # Generate __init__ method with all constants (Stage 6 will remove unused ones)
    init_method = generate_init_method(semantic_ir, layer_name_mapping)

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

    :param semantic_ir: Semantic IR
    :param layer_name_mapping: Layer name mapping
    :param vmap_mode: If True, analyze for vmap-compatible code generation
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

    :param layer: Slice layer
    :param producer_map: Producer mapping for slice length detection
    :param vmap_mode: If True, detect static slice lengths
    :param ctx: Context to update with helper needs
    :return: True if helper is needed, False otherwise
    """
    from torchonnx.analyze import ConstantInfo

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
    from torchonnx.analyze import ConstantInfo

    assert isinstance(axes_input, ConstantInfo)
    axes_data = axes_input.data.tolist()
    return axes_data if isinstance(axes_data, list) else [axes_data]


def _extract_steps_list(steps_input, axes_len: int) -> list:
    """Extract steps list from steps input."""
    if steps_input is None:
        return [1] * axes_len
    from torchonnx.analyze import ConstantInfo

    assert isinstance(steps_input, ConstantInfo)
    steps_data = steps_input.data.tolist()
    return steps_data if isinstance(steps_data, list) else [steps_data]


def _check_expand_needs_helper(layer) -> bool:
    """Check if Expand operation needs helper.

    :param layer: Expand layer
    :return: True if helper is needed
    """
    from torchonnx.analyze import ConstantInfo

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

    :param semantic_ir: Semantic IR
    :param vmap_mode: If True, detect static slice lengths for vmap compatibility
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
    ends = starts + constant → slice_length = constant

    :param layer: Slice layer
    :param starts_input: Starts tensor info
    :param ends_input: Ends tensor info
    :param axes_input: Axes tensor info (may be None or ConstantInfo)
    :param steps_input: Steps tensor info (may be None or ConstantInfo)
    :param producer_map: Maps onnx_name -> (producer_layer, output_index)
    :return: List of static slice lengths, or None if can't determine
    """
    from torchonnx.analyze import ConstantInfo

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

    :param data: Converted data (list or scalar)
    :param axis_idx: Index into array
    :return: Scalar value
    """
    if isinstance(data, list) and axis_idx < len(data):
        return data[axis_idx]
    return data if not isinstance(data, list) else data[0]


def _try_constant_case(starts_input, ends_input, axis_idx: int, step: int) -> int | None:
    """Try to determine length when both starts and ends are constants.

    :param starts_input: Starts input (should be ConstantInfo)
    :param ends_input: Ends input (should be ConstantInfo)
    :param axis_idx: Axis index
    :param step: Step size
    :return: Slice length or None
    """
    from torchonnx.analyze import ConstantInfo

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

    :param starts_input: Starts input
    :param ends_input: Ends input
    :param axis_idx: Axis index (unused but kept for symmetry)
    :param step: Step size
    :param producer_map: Producer mapping
    :return: Slice length or None
    """
    from torchonnx.analyze import ConstantInfo, VariableInfo

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

    :param starts_input: Starts tensor info
    :param ends_input: Ends tensor info
    :param axis_idx: Index into starts/ends arrays
    :param step: Step size (assumed constant)
    :param producer_map: Maps onnx_name -> (producer_layer, output_index)
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


def _find_producer_through_shape_ops(onnx_name: str, producer_map, depth: int = 5):
    """Find the actual producer by tracing through shape-preserving operations.

    :param onnx_name: Starting variable onnx_name
    :param producer_map: Producer mapping
    :param depth: Max depth to trace
    :return: Producer layer, or None if can't find
    """
    from torchonnx.analyze import VariableInfo

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
    from torchonnx.analyze import VariableInfo

    if not isinstance(var1, VariableInfo) or not isinstance(var2, VariableInfo):
        return False

    # Get the source of each by tracing through Unsqueeze/Reshape
    source1 = _trace_to_source(var1, producer_map)
    source2 = _trace_to_source(var2, producer_map)

    # If both trace to the same source, they're related
    return bool(source1 and source2 and source1 == source2)


def _trace_to_source(var, producer_map, depth: int = 5) -> str | None:
    """Trace a variable back to its source, skipping shape-preserving ops.

    :param var: Variable to trace
    :param producer_map: Producer mapping
    :param depth: Max depth to trace
    :return: Source onnx_name, or None if can't trace
    """
    from torchonnx.analyze import VariableInfo

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
                from torchonnx.analyze import ConstantInfo

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

    :param semantic_ir: Semantic IR
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

    :param ctx: Forward generation context with helper usage flags
    :param vmap_mode: If True, use vmap-compatible helpers that avoid .item()
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


# Helper function templates
DYNAMIC_SLICE_HELPER = '''\
def dynamic_slice(data, starts, ends, axes=None, steps=None):
    """Dynamic slice helper for ONNX Slice operation."""
    # Ensure tensor
    starts = torch.as_tensor(starts, device=data.device)
    ends   = torch.as_tensor(ends,   device=data.device)
    if axes is None:
        axes = torch.arange(starts.numel(), device=data.device)
    else:
        axes = torch.as_tensor(axes, device=data.device)
    if steps is None:
        steps = torch.ones_like(starts, device=starts.device)
    else:
        steps = torch.as_tensor(steps, device=data.device)

    # Normalize negative starts/ends
    dims = torch.as_tensor(data.shape, device=data.device)
    # axes tells where to read dim size
    dim_sizes = dims[axes]

    starts = torch.where(starts < 0, dim_sizes + starts, starts)
    ends   = torch.where(ends   < 0, dim_sizes + ends,   ends)

    # Clip to bounds (ONNX semantics)
    # Use tensors for both min and max to avoid type mismatch
    zero = torch.zeros_like(dim_sizes, device=dim_sizes.device)
    starts = torch.clamp(starts, min=zero, max=dim_sizes)
    ends   = torch.clamp(ends,   min=zero, max=dim_sizes)

    # Build index tuple dynamically
    index = [slice(None)] * data.ndim
    for i in range(axes.shape[0]):
        ax = axes[i].item()
        idx = torch.arange(starts[i], ends[i], steps[i], device=data.device)
        index[ax] = idx

    return data[tuple(index)]'''


SCATTER_ND_HELPER = '''\
def scatter_nd(data, indices, updates, reduction="none"):
    """PyTorch equivalent of ONNX ScatterND using tensor-based indexing.

    indices: (..., K)
    updates: broadcastable to indices[..., :-1] + data.shape[K:]
    reduction: "none", "add" (ONNX opset >= 11)
    """
    result = data.clone()
    indices = indices.to(torch.long)

    # Ensure updates has the same dtype as result (important for float64 testing)
    updates = updates.to(result.dtype)

    # Convert indices (..., K) -> tuple of K tensors
    idx_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))

    if reduction == "none":
        result.index_put_(idx_tuple, updates)
    elif reduction == "add":
        result.index_put_(idx_tuple, updates, accumulate=True)
    else:
        raise NotImplementedError(f"Unsupported reduction: {reduction}")

    return result'''


EXPAND_HELPER = '''\
def dynamic_expand(data, target_shape):
    """Dynamic expand helper for ONNX Expand operation.

    Handles dimension mismatches and ONNX semantics conversion:
    - ONNX allows expanding from higher dims to lower dims (e.g., 4D->3D)
    - ONNX uses 1 to mean "keep dimension", PyTorch uses -1
    """
    # Ensure target_shape is a list of integers
    if isinstance(target_shape, torch.Tensor):
        target_shape = target_shape.to(torch.int64).tolist()
    # Convert to integers (handle cases where list contains floats or numpy ints)
    target_shape = [int(x) for x in target_shape]

    # If data has more dimensions than target, squeeze leading dimensions
    if data.ndim > len(target_shape):
        # Remove leading dimensions by reshaping to match target length
        new_shape = tuple(int(s) for s in data.shape[data.ndim - len(target_shape):])
        data = data.reshape(new_shape)

    # Convert ONNX semantics to PyTorch
    # For each dimension in target_shape:
    # - If target[i] == 1 and data has a non-1 dim at that position,
    #   keep data's dim (-1)
    # - Otherwise, use target[i]
    converted_shape = []
    for i in range(len(target_shape)):
        if i < len(target_shape) - data.ndim:
            # Dimension doesn't exist in data, use target value
            converted_shape.append(int(target_shape[i]))
        else:
            # Dimension exists in data
            data_idx = i - (len(target_shape) - data.ndim)
            if target_shape[i] == 1 and data.shape[data_idx] != 1:
                # Keep data's dimension
                converted_shape.append(-1)
            else:
                # Use target dimension
                converted_shape.append(int(target_shape[i]))

    return data.expand(converted_shape)'''


# =============================================================================
# VMAP-COMPATIBLE HELPER TEMPLATES
# =============================================================================
# These helpers are designed for improved vmap compatibility:
# - scatter_nd: uses functional torch.scatter instead of in-place index_put_
# - dynamic_expand: handles tensor shapes for vmap contexts
# - dynamic_slice: uses torch.gather, but requires .item() for slice length
#   (vmap works only when all batch elements have the same slice bounds)
#
# Note: Models with input-dependent slice bounds that vary across batch elements
# cannot be vmapped - this is a fundamental limitation, not a bug.

DYNAMIC_SLICE_VMAP_HELPER = '''\
def dynamic_slice(data, starts, ends, axes=None, steps=None, slice_lengths=None):
    """Vmap-compatible dynamic slice helper for ONNX Slice operation.

    Returns a tuple (result, valid_flag) where:
    - result: The sliced data (zeros if slice was empty/out-of-bounds)
    - valid_flag: 1.0 if slice was non-empty, 0.0 if empty

    The valid_flag should be accumulated across multiple slices and passed
    to scatter_nd to determine whether to actually perform the scatter.

    For vmap compatibility:
    - axes/steps MUST be constant (Python ints or lists)
    - starts/ends can be tensors (input-dependent)
    - slice_lengths MUST be provided (list of ints, one per axis)

    Args:
        data: Input tensor to slice
        starts: Start indices (tensor or list)
        ends: End indices (tensor or list)
        axes: Axes to slice along (constant list or None)
        steps: Step sizes (constant list or None, must be 1 for now)
        slice_lengths: Static slice lengths for each axis (REQUIRED for vmap mode)
    """
    # Convert to tensors if needed
    starts = torch.as_tensor(starts, device=data.device)
    ends = torch.as_tensor(ends, device=data.device)

    # Handle axes - MUST be constant for vmap compatibility
    if axes is None:
        axes_list = list(range(starts.numel()))
    elif isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    elif isinstance(axes, torch.Tensor):
        axes_list = axes.tolist()
        if not isinstance(axes_list, list):
            axes_list = [axes_list]
    else:
        axes_list = [int(axes)]

    # Handle steps - MUST be constant for vmap compatibility
    if steps is None:
        steps_list = [1] * len(axes_list)
    elif isinstance(steps, (list, tuple)):
        steps_list = list(steps)
    elif isinstance(steps, torch.Tensor):
        steps_list = steps.tolist()
        if not isinstance(steps_list, list):
            steps_list = [steps_list]
    else:
        steps_list = [int(steps)]

    # Handle slice_lengths - default to 1 if not provided
    if slice_lengths is None:
        lengths_list = [1] * len(axes_list)
    else:
        lengths_list = list(slice_lengths)

    result = data
    # Track validity: 1.0 if all slices non-empty, 0.0 if any slice empty
    cumulative_valid = torch.ones((), dtype=data.dtype, device=data.device)

    for i, (axis, step, slice_len) in enumerate(zip(axes_list, steps_list, lengths_list)):
        axis = int(axis)
        step = int(step)
        slice_len = int(slice_len)
        dim_size = result.shape[axis]

        # Get start for this axis (handle scalar or 1D tensor)
        if starts.numel() == 1:
            start = starts.reshape(())
        else:
            start = starts[i]

        # Normalize negative start index
        start = torch.where(start < 0, dim_size + start, start)

        # Clamp start to valid range
        start = torch.clamp(start.long(), 0, dim_size)

        # Check if slice would be out of bounds (start + slice_len > dim_size)
        # is_valid = 1.0 if in bounds, 0.0 if out of bounds
        is_valid = (start + slice_len <= dim_size).to(data.dtype)
        cumulative_valid = cumulative_valid * is_valid

        # Generate indices for gather: start, start+step, start+2*step, ...
        # These are relative offsets that we add to start
        offsets = torch.arange(slice_len, device=data.device, dtype=torch.long) * step

        # Compute actual indices: start + offsets
        # Clamp to valid range (even if out of bounds, we need valid indices for gather)
        indices = (start + offsets).clamp(0, dim_size - 1)

        # Reshape indices for gather: [slice_len] -> proper broadcast shape
        # indices needs shape where axis dimension is slice_len, others are 1
        shape = [1] * result.ndim
        shape[axis] = slice_len
        indices = indices.view(*shape)

        # Expand indices to match result shape except for the slice axis
        expand_shape = list(result.shape)
        expand_shape[axis] = slice_len
        indices = indices.expand(*expand_shape)

        # Gather along axis
        result = torch.gather(result, axis, indices)

    # Return both the result and validity flag
    # Result is multiplied by validity so out-of-bounds slices return zeros
    return result * cumulative_valid, cumulative_valid'''


SCATTER_ND_VMAP_HELPER = '''\
def scatter_nd(data, indices, updates, reduction="none", valid=None):
    """Vmap-compatible PyTorch equivalent of ONNX ScatterND.

    Uses functional torch.scatter instead of in-place index_put_ to be
    compatible with vmap and functorch transforms.

    Args:
        data: Target tensor to scatter into
        indices: (..., K) where K is the number of dimensions to index
        updates: Values to scatter
        reduction: "none" or "add"
        valid: Optional validity flag (scalar tensor). If provided and < 0.5,
               returns data unchanged (simulates empty scatter from empty slices).

    When valid=0 (from out-of-bounds slices), returns original data unchanged,
    matching the behavior of standard mode where empty slices cause no scatter.
    """
    indices = indices.to(torch.long)
    updates = updates.to(data.dtype)

    # Compute linear indices from N-D indices
    # strides[i] = product of data.shape[i+1:]
    data_shape = torch.tensor(data.shape, device=data.device, dtype=torch.long)
    k = indices.shape[-1]

    # Compute strides for the first K dimensions
    strides = torch.ones(k, device=data.device, dtype=torch.long)
    for i in range(k - 2, -1, -1):
        strides[i] = strides[i + 1] * data_shape[i + 1]

    # Compute linear indices
    linear_idx = (indices * strides).sum(dim=-1)

    # Flatten data and updates
    flat_data = data.reshape(-1)
    flat_updates = updates.reshape(-1)
    linear_idx = linear_idx.reshape(-1)

    # Use functional scatter
    if reduction == "none":
        scattered = flat_data.scatter(0, linear_idx, flat_updates)
    elif reduction == "add":
        scattered = flat_data.scatter_add(0, linear_idx, flat_updates)
    else:
        raise NotImplementedError(f"Unsupported reduction: {reduction}")

    result = scattered.reshape(data.shape)

    # If valid flag provided, use it to select between scattered result and original data
    # valid > 0.5 means slices were non-empty, so use scattered result
    # valid <= 0.5 means slices were empty, so return original data unchanged
    if valid is not None:
        # Use torch.where for vmap compatibility (no Python if/else branching)
        result = torch.where(valid > 0.5, result, data)

    return result'''


EXPAND_VMAP_HELPER = '''\
def dynamic_expand(data, target_shape):
    """Vmap-compatible dynamic expand helper for ONNX Expand operation.

    This version handles the conversion from ONNX semantics to PyTorch
    while remaining compatible with vmap.

    Note: For full vmap compatibility, target_shape should be constant
    (known at code generation time). Dynamic target_shape from tensors
    may still work but with limitations.
    """
    # Convert target_shape to list of ints
    if isinstance(target_shape, torch.Tensor):
        target_shape = target_shape.to(torch.int64).tolist()
    target_shape = [int(x) for x in target_shape]

    # If data has more dimensions than target, squeeze leading dimensions
    if data.ndim > len(target_shape):
        new_shape = tuple(int(s) for s in data.shape[data.ndim - len(target_shape):])
        data = data.reshape(new_shape)

    # Convert ONNX semantics to PyTorch
    # ONNX: 1 means "keep dimension", PyTorch: -1 means "keep dimension"
    converted_shape = []
    offset = len(target_shape) - data.ndim

    for i in range(len(target_shape)):
        if i < offset:
            # Dimension doesn't exist in data, use target value
            converted_shape.append(target_shape[i])
        else:
            # Dimension exists in data
            data_idx = i - offset
            if target_shape[i] == 1 and data.shape[data_idx] != 1:
                # Keep data's dimension
                converted_shape.append(-1)
            else:
                # Use target dimension
                converted_shape.append(target_shape[i])

    return data.expand(*converted_shape)'''
