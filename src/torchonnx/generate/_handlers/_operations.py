"""OPERATION handlers for code generation.

Handlers for functional operations (OPERATION class).
Generate code like: x2 = x0.reshape(...), x3 = F.pad(x1, ...), etc.
"""

__docformat__ = "restructuredtext"
__all__ = ["register_operation_handlers"]

from torchonnx.analyze import ConstantInfo, ParameterInfo, SemanticLayerIR, VariableInfo
from torchonnx.generate._handlers._registry import register_handler
from torchonnx.generate._utils import format_argument


def _get_input_code_names(layer: SemanticLayerIR) -> list[str]:
    """Get code names for all inputs.

    Adds self. prefix for parameters and constants (registered as buffers).
    Marks constants/parameters as used in the forward generation context.

    :param layer: Semantic layer IR
    :return: List of code names
    """
    from torchonnx.generate._forward_gen import get_forward_gen_context

    names = []
    ctx = get_forward_gen_context()

    for inp in layer.inputs:
        if isinstance(inp, (ParameterInfo, ConstantInfo)):
            names.append(f"self.{inp.code_name}")
            # Mark as used in context
            if ctx:
                if isinstance(inp, ConstantInfo):
                    ctx.mark_constant_used(inp.code_name)
                else:
                    ctx.mark_parameter_used(inp.code_name)
        else:
            names.append(inp.code_name)
    return names


def _get_input_code_name_selective(
    inp: VariableInfo | ParameterInfo | ConstantInfo,
) -> str:
    """Get code name for a single input, marking constants/parameters as used.

    :param inp: Input info
    :return: Code name string (e.g., 'self.const_0', 'x1')
    """
    from torchonnx.generate._forward_gen import get_forward_gen_context

    if isinstance(inp, ConstantInfo):
        # Use buffer reference, mark as used
        ctx = get_forward_gen_context()
        if ctx:
            ctx.mark_constant_used(inp.code_name)
        return f"self.{inp.code_name}"
    if isinstance(inp, ParameterInfo):
        # Parameters always use buffer reference
        ctx = get_forward_gen_context()
        if ctx:
            ctx.mark_parameter_used(inp.code_name)
        return f"self.{inp.code_name}"
    # Variable - just return code name
    return inp.code_name


def _format_args_with_inputs(layer: SemanticLayerIR, extra_inputs: list[str] | None = None) -> str:
    """Format function arguments (inputs + keyword args).

    :param layer: Semantic layer IR
    :param extra_inputs: Additional input names to prepend
    :return: Formatted argument string
    """
    all_inputs = (extra_inputs or []) + _get_input_code_names(layer)
    args_parts = all_inputs.copy()

    # Add keyword arguments
    for arg in layer.arguments:
        if arg.pytorch_name and arg.value is not None:
            formatted_value = format_argument(arg.value)
            args_parts.append(f"{arg.pytorch_name}={formatted_value}")

    return ", ".join(args_parts)


def _compute_inferred_dim(
    input_shape: list[int] | tuple[int | str, ...], shape_list: list
) -> int | None:
    """Compute inferred dimension when shape contains -1.

    :param input_shape: Input tensor shape (only concrete int dimensions are used)
    :param shape_list: Target reshape shape with -1 placeholder
    :return: Inferred dimension or None if can't compute
    """
    import math

    # Convert to list of ints, filtering out symbolic dimensions
    int_shape = [d for d in input_shape if isinstance(d, int)]
    total_elements: int = int(math.prod(int_shape))

    known_product: int = 1
    minus_one_idx: int = -1
    for i, dim in enumerate(shape_list):
        if dim == -1:
            minus_one_idx = i
        elif isinstance(dim, int):
            known_product *= dim

    if known_product <= 0 or minus_one_idx < 0:
        return None

    return total_elements // known_product


def _require_min_inputs(
    layer: SemanticLayerIR, min_count: int, operation_name: str | None = None
) -> None:
    """Validate that a layer has at least the minimum number of inputs.

    Raises ValueError with a descriptive message if validation fails.

    :param layer: Semantic layer IR to validate
    :param min_count: Minimum number of required inputs
    :param operation_name: Optional operation name for error message (defaults to layer.name)
    :raises ValueError: If layer has fewer than min_count inputs

    COVERAGE NOTE: This error path is not tested because it requires malformed IR
    that earlier ONNX validation stages prevent. Keeping as defensive programming.
    """
    if len(layer.inputs) < min_count:
        op_name = operation_name or layer.name
        raise ValueError(
            f"{op_name} requires at least {min_count} input(s), got {len(layer.inputs)}"
        )


def _can_infer_reshape_statically(
    input_shape: tuple[int | str, ...] | None,
    shape_list: list[int],
) -> bool:
    """Check if reshape can be statically inferred with -1 replacement.

    Returns True when input shape is known with all-integer dimensions
    and the target shape only contains integers and -1.

    :param input_shape: Input tensor shape (may contain symbolic dimensions)
    :param shape_list: Target reshape shape with possible -1 placeholder
    :return: True if reshape can be statically inferred
    """
    if not input_shape:
        return False
    return all(isinstance(d, int) for d in input_shape) and all(
        isinstance(d, int) or d == -1 for d in shape_list
    )


def _handle_reshape(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle reshape operation.

    Detects common patterns:
    - Flatten pattern (batch, -1): generates x.flatten(1) for batch-aware flattening
    - Other reshapes: generates x.reshape(shape) with batch dimension preserved

    When the target shape contains -1 and the first dimension is 1 (hardcoded batch),
    we need to compute what -1 resolves to, replace it with the computed value,
    then set the batch dimension to -1 for dynamic batching.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    output = layer.outputs[0].code_name

    _require_min_inputs(layer, 2, "Reshape")

    data = _get_input_code_name_selective(layer.inputs[0])
    shape_input = layer.inputs[1]

    # If shape is a constant, use Python literal directly
    if isinstance(shape_input, ConstantInfo):
        shape_data = shape_input.data
        shape_list = shape_data.tolist()
        if not isinstance(shape_list, list):
            shape_list = [shape_list]

        # Detect flatten pattern: reshape(batch_size, -1) -> flatten(1)
        if len(shape_list) == 2 and shape_list[1] == -1:
            return f"{output} = {data}.flatten(1)"

        # Handle batch-aware reshaping for first dim=1 cases
        if len(shape_list) >= 1 and shape_list[0] == 1:
            if -1 in shape_list:
                input_info = layer.inputs[0]
                input_shape = input_info.shape if hasattr(input_info, "shape") else None

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
    """Handle transpose/permute operation.

    Generates: output = input.permute(dims)

    :param layer: Semantic layer IR
    :return: Generated code line
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

    :param layer: Semantic layer IR
    :return: Generated code line
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

    In ONNX opset 13+, axes is the second input tensor.
    In older opsets, axes is an attribute.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
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


def _prepare_concat_inputs(
    layer: SemanticLayerIR,
) -> tuple[list[str], int]:
    """Prepare inputs for concat operation, handling constant batch expansion.

    Processes all layer inputs (constants, parameters, variables) and:
    - Marks parameters and constants as used in code generation context
    - Expands constants with batch dim = 1 when concatenating on non-batch axes
    - Returns list of input code strings and the concat axis

    :param layer: Semantic layer IR
    :return: Tuple of (input_codes: list[str], concat_axis: int)
    """
    from torchonnx.generate._forward_gen import get_forward_gen_context

    ctx = get_forward_gen_context()

    # Get axis argument
    axis_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "axis"), None)
    axis_value = int(axis_arg.value) if axis_arg else 0

    # Find a variable input to get batch size from (for expanding constants)
    # Only needed when concatenating on non-batch dimension
    batch_size_source = None
    if axis_value != 0:
        for inp in layer.inputs:
            if isinstance(inp, VariableInfo):
                batch_size_source = inp.code_name
                break

    inputs = []
    for inp in layer.inputs:
        if isinstance(inp, ConstantInfo):
            # Always use buffer reference for constants
            if ctx:
                ctx.mark_constant_used(inp.code_name)
            const_code = f"self.{inp.code_name}"

            # Check if constant needs batch expansion
            # Conditions: axis != 0, constant has shape[0] == 1, and we have a variable input
            if (
                axis_value != 0
                and batch_size_source
                and inp.shape
                and len(inp.shape) > 0
                and inp.shape[0] == 1
            ):
                # Expand constant to match batch size
                # Use -1 for all dimensions except batch (dim 0)
                expand_dims = [-1] * len(inp.shape)
                expand_dims_str = ", ".join(str(d) for d in expand_dims)
                # Skip first "-1, " to start from dim 1
                dims_part = expand_dims_str[4:]
                const_code = f"{const_code}.expand({batch_size_source}.shape[0], {dims_part})"
            inputs.append(const_code)
        elif isinstance(inp, ParameterInfo):
            # Parameters always use buffer reference
            if ctx:
                ctx.mark_parameter_used(inp.code_name)
            inputs.append(f"self.{inp.code_name}")
        else:
            # Variables use code name directly
            inputs.append(inp.code_name)

    return inputs, axis_value


def _handle_concat(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle concat operation.

    All constants use buffer references (cached in __init__) for performance
    and proper device management.
    Generates: output = torch.cat([x0, x1, ...], dim=...)

    When concatenating on a non-batch dimension (axis != 0), constants with
    batch dimension = 1 need to be expanded to match the dynamic batch size
    of variable inputs.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name (unused)
    :return: Generated code line
    """
    output = layer.outputs[0].code_name
    input_codes, axis_value = _prepare_concat_inputs(layer)
    axis = format_argument(axis_value)

    inputs_list = f"[{', '.join(input_codes)}]"
    return f"{output} = torch.cat({inputs_list}, dim={axis})"


def _handle_pad(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle F.pad operation.

    ONNX Pad inputs: data, pads, [constant_value], [axes]
    PyTorch F.pad: input, pad, mode='constant', value=0

    ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    PyTorch pad format: [last_dim_start, last_dim_end, ..., first_dim_start, first_dim_end]
    Conversion: split into begins/ends, reverse, interleave

    When pads is a constant, preprocess to Python literal for cleaner code.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    _require_min_inputs(layer, 2, "Pad")

    data = inputs[0]
    pads_input = layer.inputs[1]
    pads_code = inputs[1]
    value_input = layer.inputs[2] if len(layer.inputs) >= 3 else None
    value_code = inputs[2] if len(inputs) >= 3 else None

    # Get mode from arguments
    mode_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "mode"), None)
    mode = format_argument(mode_arg.value) if mode_arg and mode_arg.value else "'constant'"

    # Check if pads is a constant - if so, convert to Python literal
    if isinstance(pads_input, ConstantInfo):
        pads_data = pads_input.data.tolist()
        # Convert ONNX pads to PyTorch format
        # ONNX: [begin0, begin1, ..., end0, end1, ...]
        # PyTorch: [last_begin, last_end, ..., first_begin, first_end]
        half = len(pads_data) // 2
        begins = pads_data[:half]
        ends = pads_data[half:]
        # Reverse and interleave
        pytorch_pads = []
        for b, e in zip(reversed(begins), reversed(ends), strict=False):
            pytorch_pads.extend([b, e])
        pads_str = str(pytorch_pads)
    else:
        # Runtime pads - need dynamic conversion
        pads_str = (
            f"(lambda p: [v for pair in zip(p[len(p)//2:][::-1], p[:len(p)//2][::-1]) "
            f"for v in pair])({pads_code}.tolist())"
        )

    # Build argument list
    args_parts = [data, pads_str]
    args_parts.append(f"mode={mode}")

    # Handle value - check if it's a constant scalar
    if value_input:
        if isinstance(value_input, ConstantInfo) and value_input.data.numel() == 1:
            # Constant scalar - use Python literal
            value_literal = format_argument(value_input.data.item())
            args_parts.append(f"value={value_literal}")
        else:
            # Runtime value
            args_parts.append(f"value={value_code}.item()")

    args_str = ", ".join(args_parts)
    return f"{output} = F.pad({args_str})"


def _format_bound_arg(bound_name: str, bound_input, both_bounds_constant: bool) -> str | None:
    """Format a clamp bound argument (min or max).

    :param bound_name: Name of bound ('min' or 'max')
    :param bound_input: Input tensor info
    :param both_bounds_constant: Whether both min and max are constant
    :return: Formatted argument string or None
    """
    if bound_input is None:
        return None

    if both_bounds_constant and isinstance(bound_input, ConstantInfo):
        value = bound_input.data.item()
        if isinstance(value, (int, bool)) or (isinstance(value, float) and value.is_integer()):
            return f"{bound_name}={int(value)}"
        return f"{bound_name}={value}"

    code = _get_input_code_name_selective(bound_input)
    return f"{bound_name}={code}"


def _handle_clip(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle torch.clamp operation.

    Uses literals for constant min/max values when BOTH are constants.
    This ensures type consistency (PyTorch doesn't allow mixing scalar and tensor bounds).

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    output = layer.outputs[0].code_name

    # Process data input
    data_input = layer.inputs[0]
    if isinstance(data_input, (ParameterInfo, ConstantInfo)):
        from torchonnx.generate._forward_gen import get_forward_gen_context

        ctx = get_forward_gen_context()
        if ctx:
            if isinstance(data_input, ConstantInfo):
                ctx.mark_constant_used(data_input.code_name)
            else:
                ctx.mark_parameter_used(data_input.code_name)
        data_code = f"self.{data_input.code_name}"
    else:
        data_code = data_input.code_name

    args_parts = [data_code]

    # Get min/max inputs
    min_input = layer.inputs[1] if len(layer.inputs) > 1 else None
    max_input = layer.inputs[2] if len(layer.inputs) > 2 else None

    # Check if both bounds are constant
    both_bounds_constant = (min_input is None or isinstance(min_input, ConstantInfo)) and (
        max_input is None or isinstance(max_input, ConstantInfo)
    )

    # Add formatted bound arguments
    min_arg = _format_bound_arg("min", min_input, both_bounds_constant)
    if min_arg:
        args_parts.append(min_arg)

    max_arg = _format_bound_arg("max", max_input, both_bounds_constant)
    if max_arg:
        args_parts.append(max_arg)

    # Add any additional keyword arguments from layer.arguments
    for arg in layer.arguments:
        if arg.pytorch_name and arg.value is not None:
            formatted_value = format_argument(arg.value)
            args_parts.append(f"{arg.pytorch_name}={formatted_value}")

    args_str = ", ".join(args_parts)
    return f"{output} = torch.clamp({args_str})"


def _handle_gather(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle gather operation.

    ONNX Gather: data[indices] along axis
    Uses bracket slicing for simple constant indices, otherwise index_select.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    output = layer.outputs[0].code_name

    _require_min_inputs(layer, 2, "Gather")

    # Process data input (always use buffer reference if needed)
    data = _get_input_code_name_selective(layer.inputs[0])
    indices_input = layer.inputs[1]

    # Get axis argument
    axis_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "axis"), None)
    axis = int(axis_arg.value) if axis_arg else 0

    # Check if indices is a constant - use literals only for small indices (≤10 elements)
    if isinstance(indices_input, ConstantInfo):
        indices_data = indices_input.data
        # Note: INT64_MAX (9223372036854775807) means "until the end", convert to -1
        int64_max = 9223372036854775807

        if indices_data.numel() == 1:
            # Scalar index - use bracket slicing notation (don't mark as used)
            idx_value = int(indices_data.item())
            idx_value = -1 if idx_value == int64_max else idx_value
            # Build slice with proper axis
            slices = [":" for _ in range(axis)] + [str(idx_value)]
            return f"{output} = {data}[{', '.join(slices)}]"
        # All indices use buffer reference for device control and performance
        # (avoid creating tensors on every forward pass)
        indices_code = _get_input_code_name_selective(indices_input)
        return f"{output} = {data}.index_select({axis}, {indices_code}.reshape(-1).long())"

    # Runtime indices - need dynamic handling (mark as used)
    indices_code = _get_input_code_name_selective(indices_input)
    # ONNX Gather with scalar indices reduces that dimension
    # Use index_select then squeeze if scalar, else just index_select
    return (
        f"{output} = {data}.index_select({axis}, {indices_code}.reshape(-1).long())"
        f".squeeze({axis}) if {indices_code}.dim() == 0 else "
        f"{data}.index_select({axis}, {indices_code}.reshape(-1).long())"
    )


def _handle_constant_of_shape(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle ConstantOfShape operation.

    Creates a tensor of given shape filled with a constant value.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    output = layer.outputs[0].code_name

    # Get value argument and determine dtype
    value_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "value"), None)
    dtype_str = ""
    if value_arg:
        val = value_arg.value
        # Extract dtype from numpy array if available
        if hasattr(val, "dtype"):
            # Use string comparison for dtype mapping
            dtype_name = str(val.dtype)
            dtype_map = {
                "int64": "torch.int64",
                "int32": "torch.int32",
                "float32": "torch.float32",
                "float64": "torch.float64",
            }
            if dtype_name in dtype_map:
                dtype_str = f", dtype={dtype_map[dtype_name]}"
            # Extract scalar value from array
            value = format_argument(float(val.flatten()[0]))
        else:
            value = format_argument(val)
    else:
        value = "0.0"

    # Process shape input
    shape_input = layer.inputs[0]

    # Get device from first forward input
    from torchonnx.generate._forward_gen import get_forward_gen_context

    ctx = get_forward_gen_context()
    device_expr = f"{ctx.first_input_name}.device" if ctx and ctx.first_input_name else "'cpu'"

    if isinstance(shape_input, ConstantInfo):
        # Use literal shape (don't mark as used)
        shape_data = shape_input.data.tolist()
        if isinstance(shape_data, list):
            shape_literal = tuple(shape_data)
        else:
            shape_literal = (shape_data,)
        return f"{output} = torch.full({shape_literal}, {value}{dtype_str}, device={device_expr})"
    # Dynamic shape (mark as used)
    shape_code = _get_input_code_name_selective(shape_input)
    return f"{output} = torch.full({shape_code}.tolist(), {value}{dtype_str}, device={device_expr})"


def _handle_arange(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Range/arange operation.

    Uses literals for constant arguments to avoid buffer initialization.
    Creates arange on the same device as the model.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    output = layer.outputs[0].code_name

    # Extract literal values for constants (don't mark as used)
    args = []
    has_runtime_value = False
    runtime_arg = None

    for inp in layer.inputs:
        if isinstance(inp, ConstantInfo):
            # Use literal value directly (don't mark as used)
            value = inp.data.item()
            # Format as int or float appropriately
            if isinstance(value, (int, bool)) or (isinstance(value, float) and value.is_integer()):
                args.append(str(int(value)))
            else:
                args.append(str(value))
        else:
            # Runtime value (mark as used if parameter)
            runtime_arg = _get_input_code_name_selective(inp)
            assert runtime_arg is not None
            args.append(runtime_arg)
            has_runtime_value = True

    # Add device parameter: use runtime value's device if available, else first input's device
    from torchonnx.generate._forward_gen import get_forward_gen_context

    if has_runtime_value and runtime_arg:
        device_expr = f", device={runtime_arg}.device"
    else:
        ctx = get_forward_gen_context()
        device_name = f"{ctx.first_input_name}.device" if ctx and ctx.first_input_name else "'cpu'"
        device_expr = f", device={device_name}"

    if len(args) >= 3:
        return f"{output} = torch.arange({args[0]}, {args[1]}, {args[2]}{device_expr})"
    if len(args) == 2:
        return f"{output} = torch.arange({args[0]}, {args[1]}{device_expr})"
    return f"{output} = torch.arange({args[0]}{device_expr})"


def _generate_literal_slice(
    data: str, starts: list, ends: list, axes: list, steps: list, output: str
) -> str:
    """Generate Python literal slicing code.

    :param data: Data variable name
    :param starts: Start indices
    :param ends: End indices
    :param axes: Axes to slice
    :param steps: Step sizes
    :param output: Output variable name
    :return: Generated code
    """
    int64_max = 9223372036854775807
    slice_parts = {}

    for i, axis in enumerate(axes):
        start = starts[i] if starts[i] != int64_max else 0
        end = "" if ends[i] == int64_max else ends[i]
        step = steps[i]

        # Build slice string for this axis
        if step == 1:
            if end == "":
                slice_parts[axis] = f"{start}:" if start != 0 else ":"
            else:
                slice_parts[axis] = f"{start}:{end}"
        else:
            if end == "":
                slice_parts[axis] = f"{start}::{step}" if start != 0 else f"::{step}"
            else:
                slice_parts[axis] = f"{start}:{end}:{step}"

    # Build slice tuple
    max_axis = max(axes) if axes else 0
    slice_strs = [slice_parts.get(ax, ":") for ax in range(max_axis + 1)]
    slice_expr = ", ".join(slice_strs)
    return f"{output} = {data}[{slice_expr}]"


def _try_narrow_slice(
    data: str,
    starts_input,
    ends_input,
    axes_input,
    steps_input,
    output: str,
) -> str | None:
    """Try to use torch.narrow for single-axis constant slicing.

    :param data: Data variable name
    :param starts_input: Starts input (should be ConstantInfo)
    :param ends_input: Ends input (should be ConstantInfo)
    :param axes_input: Axes input
    :param steps_input: Steps input
    :param output: Output variable name
    :return: Generated code or None if narrow can't be used
    """
    # Get axes and steps values
    if axes_input is None:
        axes_list = [0]
    else:
        assert isinstance(axes_input, ConstantInfo)
        axes_list = axes_input.data.tolist()
        if not isinstance(axes_list, list):
            axes_list = [axes_list]

    if steps_input is None:
        steps_list = [1] * len(axes_list)
    else:
        assert isinstance(steps_input, ConstantInfo)
        steps_list = steps_input.data.tolist()
        if not isinstance(steps_list, list):
            steps_list = [steps_list]

    # Only use narrow for single-axis, step=1
    if len(axes_list) != 1 or steps_list[0] != 1:
        return None

    axis = axes_list[0]
    assert isinstance(starts_input, ConstantInfo)
    assert isinstance(ends_input, ConstantInfo)
    start_val = int(starts_input.data.item())
    end_val = int(ends_input.data.item())
    length = end_val - start_val

    if length <= 0:
        return None

    return f"{output} = {data}.narrow({axis}, {start_val}, {length})"


def _normalize_int64_max(values):
    """Convert INT64_MAX to -1 in a list or scalar."""
    if isinstance(values, list):
        return [_normalize_int64_max(v) for v in values]
    return -1 if values == 9223372036854775807 else values


def _encode_slice_input(input_val, default: str) -> str:
    """Encode a slice input (constant or variable) as code.

    :param input_val: Input value (ConstantInfo, VariableInfo, or None)
    :param default: Default value if input is None
    :return: Code representation
    """
    if input_val is None:
        return default
    if isinstance(input_val, ConstantInfo):
        values = _normalize_int64_max(input_val.data.tolist())
        return str(values)

    code_val = _get_input_code_name_selective(input_val)
    assert code_val is not None
    return code_val


def _handle_slice(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Slice operation.

    ONNX Slice: data[starts:ends:steps] along axes
    When all slice parameters are constants, generates Python literal slicing.
    When axes/steps are constant but starts/ends are dynamic, uses torch.narrow.
    Otherwise uses dynamic_slice helper for runtime slicing.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    from torchonnx.generate._forward_gen import get_forward_gen_context

    output = layer.outputs[0].code_name

    # ONNX Slice inputs: data, starts, ends, [axes], [steps]
    if len(layer.inputs) < 3:
        data = _get_input_code_name_selective(layer.inputs[0])
        return f"{output} = {data}"

    data_val = _get_input_code_name_selective(layer.inputs[0])
    assert data_val is not None
    data = data_val
    starts_input = layer.inputs[1]
    ends_input = layer.inputs[2]
    axes_input = layer.inputs[3] if len(layer.inputs) > 3 else None
    steps_input = layer.inputs[4] if len(layer.inputs) > 4 else None

    # Check if all parameters are constants
    all_constants = (
        isinstance(starts_input, ConstantInfo)
        and isinstance(ends_input, ConstantInfo)
        and (axes_input is None or isinstance(axes_input, ConstantInfo))
        and (steps_input is None or isinstance(steps_input, ConstantInfo))
    )

    if all_constants:
        assert isinstance(starts_input, ConstantInfo)
        assert isinstance(ends_input, ConstantInfo)
        starts = starts_input.data.tolist()
        ends = ends_input.data.tolist()
        axes = (
            axes_input.data.tolist()
            if axes_input and isinstance(axes_input, ConstantInfo)
            else list(range(len(starts)))
        )
        steps = (
            steps_input.data.tolist()
            if steps_input and isinstance(steps_input, ConstantInfo)
            else [1] * len(starts)
        )
        return _generate_literal_slice(data, starts, ends, axes, steps, output)

    # Check if we can use torch.narrow
    axes_constant = axes_input is None or isinstance(axes_input, ConstantInfo)
    steps_constant = steps_input is None or isinstance(steps_input, ConstantInfo)
    starts_constant = isinstance(starts_input, ConstantInfo)
    ends_constant = isinstance(ends_input, ConstantInfo)

    if axes_constant and steps_constant and starts_constant and ends_constant:
        narrow_code = _try_narrow_slice(
            data, starts_input, ends_input, axes_input, steps_input, output
        )
        if narrow_code:
            return narrow_code

    # Fall back to dynamic_slice helper
    ctx = get_forward_gen_context()
    if ctx:
        ctx.needs_dynamic_slice = True

    starts_code = _encode_slice_input(starts_input, "None")
    ends_code = _encode_slice_input(ends_input, "None")
    axes_code = _encode_slice_input(axes_input, "None")
    steps_code = _encode_slice_input(steps_input, "None")

    # In vmap mode, dynamic_slice returns (result, valid_flag) tuple
    if ctx and ctx.vmap_mode:
        slice_lengths = ctx.get_slice_lengths(layer.name)
        slice_lengths_code = str(slice_lengths) if slice_lengths else "None"
        valid_var = f"{output}_valid"
        fn_args = (
            f"{data}, {starts_code}, {ends_code}, {axes_code}, {steps_code}, {slice_lengths_code}"
        )
        valid_update = f"_slice_valid = _slice_valid * {valid_var}"
        return f"{output}, {valid_var} = dynamic_slice({fn_args}); {valid_update}"

    slice_args = f"{data}, {starts_code}, {ends_code}, {axes_code}, {steps_code}"
    return f"{output} = dynamic_slice({slice_args})"


def _handle_cast(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Cast operation.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    # Get to argument (ONNX data type)
    to_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "to"), None)

    # Map ONNX types to PyTorch dtypes
    onnx_to_torch_dtype = {
        1: "torch.float32",
        2: "torch.uint8",
        3: "torch.int8",
        6: "torch.int32",
        7: "torch.int64",
        9: "torch.bool",
        10: "torch.float16",
        11: "torch.float64",
    }

    if to_arg:
        dtype = onnx_to_torch_dtype.get(to_arg.value, "torch.float32")
        return f"{output} = {inputs[0]}.to({dtype})"
    return f"{output} = {inputs[0]}"


def _handle_shape(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Shape operation.

    ONNX Shape returns int64 tensor on the same device as the input.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    return (
        f"{output} = torch.tensor({inputs[0]}.shape, dtype=torch.int64, device={inputs[0]}.device)"
    )


def _convert_expand_semantics(data_shape: tuple[int, ...], target_shape: list[int]) -> list[int]:
    """Convert ONNX expand semantics to PyTorch expand format.

    In ONNX: if target[i]==1 and data[i]!=1, expand keeps original dimension.
    This function converts to PyTorch format using -1 to indicate "keep original".

    :param data_shape: Input data shape
    :param target_shape: Target shape from constant
    :return: Converted shape with -1 for dimensions to keep
    """
    converted = []
    data_ndim = len(data_shape)
    target_ndim = len(target_shape)

    for i, t in enumerate(target_shape):
        if i < target_ndim - data_ndim:
            # New dimension from target
            converted.append(int(t))
        else:
            # Existing dimension: check if we should keep original (use -1)
            data_idx = i - (target_ndim - data_ndim)
            if t == 1 and data_shape[data_idx] != 1:
                converted.append(-1)  # Keep original dimension
            else:
                converted.append(int(t))

    return converted


def _handle_expand_constant_shape(
    data: str,
    shape_input: ConstantInfo,
    data_shape: tuple[int | str, ...] | None,
    output: str,
) -> str:
    """Handle expand with constant shape input.

    :param data: Code name for data input
    :param shape_input: Constant shape input
    :param data_shape: Data input shape if known (may contain symbolic dims), None otherwise
    :param output: Output variable name
    :return: Generated code or None to defer to runtime helper
    """
    from torchonnx.generate._forward_gen import get_forward_gen_context

    target_shape = shape_input.data.tolist()
    if not isinstance(target_shape, list):
        target_shape = [target_shape]

    # If we know both shapes with all concrete (integer) dimensions, convert semantics
    if data_shape and all(isinstance(d, int) for d in data_shape):
        # Extract all int elements (we've verified all are int above)
        int_dims = [d for d in data_shape if isinstance(d, int)]
        concrete_shape: tuple[int, ...] = tuple(int_dims)
        converted = _convert_expand_semantics(concrete_shape, target_shape)
        return f"{output} = {data}.expand({converted})"

    # Data shape unknown or contains symbolic dims - use dynamic helper
    ctx = get_forward_gen_context()
    if ctx:
        ctx.needs_dynamic_expand = True
    return f"{output} = dynamic_expand({data}, {target_shape})"


def _handle_expand_runtime_shape(
    data: str,
    shape_input: VariableInfo | ParameterInfo | ConstantInfo,
    output: str,
) -> str:
    """Handle expand with runtime shape input.

    :param data: Code name for data input
    :param shape_input: Runtime shape input
    :param output: Output variable name
    :return: Generated code
    """
    from torchonnx.generate._forward_gen import get_forward_gen_context

    ctx = get_forward_gen_context()
    if ctx:
        ctx.needs_dynamic_expand = True
    shape_code = _get_input_code_name_selective(shape_input)
    return f"{output} = dynamic_expand({data}, {shape_code})"


def _handle_expand(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle expand operation.

    If we know the output shape from ONNX shape inference, simply use reshape.
    For constant shapes, use inline .expand() with ONNX->PyTorch semantics conversion.
    Otherwise, use dynamic_expand helper for runtime shapes.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    output = layer.outputs[0].code_name

    _require_min_inputs(layer, 2, "Expand")

    # Process data input
    data = _get_input_code_name_selective(layer.inputs[0])
    assert data is not None, "Expand data input must have a code name"
    shape_input = layer.inputs[1]

    # Get the output shape from ONNX shape inference
    output_info = layer.outputs[0]
    output_shape = output_info.shape if output_info else None

    # If we know the output shape with all concrete dimensions, just use reshape
    if output_shape and all(isinstance(dim, int) for dim in output_shape):
        return f"{output} = {data}.reshape({tuple(output_shape)})"

    # For constant shapes, use optimized constant shape handler
    if isinstance(shape_input, ConstantInfo):
        # Get data shape info if available
        data_input = layer.inputs[0]
        data_shape = None
        if hasattr(data_input, "shape") and data_input.shape:
            data_shape = data_input.shape
        return _handle_expand_constant_shape(data, shape_input, data_shape, output)

    # Runtime shape - use helper
    return _handle_expand_runtime_shape(data, shape_input, output)


def _handle_split(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle split operation.

    ONNX Split uses 'axis', PyTorch uses 'dim'.
    ONNX Split can have multiple outputs that need to be unpacked.
    Uses literals for constant split sizes.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    # Process data input (first input)
    data = _get_input_code_name_selective(layer.inputs[0])

    # Get axis argument and convert to dim
    axis_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "axis"), None)
    dim = format_argument(axis_arg.value) if axis_arg else "0"

    # Process split sizes (second input if present)
    split_sizes_code = None
    if len(layer.inputs) >= 2:
        split_input = layer.inputs[1]
        if isinstance(split_input, ConstantInfo):
            # Use literal (don't mark as used)
            split_sizes_code = str(split_input.data.tolist())
        else:
            # Use buffer reference (mark as used)
            split_code = _get_input_code_name_selective(split_input)
            split_sizes_code = f"{split_code}.tolist()"

    # Handle multiple outputs - ONNX Split produces multiple tensors
    if len(layer.outputs) > 1:
        output_names = ", ".join(out.code_name for out in layer.outputs)
        if split_sizes_code:
            return f"{output_names} = {data}.split({split_sizes_code}, dim={dim})"
        return f"{output_names} = {data}.chunk({len(layer.outputs)}, dim={dim})"
    # Single output - return as tuple (rare but possible)
    output = layer.outputs[0].code_name
    if split_sizes_code:
        return f"{output} = {data}.split({split_sizes_code}, dim={dim})"
    return f"{output} = {data}.chunk(2, dim={dim})"


def _handle_scatter_nd(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle ScatterND operation.

    ScatterND requires a helper function as there's no direct PyTorch equivalent.
    The helper wraps torch.index_put_ with proper index format conversion.

    In vmap mode, passes the accumulated validity flag from dynamic_slice operations.
    When validity is 0 (any slice was empty/out-of-bounds), scatter_nd returns
    the original data unchanged, matching standard mode's empty-tensor behavior.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    from torchonnx.generate._forward_gen import get_forward_gen_context

    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    # Mark that we need the scatter_nd helper
    ctx = get_forward_gen_context()
    if ctx:
        ctx.needs_scatter_nd = True

    # ScatterND(data, indices, updates) - no direct PyTorch equivalent
    if len(inputs) >= 3:
        # In vmap mode, pass the accumulated validity flag
        if ctx and ctx.vmap_mode:
            return (
                f"{output} = scatter_nd({inputs[0]}, {inputs[1]}, {inputs[2]}, valid=_slice_valid)"
            )
        return f"{output} = scatter_nd({inputs[0]}, {inputs[1]}, {inputs[2]})"
    return f"{output} = {inputs[0]}"


def _handle_reduce(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle reduce operations (torch.sum, torch.mean).

    ONNX uses 'keepdims' but PyTorch uses 'keepdim' (no 's').
    ONNX passes axes as second input, PyTorch uses 'dim' keyword.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    # First input is data, second might be axes (for opset 13+)
    data = inputs[0]
    axes_input = inputs[1] if len(inputs) >= 2 else None

    # Build arguments
    args_parts = [data]

    # Add dim argument - can come from input or from arguments
    if axes_input:
        args_parts.append(f"{axes_input}.tolist()")
    else:
        # Check arguments for axes
        axes_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "axes"), None)
        if axes_arg and axes_arg.value is not None:
            args_parts.append(format_argument(axes_arg.value))

    # Add keepdim (note: PyTorch uses 'keepdim' not 'keepdims')
    keepdims_arg = next((arg for arg in layer.arguments if arg.pytorch_name == "keepdims"), None)
    if keepdims_arg and keepdims_arg.value is not None:
        args_parts.append(f"keepdim={format_argument(keepdims_arg.value)}")

    args_str = ", ".join(args_parts)
    return f"{output} = {layer.pytorch_type}({args_str})"


def _handle_linear(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle F.linear operation.

    F.linear(input, weight, bias=None) performs a linear transformation.

    ONNX Gemm behavior depends on transB attribute:
      - transB=0 (default): weight shape = (in_features, out_features) → needs transpose
      - transB=1: weight shape = (out_features, in_features) → already correct
    PyTorch F.linear expects: weight shape = (out_features, in_features)

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    _require_min_inputs(layer, 2, "Linear")

    # F.linear takes input, weight, and optional bias
    data = inputs[0]
    weight = inputs[1]
    bias = inputs[2] if len(inputs) >= 3 else None

    # Check transB attribute to determine if weight needs transpose
    trans_b = 0  # Default value
    for arg in layer.arguments:
        if arg.pytorch_name == "transB":
            trans_b = arg.value
            break

    # Only transpose if transB=0 (ONNX weight is in_features x out_features)
    if trans_b == 0:
        weight = f"{weight}.t()"

    if bias:
        return f"{output} = F.linear({data}, {weight}, {bias})"
    return f"{output} = F.linear({data}, {weight})"


def _get_conv_func_from_ndim(ndim: int) -> str | None:
    """Get conv function name from tensor dimensionality.

    :param ndim: Number of dimensions (3=1d, 4=2d, 5=3d)
    :return: Conv function name or None if invalid
    """
    if ndim == 3:
        return "F.conv1d"
    if ndim == 4:
        return "F.conv2d"
    if ndim == 5:
        return "F.conv3d"
    return None


def _determine_conv_func(layer: SemanticLayerIR) -> str:
    """Determine conv function type from shape information.

    First tries weight shape (most reliable), then data shape.
    Raises error if neither can be determined.

    :param layer: Semantic layer IR
    :return: Conv function name (F.conv1d, F.conv2d, or F.conv3d)
    """
    # Try weight shape (most reliable)
    weight_input = layer.inputs[1]
    if weight_input.shape:
        conv_func = _get_conv_func_from_ndim(len(weight_input.shape))
        if conv_func:
            return conv_func

    # Try data input shape
    data_input = layer.inputs[0]
    if data_input.shape:
        conv_func = _get_conv_func_from_ndim(len(data_input.shape))
        if conv_func:
            return conv_func

    # Could not determine
    raise ValueError(
        f"Cannot determine conv dimensionality statically for layer {layer.name}. "
        "Missing shape information for both input and weight."
    )


def _handle_conv(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle F.conv operation with static dimension detection.

    Determines whether to use F.conv1d, F.conv2d, or F.conv3d based on shape information.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    output = layer.outputs[0].code_name

    _require_min_inputs(layer, 2, "Conv")

    # Determine conv function type from shape information
    conv_func = _determine_conv_func(layer)

    # Build arguments
    args_str = _format_args_with_inputs(layer)

    return f"{output} = {conv_func}({args_str})"


def _handle_generic_torch_function(
    layer: SemanticLayerIR, layer_name_mapping: dict[str, str]
) -> str:
    """Handle generic torch.* functions.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    output = layer.outputs[0].code_name
    args_str = _format_args_with_inputs(layer)
    return f"{output} = {layer.pytorch_type}({args_str})"


def _handle_generic_method(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle generic tensor methods.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    if not inputs:
        raise ValueError(f"Method {layer.pytorch_type} requires at least one input")

    # Extract method name from pytorch_type (e.g., "reshape" from "reshape")
    method_name = layer.pytorch_type

    # Build arguments (excluding first input which is the object)
    remaining_inputs = inputs[1:] if len(inputs) > 1 else []
    args_parts = remaining_inputs.copy()

    # Add keyword arguments
    for arg in layer.arguments:
        if arg.pytorch_name and arg.value is not None:
            formatted_value = format_argument(arg.value)
            args_parts.append(f"{arg.pytorch_name}={formatted_value}")

    args_str = ", ".join(args_parts) if args_parts else ""

    return f"{output} = {inputs[0]}.{method_name}({args_str})"


def register_operation_handlers() -> None:
    """Register all OPERATION handlers."""
    # Tensor methods
    register_handler("reshape", _handle_reshape)
    register_handler("permute", _handle_transpose)
    register_handler("squeeze", _handle_squeeze)
    register_handler("unsqueeze", _handle_unsqueeze)
    register_handler("shape", _handle_shape)
    register_handler("slice", _handle_slice)
    register_handler("sign", _handle_generic_method)
    register_handler("split", _handle_split)
    register_handler("cos", _handle_generic_method)
    register_handler("sin", _handle_generic_method)
    register_handler("floor", _handle_generic_method)
    register_handler("expand", _handle_expand)
    register_handler("cast", _handle_cast)

    # torch.* functions
    register_handler("torch.cat", _handle_concat)
    register_handler("torch.gather", _handle_gather)
    register_handler("torch.argmax", _handle_generic_torch_function)
    register_handler("torch.min", _handle_generic_torch_function)
    register_handler("torch.max", _handle_generic_torch_function)
    register_handler("torch.clamp", _handle_clip)
    register_handler("torch.where", _handle_generic_torch_function)
    register_handler("torch.full", _handle_constant_of_shape)
    register_handler("torch.arange", _handle_arange)
    register_handler("torch.mean", _handle_reduce)
    register_handler("torch.sum", _handle_reduce)
    register_handler("scatter_nd", _handle_scatter_nd)

    # F.* functions
    register_handler("F.pad", _handle_pad)
    register_handler("F.conv", _handle_conv)
    register_handler("F.conv_transpose", _handle_generic_torch_function)
    register_handler("F.linear", _handle_linear)
    register_handler("F.interpolate", _handle_generic_torch_function)
