"""OPERATION handlers for code generation.

Handlers for functional operations (OPERATION class).
Generate code like: x2 = x0.reshape(...), x3 = F.pad(x1, ...), etc.
"""

__docformat__ = "restructuredtext"
__all__ = ["register_operation_handlers"]

from ._registry import register_handler
from .._utils import format_argument
from ...analyze import (
    SemanticLayerIR,
    VariableInfo,
    ParameterInfo,
    ConstantInfo,
)


def _get_input_code_names(layer: SemanticLayerIR) -> list[str]:
    """Get code names for all inputs.

    Adds self. prefix for parameters and constants (registered as buffers).
    Marks constants/parameters as used in the forward generation context.

    :param layer: Semantic layer IR
    :return: List of code names
    """
    from .._forward_gen import get_forward_gen_context

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
    use_literal: bool = False,
) -> str:
    """Get code name for a single input with optional literal optimization.

    :param inp: Input info
    :param use_literal: If True, use literal for constant values (don't mark as used)
    :return: Code name string
    """
    from .._forward_gen import get_forward_gen_context

    if isinstance(inp, ConstantInfo):
        if use_literal:
            # Use literal value, don't mark as used
            return None  # Caller should extract literal themselves
        else:
            # Use buffer reference, mark as used
            ctx = get_forward_gen_context()
            if ctx:
                ctx.mark_constant_used(inp.code_name)
            return f"self.{inp.code_name}"
    elif isinstance(inp, ParameterInfo):
        # Parameters always use buffer reference
        ctx = get_forward_gen_context()
        if ctx:
            ctx.mark_parameter_used(inp.code_name)
        return f"self.{inp.code_name}"
    else:
        # Variable - just return code name
        return inp.code_name


def _format_args_with_inputs(
    layer: SemanticLayerIR, extra_inputs: list[str] | None = None
) -> str:
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

    if len(layer.inputs) < 2:
        raise ValueError("Reshape requires input and shape")

    # Process data input (always use buffer reference if needed)
    data = _get_input_code_name_selective(layer.inputs[0], use_literal=False)
    shape_input = layer.inputs[1]

    # If shape is a constant, use Python literal directly (don't mark as used)
    if isinstance(shape_input, ConstantInfo):
        shape_data = shape_input.data
        shape_list = shape_data.tolist()
        # Convert to list for modification
        if not isinstance(shape_list, list):
            shape_list = [shape_list]

        # Detect flatten pattern: reshape(batch_size, -1) or reshape(1, -1)
        # This is common before fully connected layers in CNNs
        # Use flatten(1) which is batch-aware: flattens from dim 1 onwards
        if len(shape_list) == 2 and shape_list[1] == -1:
            return f"{output} = {data}.flatten(1)"

        # For other reshapes, make batch-aware by replacing first dim with -1
        # ONNX models are often traced with batch_size=1, but we want dynamic batch
        #
        # If shape contains -1 and first dim is 1, we need to:
        # 1. Compute what -1 resolves to using input shape
        # 2. Replace -1 with computed value
        # 3. Set first dim to -1 for dynamic batch
        if len(shape_list) >= 1 and shape_list[0] == 1:
            if -1 in shape_list:
                # Get input shape to compute the -1 dimension
                input_info = layer.inputs[0]
                input_shape = input_info.shape if hasattr(input_info, 'shape') else None

                if input_shape and all(isinstance(d, int) for d in input_shape):
                    # Calculate total elements (excluding batch dimension)
                    # Input shape includes batch=1, so total = product of all dims
                    import math
                    total_elements = math.prod(input_shape)

                    # Calculate product of known dimensions in target shape
                    known_product = 1
                    minus_one_idx = -1
                    for i, dim in enumerate(shape_list):
                        if dim == -1:
                            minus_one_idx = i
                        else:
                            known_product *= dim

                    # Compute what -1 should be
                    if known_product > 0 and minus_one_idx >= 0:
                        inferred_dim = total_elements // known_product
                        shape_list[minus_one_idx] = inferred_dim
                        # Now set batch dim to -1
                        shape_list[0] = -1
            else:
                # No -1 in shape, safe to replace first dim with -1
                shape_list[0] = -1

        # Convert to tuple for cleaner code
        shape_literal = tuple(shape_list)
        return f"{output} = {data}.reshape{shape_literal}"

    # Dynamic shape - use tolist() at runtime (mark as used)
    # Convert to int to handle cases where shape tensor has float dtype
    shape_code = _get_input_code_name_selective(shape_input, use_literal=False)
    return f"{output} = {data}.reshape([int(x) for x in {shape_code}.tolist()])"


def _handle_transpose(
    layer: SemanticLayerIR, layer_name_mapping: dict[str, str]
) -> str:
    """Handle transpose/permute operation.

    Generates: output = input.permute(dims)

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    # Get perm argument
    perm_arg = next(
        (arg for arg in layer.arguments if arg.pytorch_name == "perm"), None
    )

    if perm_arg:
        perm = format_argument(perm_arg.value)
        return f"{output} = {inputs[0]}.permute({perm})"
    else:
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
    else:
        return f"{output} = {inputs[0]}.squeeze()"


def _handle_unsqueeze(
    layer: SemanticLayerIR, layer_name_mapping: dict[str, str]
) -> str:
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
    data = _get_input_code_name_selective(layer.inputs[0], use_literal=False)

    if dim_arg and dim_arg.value is not None:
        dim = format_argument(dim_arg.value)
        return f"{output} = {data}.unsqueeze({dim})"
    elif len(layer.inputs) >= 2:
        # ONNX opset 13+: axes is the second input
        axes_input = layer.inputs[1]
        if isinstance(axes_input, ConstantInfo):
            # Use literal (don't mark as used)
            axes_value = int(axes_input.data.item())
            return f"{output} = {data}.unsqueeze({axes_value})"
        else:
            # Dynamic axes (mark as used)
            axes_code = _get_input_code_name_selective(axes_input, use_literal=False)
            return f"{output} = {data}.unsqueeze({axes_code}.item())"
    else:
        # Fallback: unsqueeze at dim 0
        return f"{output} = {data}.unsqueeze(0)"


def _handle_concat(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle concat operation.

    All constants use buffer references (cached in __init__) for performance
    and proper device management.
    Generates: output = torch.cat([x0, x1, ...], dim=...)

    When concatenating on a non-batch dimension (axis != 0), constants with
    batch dimension = 1 need to be expanded to match the dynamic batch size
    of variable inputs.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    from .._forward_gen import get_forward_gen_context

    ctx = get_forward_gen_context()

    # Get axis argument
    axis_arg = next(
        (arg for arg in layer.arguments if arg.pytorch_name == "axis"), None
    )
    axis_value = int(axis_arg.value) if axis_arg else 0
    axis = format_argument(axis_value)

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
            if (axis_value != 0 and batch_size_source and
                inp.shape and len(inp.shape) > 0 and inp.shape[0] == 1):
                # Expand constant to match batch size
                # Use -1 for all dimensions except batch (dim 0)
                expand_dims = [-1] * len(inp.shape)
                expand_dims_str = ", ".join(str(d) for d in expand_dims)
                const_code = f"{const_code}.expand({batch_size_source}.shape[0], {expand_dims_str[4:]})"  # Skip first "-1, "
            inputs.append(const_code)
        elif isinstance(inp, ParameterInfo):
            # Parameters always use buffer reference
            if ctx:
                ctx.mark_parameter_used(inp.code_name)
            inputs.append(f"self.{inp.code_name}")
        else:
            # Variables use code name directly
            inputs.append(inp.code_name)

    output = layer.outputs[0].code_name
    inputs_list = f"[{', '.join(inputs)}]"
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

    if len(inputs) < 2:
        raise ValueError("Pad requires input and pads")

    data = inputs[0]
    pads_input = layer.inputs[1]
    pads_code = inputs[1]
    value_input = layer.inputs[2] if len(layer.inputs) >= 3 else None
    value_code = inputs[2] if len(inputs) >= 3 else None

    # Get mode from arguments
    mode_arg = next(
        (arg for arg in layer.arguments if arg.pytorch_name == "mode"), None
    )
    mode = (
        format_argument(mode_arg.value) if mode_arg and mode_arg.value else "'constant'"
    )

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
        for b, e in zip(reversed(begins), reversed(ends)):
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
        from .._forward_gen import get_forward_gen_context

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

    # Check if both min and max are constants (for type consistency)
    min_input = layer.inputs[1] if len(layer.inputs) > 1 else None
    max_input = layer.inputs[2] if len(layer.inputs) > 2 else None

    both_bounds_constant = (
        min_input is None or isinstance(min_input, ConstantInfo)
    ) and (max_input is None or isinstance(max_input, ConstantInfo))

    # Handle min value
    if min_input is not None:
        if both_bounds_constant and isinstance(min_input, ConstantInfo):
            # Use literal (don't mark as used)
            min_value = min_input.data.item()
            if isinstance(min_value, (int, bool)) or (
                isinstance(min_value, float) and min_value.is_integer()
            ):
                args_parts.append(f"min={int(min_value)}")
            else:
                args_parts.append(f"min={min_value}")
        else:
            # Use tensor reference (mark as used)
            min_code = _get_input_code_name_selective(min_input, use_literal=False)
            args_parts.append(f"min={min_code}")

    # Handle max value
    if max_input is not None:
        if both_bounds_constant and isinstance(max_input, ConstantInfo):
            # Use literal (don't mark as used)
            max_value = max_input.data.item()
            if isinstance(max_value, (int, bool)) or (
                isinstance(max_value, float) and max_value.is_integer()
            ):
                args_parts.append(f"max={int(max_value)}")
            else:
                args_parts.append(f"max={max_value}")
        else:
            # Use tensor reference (mark as used)
            max_code = _get_input_code_name_selective(max_input, use_literal=False)
            args_parts.append(f"max={max_code}")

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

    if len(layer.inputs) < 2:
        raise ValueError("Gather requires input and indices")

    # Process data input (always use buffer reference if needed)
    data = _get_input_code_name_selective(layer.inputs[0], use_literal=False)
    indices_input = layer.inputs[1]

    # Get axis argument
    axis_arg = next(
        (arg for arg in layer.arguments if arg.pytorch_name == "axis"), None
    )
    axis = int(axis_arg.value) if axis_arg else 0

    # Check if indices is a constant - use literals only for small indices (≤10 elements)
    if isinstance(indices_input, ConstantInfo):
        indices_data = indices_input.data
        # Note: INT64_MAX (9223372036854775807) means "until the end", convert to -1
        INT64_MAX = 9223372036854775807

        if indices_data.numel() == 1:
            # Scalar index - use bracket slicing notation (don't mark as used)
            idx_value = int(indices_data.item())
            idx_value = -1 if idx_value == INT64_MAX else idx_value
            # Build slice with proper axis
            slices = [":" for _ in range(axis)] + [str(idx_value)]
            return f"{output} = {data}[{', '.join(slices)}]"
        else:
            # All indices use buffer reference for device control and performance
            # (avoid creating tensors on every forward pass)
            indices_code = _get_input_code_name_selective(
                indices_input, use_literal=False
            )
            return f"{output} = {data}.index_select({axis}, {indices_code}.reshape(-1).long())"

    # Runtime indices - need dynamic handling (mark as used)
    indices_code = _get_input_code_name_selective(indices_input, use_literal=False)
    # ONNX Gather with scalar indices reduces that dimension
    # Use index_select then squeeze if scalar, else just index_select
    return (
        f"{output} = {data}.index_select({axis}, {indices_code}.reshape(-1).long())"
        f".squeeze({axis}) if {indices_code}.dim() == 0 else "
        f"{data}.index_select({axis}, {indices_code}.reshape(-1).long())"
    )


def _handle_constant_of_shape(
    layer: SemanticLayerIR, layer_name_mapping: dict[str, str]
) -> str:
    """Handle ConstantOfShape operation.

    Creates a tensor of given shape filled with a constant value.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    output = layer.outputs[0].code_name

    # Get value argument and determine dtype
    value_arg = next(
        (arg for arg in layer.arguments if arg.pytorch_name == "value"), None
    )
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
    from .._forward_gen import get_forward_gen_context

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
    else:
        # Dynamic shape (mark as used)
        shape_code = _get_input_code_name_selective(shape_input, use_literal=False)
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
            if isinstance(value, (int, bool)) or (
                isinstance(value, float) and value.is_integer()
            ):
                args.append(str(int(value)))
            else:
                args.append(str(value))
        else:
            # Runtime value (mark as used if parameter)
            runtime_arg = _get_input_code_name_selective(inp, use_literal=False)
            args.append(runtime_arg)
            has_runtime_value = True

    # Add device parameter: use runtime value's device if available, else first input's device
    from .._forward_gen import get_forward_gen_context

    if has_runtime_value and runtime_arg:
        device_expr = f", device={runtime_arg}.device"
    else:
        ctx = get_forward_gen_context()
        device_name = f"{ctx.first_input_name}.device" if ctx and ctx.first_input_name else "'cpu'"
        device_expr = f", device={device_name}"

    if len(args) >= 3:
        return f"{output} = torch.arange({args[0]}, {args[1]}, {args[2]}{device_expr})"
    elif len(args) == 2:
        return f"{output} = torch.arange({args[0]}, {args[1]}{device_expr})"
    else:
        return f"{output} = torch.arange({args[0]}{device_expr})"


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
    from .._forward_gen import get_forward_gen_context

    output = layer.outputs[0].code_name

    # ONNX Slice inputs: data, starts, ends, [axes], [steps]
    if len(layer.inputs) >= 3:
        # Process data input (always use buffer reference if needed)
        data = _get_input_code_name_selective(layer.inputs[0], use_literal=False)

        # Check if all slice parameters are constants
        starts_input = layer.inputs[1]
        ends_input = layer.inputs[2]
        axes_input = layer.inputs[3] if len(layer.inputs) > 3 else None
        steps_input = layer.inputs[4] if len(layer.inputs) > 4 else None

        # Try to convert to Python literal slicing if all are constants
        all_constants = isinstance(starts_input, ConstantInfo) and isinstance(
            ends_input, ConstantInfo
        )
        if axes_input is not None:
            all_constants = all_constants and isinstance(axes_input, ConstantInfo)
        if steps_input is not None:
            all_constants = all_constants and isinstance(steps_input, ConstantInfo)

        if all_constants:
            # Use literals (don't mark as used)
            starts = starts_input.data.tolist()
            ends = ends_input.data.tolist()
            axes = axes_input.data.tolist() if axes_input else list(range(len(starts)))
            steps = steps_input.data.tolist() if steps_input else [1] * len(starts)

            # Build slice expression for each axis
            # We need to know the data ndim to build proper slices
            # For safety, build a generic slice tuple
            # Note: INT64_MAX (9223372036854775807) means "until the end", omit it
            INT64_MAX = 9223372036854775807
            slice_parts = {}
            for i, axis in enumerate(axes):
                start = starts[i] if starts[i] != INT64_MAX else 0
                # For end: if INT64_MAX, omit it (empty string) to mean "to the end"
                if ends[i] == INT64_MAX:
                    end = ""
                else:
                    end = ends[i]
                step = steps[i]
                # Build slice string
                if step == 1:
                    if end == "":
                        slice_parts[axis] = f"{start}:" if start != 0 else ":"
                    else:
                        slice_parts[axis] = f"{start}:{end}"
                else:
                    if end == "":
                        slice_parts[axis] = (
                            f"{start}::{step}" if start != 0 else f"::{step}"
                        )
                    else:
                        slice_parts[axis] = f"{start}:{end}:{step}"

            # Generate indexing code
            # Sort by axis and build tuple
            max_axis = max(axes) if axes else 0
            slice_strs = []
            for ax in range(max_axis + 1):
                if ax in slice_parts:
                    slice_strs.append(slice_parts[ax])
                else:
                    slice_strs.append(":")

            # Add ... for remaining dimensions if needed
            slice_expr = ", ".join(slice_strs)
            return f"{output} = {data}[{slice_expr}]"

        # Check if we can use torch.narrow for single-axis slicing with step=1
        # This avoids the dynamic_slice helper when axes and steps are constant
        # NOTE: We only use narrow when BOTH starts and ends are constants, because
        # ONNX Slice semantics include bounds clipping that torch.narrow doesn't handle.
        # Dynamic indices could be out of bounds which torch.narrow can't handle.
        axes_constant = axes_input is None or isinstance(axes_input, ConstantInfo)
        steps_constant = steps_input is None or isinstance(steps_input, ConstantInfo)
        starts_constant = isinstance(starts_input, ConstantInfo)
        ends_constant = isinstance(ends_input, ConstantInfo)

        if axes_constant and steps_constant and starts_constant and ends_constant:
            # Get axes and steps values
            if axes_input is None:
                axes_list = [0]  # Default to axis 0
            else:
                axes_list = axes_input.data.tolist()
                if not isinstance(axes_list, list):
                    axes_list = [axes_list]

            if steps_input is None:
                steps_list = [1] * len(axes_list)
            else:
                steps_list = steps_input.data.tolist()
                if not isinstance(steps_list, list):
                    steps_list = [steps_list]

            # Use torch.narrow for single-axis slicing with step=1 and constant bounds
            if len(axes_list) == 1 and steps_list[0] == 1:
                axis = axes_list[0]
                start_val = int(starts_input.data.item())
                end_val = int(ends_input.data.item())
                length = end_val - start_val

                # Only use narrow if we have valid positive length
                if length > 0:
                    return f"{output} = {data}.narrow({axis}, {start_val}, {length})"

        # Fall back to dynamic slice - use literals for constants
        # Mark that we need the dynamic_slice helper
        ctx = get_forward_gen_context()
        if ctx:
            ctx.needs_dynamic_slice = True

        # For constant inputs, extract literal values (don't mark as used)
        # Note: INT64_MAX (9223372036854775807) means "until the end", convert to -1
        INT64_MAX = 9223372036854775807

        def normalize_int64_max(values):
            """Convert INT64_MAX to -1 in a list or scalar."""
            if isinstance(values, list):
                return [normalize_int64_max(v) for v in values]
            return -1 if values == INT64_MAX else values

        if isinstance(starts_input, ConstantInfo):
            starts_values = normalize_int64_max(starts_input.data.tolist())
            starts_code = str(starts_values)
        else:
            starts_code = _get_input_code_name_selective(
                starts_input, use_literal=False
            )

        if isinstance(ends_input, ConstantInfo):
            ends_values = normalize_int64_max(ends_input.data.tolist())
            ends_code = str(ends_values)
        else:
            ends_code = _get_input_code_name_selective(ends_input, use_literal=False)

        if axes_input is None:
            axes_code = "None"
        elif isinstance(axes_input, ConstantInfo):
            axes_code = str(axes_input.data.tolist())
        else:
            axes_code = _get_input_code_name_selective(axes_input, use_literal=False)

        if steps_input is None:
            steps_code = "None"
        elif isinstance(steps_input, ConstantInfo):
            steps_code = str(steps_input.data.tolist())
        else:
            steps_code = _get_input_code_name_selective(steps_input, use_literal=False)

        # Use dynamic_slice helper for complex slicing
        return f"{output} = dynamic_slice({data}, {starts_code}, {ends_code}, {axes_code}, {steps_code})"

    # Fallback for invalid inputs
    data = _get_input_code_name_selective(layer.inputs[0], use_literal=False)
    return f"{output} = {data}"


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

    return f"{output} = torch.tensor({inputs[0]}.shape, dtype=torch.int64, device={inputs[0]}.device)"


def _handle_expand(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle expand operation.

    If we know the output shape from ONNX shape inference, simply use reshape.
    For constant shapes, use inline .expand() with ONNX->PyTorch semantics conversion.
    Otherwise, use dynamic_expand helper for runtime shapes.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    from .._forward_gen import get_forward_gen_context

    output = layer.outputs[0].code_name

    if len(layer.inputs) < 2:
        raise ValueError("Expand requires input and shape")

    # Process data input
    data = _get_input_code_name_selective(layer.inputs[0], use_literal=False)
    shape_input = layer.inputs[1]

    # Get the output shape from ONNX shape inference
    output_info = layer.outputs[0]
    output_shape = output_info.shape if output_info else None

    # If we know the output shape with all concrete dimensions, just use reshape
    # Only use this shortcut if all dimensions are integers (not symbolic like 'unk__35')
    if output_shape and all(isinstance(dim, int) for dim in output_shape):
        return f"{output} = {data}.reshape({tuple(output_shape)})"

    # For constant shapes, try to use inline .expand() with ONNX semantics conversion
    if isinstance(shape_input, ConstantInfo):
        target_shape = shape_input.data.tolist()
        if not isinstance(target_shape, list):
            target_shape = [target_shape]

        # Get data shape info if available
        data_input = layer.inputs[0]
        data_shape = None
        if hasattr(data_input, 'shape') and data_input.shape:
            data_shape = data_input.shape

        # If we know both shapes, we can do the ONNX->PyTorch conversion at codegen time
        if data_shape and all(isinstance(d, int) for d in data_shape):
            # Convert ONNX semantics: if target[i]==1 and data[i]!=1, use -1 (keep dim)
            converted = []
            data_ndim = len(data_shape)
            target_ndim = len(target_shape)
            for i, t in enumerate(target_shape):
                if i < target_ndim - data_ndim:
                    # New dimension from target
                    converted.append(int(t))
                else:
                    # Existing dimension
                    data_idx = i - (target_ndim - data_ndim)
                    if t == 1 and data_shape[data_idx] != 1:
                        converted.append(-1)  # Keep original dimension
                    else:
                        converted.append(int(t))
            return f"{output} = {data}.expand({converted})"

        # Data shape unknown - use helper
        ctx = get_forward_gen_context()
        if ctx:
            ctx.needs_dynamic_expand = True
        return f"{output} = dynamic_expand({data}, {target_shape})"

    # Runtime shape - use helper
    ctx = get_forward_gen_context()
    if ctx:
        ctx.needs_dynamic_expand = True
    shape_code = _get_input_code_name_selective(shape_input, use_literal=False)
    return f"{output} = dynamic_expand({data}, {shape_code})"


def _handle_split(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle split operation.

    ONNX Split uses 'axis', PyTorch uses 'dim'.
    ONNX Split can have multiple outputs that need to be unpacked.
    Uses literals for constant split sizes.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    # Process data input (first input)
    data = _get_input_code_name_selective(layer.inputs[0], use_literal=False)

    # Get axis argument and convert to dim
    axis_arg = next(
        (arg for arg in layer.arguments if arg.pytorch_name == "axis"), None
    )
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
            split_code = _get_input_code_name_selective(split_input, use_literal=False)
            split_sizes_code = f"{split_code}.tolist()"

    # Handle multiple outputs - ONNX Split produces multiple tensors
    if len(layer.outputs) > 1:
        output_names = ", ".join(out.code_name for out in layer.outputs)
        if split_sizes_code:
            return f"{output_names} = {data}.split({split_sizes_code}, dim={dim})"
        else:
            return f"{output_names} = {data}.chunk({len(layer.outputs)}, dim={dim})"
    else:
        # Single output - return as tuple (rare but possible)
        output = layer.outputs[0].code_name
        if split_sizes_code:
            return f"{output} = {data}.split({split_sizes_code}, dim={dim})"
        else:
            return f"{output} = {data}.chunk(2, dim={dim})"


def _handle_scatter_nd(
    layer: SemanticLayerIR, layer_name_mapping: dict[str, str]
) -> str:
    """Handle ScatterND operation.

    ScatterND requires a helper function as there's no direct PyTorch equivalent.
    The helper wraps torch.index_put_ with proper index format conversion.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    from .._forward_gen import get_forward_gen_context

    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    # Mark that we need the scatter_nd helper
    ctx = get_forward_gen_context()
    if ctx:
        ctx.needs_scatter_nd = True

    # ScatterND(data, indices, updates) - no direct PyTorch equivalent
    if len(inputs) >= 3:
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
        axes_arg = next(
            (arg for arg in layer.arguments if arg.pytorch_name == "axes"), None
        )
        if axes_arg and axes_arg.value is not None:
            args_parts.append(format_argument(axes_arg.value))

    # Add keepdim (note: PyTorch uses 'keepdim' not 'keepdims')
    keepdims_arg = next(
        (arg for arg in layer.arguments if arg.pytorch_name == "keepdims"), None
    )
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

    if len(inputs) < 2:
        raise ValueError("Linear requires input and weight")

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

    # Only transpose if transB=0 (ONNX weight is in_features × out_features)
    if trans_b == 0:
        weight = f"{weight}.t()"

    if bias:
        return f"{output} = F.linear({data}, {weight}, {bias})"
    else:
        return f"{output} = F.linear({data}, {weight})"


def _handle_conv(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle F.conv operation with static dimension detection.

    Determines whether to use F.conv1d, F.conv2d, or F.conv3d based on shape information.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    inputs = _get_input_code_names(layer)
    output = layer.outputs[0].code_name

    if len(inputs) < 2:
        raise ValueError("Conv requires input and weight")

    data = inputs[0]
    weight = inputs[1]

    # Determine conv type statically from shape information
    conv_func = None

    # Try to determine from weight shape (most reliable)
    # Weight can be ParameterInfo, ConstantInfo, or VariableInfo (if computed at runtime)
    weight_input = layer.inputs[1]
    if weight_input.shape:
        weight_ndim = len(weight_input.shape)
        # Conv1d weight: (out_channels, in_channels, kernel_size) = 3D
        # Conv2d weight: (out_channels, in_channels, kernel_h, kernel_w) = 4D
        # Conv3d weight: (out_channels, in_channels, kernel_d, kernel_h, kernel_w) = 5D
        if weight_ndim == 3:
            conv_func = "F.conv1d"
        elif weight_ndim == 4:
            conv_func = "F.conv2d"
        elif weight_ndim == 5:
            conv_func = "F.conv3d"

    # If weight doesn't have shape, try input data shape
    if conv_func is None:
        data_input = layer.inputs[0]
        if data_input.shape:
            data_ndim = len(data_input.shape)
            # Conv1d input: (batch, channels, length) = 3D
            # Conv2d input: (batch, channels, height, width) = 4D
            # Conv3d input: (batch, channels, depth, height, width) = 5D
            if data_ndim == 3:
                conv_func = "F.conv1d"
            elif data_ndim == 4:
                conv_func = "F.conv2d"
            elif data_ndim == 5:
                conv_func = "F.conv3d"

    # If still can't determine, raise error (should not happen with valid ONNX)
    if conv_func is None:
        raise ValueError(
            f"Cannot determine conv dimensionality statically for layer {layer.name}. "
            "Missing shape information for both input and weight."
        )

    # Build arguments
    args_str = _format_args_with_inputs(layer)

    return f"{output} = {conv_func}({args_str})"


def _handle_generic_torch_function(
    layer: SemanticLayerIR, layer_name_mapping: dict[str, str]
) -> str:
    """Generic handler for torch.* functions.

    :param layer: Semantic layer IR
    :return: Generated code line
    """
    output = layer.outputs[0].code_name
    args_str = _format_args_with_inputs(layer)
    return f"{output} = {layer.pytorch_type}({args_str})"


def _handle_generic_method(
    layer: SemanticLayerIR, layer_name_mapping: dict[str, str]
) -> str:
    """Generic handler for tensor methods.

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
