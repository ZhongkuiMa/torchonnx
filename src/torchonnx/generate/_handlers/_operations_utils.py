"""Shared utilities for OPERATION-class handlers.

Extracted from the ``_operations.py`` monolith so each individual
handler module can import only what it needs and so the handlers and
the utilities are independently lintable / testable. ``_operations.py``
re-exports these names for back-compatibility with test imports.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "INT64_MAX",
    "_can_infer_reshape_statically",
    "_compute_inferred_dim",
    "_format_args_with_inputs",
    "_get_input_code_name_selective",
    "_get_input_code_names",
    "_require_min_inputs",
]

import math

from torchonnx.analyze import ConstantInfo, ParameterInfo, SemanticLayerIR, VariableInfo
from torchonnx.generate._context import _get_ctx
from torchonnx.generate._utils import format_argument

#: ONNX sentinel signalling "until end of axis" in Slice / Gather indices.
INT64_MAX = 9223372036854775807


def _get_input_code_names(layer: SemanticLayerIR) -> list[str]:
    """Get code names for all inputs.

    Adds ``self.`` prefix for parameters and constants (registered as
    buffers). Marks constants / parameters as used in the forward
    generation context.

    :param layer: Semantic layer IR.

    :return: List of code names.
    """
    names = []
    ctx = _get_ctx()

    for inp in layer.inputs:
        if isinstance(inp, (ParameterInfo, ConstantInfo)):
            names.append(f"self.{inp.code_name}")
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
    """Get code name for a single input, marking constants / parameters as used.

    :param inp: Input info.

    :return: Code name string (e.g., ``'self.const_0'``, ``'x1'``).
    """
    if isinstance(inp, ConstantInfo):
        _get_ctx().mark_constant_used(inp.code_name)
        return f"self.{inp.code_name}"
    if isinstance(inp, ParameterInfo):
        _get_ctx().mark_parameter_used(inp.code_name)
        return f"self.{inp.code_name}"
    # Variable - just return code name
    return inp.code_name


def _format_args_with_inputs(layer: SemanticLayerIR, extra_inputs: list[str] | None = None) -> str:
    """Format function arguments (positional inputs + keyword args).

    Skips arguments whose value equals the PyTorch default; emitting the
    default at codegen time only forces the simplify stage to strip it
    back out via regex line-rewrites, which is fragile (string literals
    containing commas can break the parser) and slow. The Stage 5 default-
    stripping tables remain for handlers that bypass this helper.

    :param layer: Semantic layer IR.
    :param extra_inputs: Additional input names to prepend.

    :return: Formatted argument string.
    """
    all_inputs = (extra_inputs or []) + _get_input_code_names(layer)
    args_parts = all_inputs.copy()

    for arg in layer.arguments:
        if not arg.pytorch_name or arg.value is None:
            continue
        if arg.is_default():
            continue
        formatted_value = format_argument(arg.value)
        args_parts.append(f"{arg.pytorch_name}={formatted_value}")

    return ", ".join(args_parts)


def _compute_inferred_dim(
    input_shape: list[int] | tuple[int | str, ...], shape_list: list
) -> int | None:
    """Compute inferred dimension when shape contains -1.

    :param input_shape: Input tensor shape (only concrete int dimensions are used).
    :param shape_list: Target reshape shape with -1 placeholder.

    :return: Inferred dimension or None if can't compute.
    """
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
) -> None:  # pragma: no cover
    """Validate that a layer has at least the minimum number of inputs.

    Raises ``ValueError`` with a descriptive message if validation fails.

    :param layer: Semantic layer IR to validate.
    :param min_count: Minimum number of required inputs.
    :param operation_name: Optional operation name for error message
        (defaults to ``layer.name``).

    :raises ValueError: If layer has fewer than ``min_count`` inputs.
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

    :param input_shape: Input tensor shape (may contain symbolic dimensions).
    :param shape_list: Target reshape shape with possible -1 placeholder.

    :return: True if reshape can be statically inferred.
    """
    if not input_shape:
        return False
    return all(isinstance(d, int) for d in input_shape) and all(
        isinstance(d, int) or d == -1 for d in shape_list
    )
