"""Generate forward() method from semantic IR.

Main forward generation logic with handler dispatch.
"""

__docformat__ = "restructuredtext"
__all__ = ["ForwardGenContext", "generate_forward_method", "set_forward_gen_context"]

from torchonnx.analyze import SemanticLayerIR, SemanticModelIR
from torchonnx.generate._handlers import (
    HANDLERS,
    get_handler,
    register_layer_handlers,
    register_operation_handlers,
    register_operator_handlers,
)
from torchonnx.generate._templates import FORWARD_TEMPLATE, INDENT


# Global context for tracking used constants during forward generation
class ForwardGenContext:
    """Context for tracking constants and helpers used during forward generation."""

    def __init__(self):
        self.used_constants: set[str] = set()
        self.used_parameters: set[str] = set()
        # Track which helper functions are actually needed in the generated code
        self.needs_dynamic_slice: bool = False
        self.needs_scatter_nd: bool = False
        self.needs_dynamic_expand: bool = False
        # First input name for device inference (e.g., "x0")
        self.first_input_name: str | None = None
        # Vmap mode settings
        self.vmap_mode: bool = True
        # Pre-computed slice lengths for vmap mode: {layer_name: [lengths]}
        # When provided, dynamic_slice uses these instead of computing with .item()
        self.slice_length_hints: dict[str, list[int]] = {}

    def mark_constant_used(self, constant_name: str) -> None:
        """Mark a constant as used in forward method."""
        self.used_constants.add(constant_name)

    def mark_parameter_used(self, parameter_name: str) -> None:
        """Mark a parameter as used in forward method."""
        self.used_parameters.add(parameter_name)

    def get_slice_lengths(self, layer_name: str) -> list[int] | None:
        """Get pre-computed slice lengths for a Slice layer (vmap mode)."""
        return self.slice_length_hints.get(layer_name)


# Global instance - will be set during forward generation
_forward_gen_context: ForwardGenContext | None = None


def get_forward_gen_context() -> ForwardGenContext | None:
    """Get current forward generation context."""
    return _forward_gen_context


def set_forward_gen_context(context: ForwardGenContext | None) -> None:
    """Set forward generation context.

    :param context: Context to set or None to clear
    """
    global _forward_gen_context
    _forward_gen_context = context


def _ensure_handlers_registered() -> None:
    """Ensure all handlers are registered (lazy initialization).

    Registers handlers only once, on first call. This avoids module-level
    side effects while ensuring handlers are available when needed.
    """
    if not HANDLERS:
        register_layer_handlers()
        register_operation_handlers()
        register_operator_handlers()


def _build_io_code_names(semantic_ir: SemanticModelIR) -> tuple[list[str], list[str]]:
    """Build input and output code name mappings from semantic IR.

    :param semantic_ir: Semantic IR from Stage 3
    :return: Tuple of (input_code_names, output_code_names)
    """
    input_code_names: list[str] = []
    for input_name in semantic_ir.input_names:
        # Find the variable with this onnx_name
        var = next(
            (v for v in semantic_ir.variables if v.onnx_name == input_name),
            None,
        )
        if var:
            input_code_names.append(var.code_name)
        else:
            # Fallback (shouldn't happen with valid IR)
            input_code_names.append(input_name)

    output_code_names: list[str] = []
    for output_name in semantic_ir.output_names:
        # Find the variable with this onnx_name
        var = next(
            (v for v in semantic_ir.variables if v.onnx_name == output_name),
            None,
        )
        if var:
            output_code_names.append(var.code_name)
        else:
            # Fallback (shouldn't happen with valid IR)
            output_code_names.append(output_name)

    return input_code_names, output_code_names


def _initialize_forward_context(input_code_names: list[str]) -> None:
    """Initialize forward generation context.

    Sets first input name for device inference in handlers.

    :param input_code_names: List of input variable code names
    """
    global _forward_gen_context

    if _forward_gen_context is None:
        _forward_gen_context = ForwardGenContext()

    if input_code_names:
        _forward_gen_context.first_input_name = input_code_names[0]


def _generate_forward_body(
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str],
) -> list[str]:
    """Generate forward method body lines from all layers.

    Handles vmap mode slice initialization and error handling for each layer.

    :param semantic_ir: Semantic IR from Stage 3
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: List of indented code lines
    """
    global _forward_gen_context

    body_lines: list[str] = []

    # In vmap mode with dynamic slices, initialize validity tracking variable
    if (
        _forward_gen_context
        and _forward_gen_context.vmap_mode
        and _forward_gen_context.needs_dynamic_slice
    ):
        first_input = _forward_gen_context.first_input_name or "x0"
        valid_init = (
            f"_slice_valid = torch.ones((), dtype={first_input}.dtype, device={first_input}.device)"
        )
        body_lines.append(f"{INDENT}{INDENT}{valid_init}")

    # Generate code for each layer
    for layer in semantic_ir.layers:
        try:
            code_line = _generate_layer_code(layer, layer_name_mapping)
            body_lines.append(f"{INDENT}{INDENT}{code_line}")
        except (KeyError, RuntimeError, ValueError, AttributeError) as error:
            # Add comment for failed layer
            body_lines.append(
                f"{INDENT}{INDENT}# ERROR generating {layer.name} ({layer.pytorch_type}): {error}"
            )
            # Add placeholder
            if layer.outputs:
                output_name = layer.outputs[0].code_name
                body_lines.append(f"{INDENT}{INDENT}{output_name} = None  # Placeholder")

    return body_lines


def generate_forward_method(
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str] | None = None,
) -> str:
    """Generate forward() method code.

    Generates code for each layer using registered handlers, then assembles
    into complete forward method.

    :param semantic_ir: Semantic IR from Stage 3
    :param layer_name_mapping: Optional mapping of ONNX name -> clean name
    :return: Generated forward method code
    """
    global _forward_gen_context

    # Ensure handlers are registered
    _ensure_handlers_registered()

    # Use empty mapping if none provided
    if layer_name_mapping is None:
        layer_name_mapping = {}

    # Build input and output code names
    input_code_names, output_code_names = _build_io_code_names(semantic_ir)

    # Initialize forward generation context
    _initialize_forward_context(input_code_names)

    # Generate forward method body
    body_lines = _generate_forward_body(semantic_ir, layer_name_mapping)

    # Assemble forward method
    input_args_str = ", ".join(input_code_names)
    output_return_str = ", ".join(output_code_names) if output_code_names else "None"

    body = "\n".join(body_lines) if body_lines else f"{INDENT}{INDENT}pass"

    forward_code = FORWARD_TEMPLATE.format(
        indent=INDENT,
        input_args=input_args_str,
        body=body,
        output_return=output_return_str,
    )

    # Clean up global context
    _forward_gen_context = None

    return forward_code


def _generate_layer_code(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Generate code for a single layer using registered handler.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    # Get handler for this pytorch_type
    handler = get_handler(layer.pytorch_type)

    if handler:
        # Use registered handler
        return handler(layer, layer_name_mapping)
    # No handler found - generate error comment
    raise ValueError(f"No handler registered for pytorch_type: {layer.pytorch_type}")
