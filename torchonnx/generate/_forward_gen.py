"""Generate forward() method from semantic IR.

Main forward generation logic with handler dispatch.
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_forward_method", "ForwardGenContext"]

from ._handlers import (
    get_handler,
    register_layer_handlers,
    register_operation_handlers,
    register_operator_handlers,
    HANDLERS,
)
from ._templates import INDENT, FORWARD_TEMPLATE
from ..analyze import (
    SemanticModelIR,
    SemanticLayerIR,
)


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

    def mark_constant_used(self, constant_name: str) -> None:
        """Mark a constant as used in forward method."""
        self.used_constants.add(constant_name)

    def mark_parameter_used(self, parameter_name: str) -> None:
        """Mark a parameter as used in forward method."""
        self.used_parameters.add(parameter_name)


# Global instance - will be set during forward generation
_forward_gen_context: ForwardGenContext | None = None


def get_forward_gen_context() -> ForwardGenContext | None:
    """Get current forward generation context."""
    return _forward_gen_context


def _ensure_handlers_registered() -> None:
    """Ensure all handlers are registered (lazy initialization).

    Registers handlers only once, on first call. This avoids module-level
    side effects while ensuring handlers are available when needed.
    """
    if not HANDLERS:
        register_layer_handlers()
        register_operation_handlers()
        register_operator_handlers()


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

    # Create context to track used constants
    _forward_gen_context = ForwardGenContext()

    # Build mapping from onnx_name to code_name for inputs
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

    # Build mapping from onnx_name to code_name for outputs
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

    # Generate code for each layer
    body_lines: list[str] = []

    for layer in semantic_ir.layers:
        try:
            code_line = _generate_layer_code(layer, layer_name_mapping)
            body_lines.append(f"{INDENT}{INDENT}{code_line}")
        except Exception as error:
            # Add comment for failed layer
            body_lines.append(
                f"{INDENT}{INDENT}# ERROR generating {layer.name} "
                f"({layer.pytorch_type}): {error}"
            )
            # Add placeholder
            if layer.outputs:
                output_name = layer.outputs[0].code_name
                body_lines.append(
                    f"{INDENT}{INDENT}{output_name} = None  # Placeholder"
                )

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


def _generate_layer_code(
    layer: SemanticLayerIR, layer_name_mapping: dict[str, str]
) -> str:
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
    else:
        # No handler found - generate error comment
        raise ValueError(
            f"No handler registered for pytorch_type: {layer.pytorch_type}"
        )
