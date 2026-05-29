"""Generate forward() method from semantic IR.

Main forward generation logic with handler dispatch. The
``ForwardGenContext`` type and the module-singleton accessors now live in
``_context.py`` so per-op handler files can depend on the context without
re-introducing a circular import; this module re-exports them so existing
call sites continue to work.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ForwardGenContext",
    "generate_forward_method",
    "get_forward_gen_context",
    "set_forward_gen_context",
]

from torchonnx.analyze import SemanticLayerIR, SemanticModelIR
from torchonnx.generate._context import (
    ForwardGenContext,
    _get_ctx,  # noqa: F401  -- re-exported for tests/handlers that still import via this module
    get_forward_gen_context,
    set_forward_gen_context,
)
from torchonnx.generate._handlers import get_handler
from torchonnx.generate._templates import FORWARD_TEMPLATE, INDENT


def _build_io_code_names(semantic_ir: SemanticModelIR) -> tuple[list[str], list[str]]:
    """Build input and output code name mappings from semantic IR.

    :param semantic_ir: Semantic IR from Stage 3.

    :return: Tuple of (input_code_names, output_code_names).
    """
    input_code_names: list[str] = []
    for input_name in semantic_ir.input_names:
        var = next(
            (v for v in semantic_ir.variables if v.onnx_name == input_name),
            None,
        )
        if var:
            input_code_names.append(var.code_name)
        else:
            # Fallback (shouldn't happen with valid IR).
            input_code_names.append(input_name)

    output_code_names: list[str] = []
    for output_name in semantic_ir.output_names:
        var = next(
            (v for v in semantic_ir.variables if v.onnx_name == output_name),
            None,
        )
        if var:
            output_code_names.append(var.code_name)
        else:
            # Fallback (shouldn't happen with valid IR).
            output_code_names.append(output_name)

    return input_code_names, output_code_names


def _initialize_forward_context(input_code_names: list[str]) -> None:
    """Initialize forward generation context.

    Sets first input name for device inference in handlers.

    :param input_code_names: List of input variable code names.
    """
    ctx = get_forward_gen_context()
    if ctx is None:
        ctx = ForwardGenContext()
        set_forward_gen_context(ctx)
    if input_code_names:
        ctx.first_input_name = input_code_names[0]


def _generate_forward_body(
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str],
) -> list[str]:
    """Generate forward method body lines from all layers.

    Fails fast on handler errors: a partial conversion silently emits
    ``None`` placeholders for the failed layer, which then poisons every
    downstream consumer at inference time. We raise here so the caller
    learns about the unsupported op at compile time rather than the first
    forward pass on a benchmark sweep.

    :param semantic_ir: Semantic IR from Stage 3.
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name.

    :return: List of indented code lines.
    :raises RuntimeError: If any layer's handler fails.
    """
    body_lines: list[str] = []

    for layer in semantic_ir.layers:
        try:
            code_line = _generate_layer_code(layer, layer_name_mapping)
        except (KeyError, RuntimeError, ValueError, AttributeError) as error:
            raise RuntimeError(
                f"Failed to generate code for layer {layer.name!r} ({layer.pytorch_type}): {error}"
            ) from error
        body_lines.append(f"{INDENT}{INDENT}{code_line}")

    return body_lines


def generate_forward_method(
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str] | None = None,
) -> str:
    """Generate forward() method code.

    Generates code for each layer using registered handlers, then assembles
    into the complete forward method.

    :param semantic_ir: Semantic IR from Stage 3.
    :param layer_name_mapping: Optional mapping of ONNX name -> clean name.

    :return: Generated forward method code.
    """
    if layer_name_mapping is None:
        layer_name_mapping = {}

    input_code_names, output_code_names = _build_io_code_names(semantic_ir)

    _initialize_forward_context(input_code_names)

    body_lines = _generate_forward_body(semantic_ir, layer_name_mapping)

    input_args_str = ", ".join(input_code_names)
    output_return_str = ", ".join(output_code_names) if output_code_names else "None"

    body = "\n".join(body_lines) if body_lines else f"{INDENT}{INDENT}pass"

    forward_code = FORWARD_TEMPLATE.format(
        indent=INDENT,
        input_args=input_args_str,
        body=body,
        output_return=output_return_str,
    )

    # Clean up global context so a subsequent generation starts fresh.
    set_forward_gen_context(None)

    return forward_code


def _generate_layer_code(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Generate code for a single layer using the registered handler.

    :param layer: Semantic layer IR.
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name.

    :return: Generated code line.
    :raises ValueError: If no handler is registered for the layer's pytorch_type.
    """
    handler = get_handler(layer.pytorch_type)
    if handler:
        return handler(layer, layer_name_mapping)
    raise ValueError(f"No handler registered for pytorch_type: {layer.pytorch_type}")
