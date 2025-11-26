"""OPERATOR handlers for code generation.

Handlers for simple operators (OPERATOR class).
Generate code like: x2 = x0 + x1, x3 = x1 @ x2, etc.
"""

__docformat__ = "restructuredtext"
__all__ = ["register_operator_handlers"]

from ._registry import register_handler
from ...analyze import (
    SemanticLayerIR,
    VariableInfo,
    ParameterInfo,
    ConstantInfo,
)


def _get_input_code_name(
    inp: VariableInfo | ParameterInfo | ConstantInfo,
    use_literal_for_scalar: bool = False,
    use_literal_for_small_vector: bool = False,
) -> str:
    """Get code name for input (handles parameters and constants with self.).

    :param inp: Input info
    :param use_literal_for_scalar: If True, use literal value for scalar constants
    :param use_literal_for_small_vector: If True, use literal for 1D vectors with ≤10 elements
    :return: Code name string
    """
    if isinstance(inp, ConstantInfo):
        # Check if we should use literal
        should_use_literal = False

        if use_literal_for_scalar and inp.data.numel() == 1:
            # Scalar literal
            should_use_literal = True
        elif (
            use_literal_for_small_vector
            and inp.data.ndim == 1
            and inp.data.numel() <= 10
        ):
            # Small 1D vector literal
            should_use_literal = True

        if should_use_literal:
            if inp.data.numel() == 1:
                # Scalar
                value = inp.data.item()
                if isinstance(value, (int, bool)) or (
                    isinstance(value, float) and value.is_integer()
                ):
                    return str(int(value))
                else:
                    return str(value)
            else:
                # Small 1D vector - use torch.tensor([...])
                values = inp.data.tolist()
                return f"torch.tensor({values})"
        else:
            # Non-literal constant - mark as used
            from .._forward_gen import get_forward_gen_context

            ctx = get_forward_gen_context()
            if ctx:
                ctx.mark_constant_used(inp.code_name)
            return f"self.{inp.code_name}"
    elif isinstance(inp, ParameterInfo):
        # Import here to avoid circular dependency
        from .._forward_gen import get_forward_gen_context

        ctx = get_forward_gen_context()
        if ctx:
            ctx.mark_parameter_used(inp.code_name)
        return f"self.{inp.code_name}"
    else:
        return inp.code_name


def _handle_add(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Add operator (torch.add or +).

    Uses literals for scalar constants and small 1D vectors (≤10 elements).

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    if len(layer.inputs) < 2:
        raise ValueError(f"Add requires 2 inputs, got {len(layer.inputs)}")

    input1 = _get_input_code_name(
        layer.inputs[0], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    input2 = _get_input_code_name(
        layer.inputs[1], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    output = layer.outputs[0].code_name

    # Check if there are arguments (e.g., alpha for scaled add)
    if layer.arguments:
        # Use torch.add with arguments
        args_str = ", ".join(
            f"{arg.pytorch_name}={arg.value}" for arg in layer.arguments
        )
        return f"{output} = torch.add({input1}, {input2}, {args_str})"
    else:
        # Use simple + operator
        return f"{output} = {input1} + {input2}"


def _handle_sub(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Sub operator.

    Uses literals for scalar constants and small 1D vectors (≤10 elements).

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    input1 = _get_input_code_name(
        layer.inputs[0], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    input2 = _get_input_code_name(
        layer.inputs[1], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    output = layer.outputs[0].code_name
    return f"{output} = {input1} - {input2}"


def _handle_mul(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Mul operator.

    Uses literals for scalar constants and small 1D vectors (≤10 elements).

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    input1 = _get_input_code_name(
        layer.inputs[0], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    input2 = _get_input_code_name(
        layer.inputs[1], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    output = layer.outputs[0].code_name
    return f"{output} = {input1} * {input2}"


def _handle_div(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Div operator.

    Uses literals for scalar constants and small 1D vectors (≤10 elements).

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    input1 = _get_input_code_name(
        layer.inputs[0], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    input2 = _get_input_code_name(
        layer.inputs[1], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    output = layer.outputs[0].code_name
    return f"{output} = {input1} / {input2}"


def _handle_matmul(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle MatMul operator.

    Note: MatMul requires tensor operands, cannot use scalar literals.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    input1 = _get_input_code_name(layer.inputs[0], use_literal_for_scalar=False)
    input2 = _get_input_code_name(layer.inputs[1], use_literal_for_scalar=False)
    output = layer.outputs[0].code_name
    return f"{output} = {input1} @ {input2}"


def _handle_pow(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Pow operator.

    Uses literals for scalar constants and small 1D vectors (≤10 elements).

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    input1 = _get_input_code_name(
        layer.inputs[0], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    input2 = _get_input_code_name(
        layer.inputs[1], use_literal_for_scalar=True, use_literal_for_small_vector=True
    )
    output = layer.outputs[0].code_name
    return f"{output} = {input1} ** {input2}"


def _handle_neg(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Neg operator.

    Uses literals for scalar constant operands.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    input1 = _get_input_code_name(layer.inputs[0], use_literal_for_scalar=True)
    output = layer.outputs[0].code_name
    return f"{output} = -{input1}"


def _handle_equal(layer: SemanticLayerIR, layer_name_mapping: dict[str, str]) -> str:
    """Handle Equal operator.

    Uses literals for scalar constant operands.

    :param layer: Semantic layer IR
    :param layer_name_mapping: Mapping from ONNX layer name to clean Python name
    :return: Generated code line
    """
    input1 = _get_input_code_name(layer.inputs[0], use_literal_for_scalar=True)
    input2 = _get_input_code_name(layer.inputs[1], use_literal_for_scalar=True)
    output = layer.outputs[0].code_name
    return f"{output} = {input1} == {input2}"


def register_operator_handlers() -> None:
    """Register all OPERATOR handlers."""
    register_handler("torch.add", _handle_add)
    register_handler("torch.sub", _handle_sub)
    register_handler("torch.mul", _handle_mul)
    register_handler("torch.div", _handle_div)
    register_handler("torch.matmul", _handle_matmul)
    register_handler("torch.pow", _handle_pow)
    register_handler("torch.neg", _handle_neg)
    register_handler("torch.equal", _handle_equal)
