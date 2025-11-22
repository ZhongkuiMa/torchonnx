"""Parameter naming utilities for simplified PyTorch parameter names.

This module provides utilities to generate clean, simplified names for
parameters extracted from ONNX initializers.
"""

__docformat__ = "restructuredtext"
__all__ = ["sanitize_parameter_names"]


def infer_parameter_type(initializer_name: str) -> str:
    """Infer parameter type from ONNX initializer name.

    :param initializer_name: ONNX initializer name
    :return: Parameter type ("weight", "bias", or "param")
    """
    name_lower = initializer_name.lower()

    if any(x in name_lower for x in ["weight", "w", "kernel", "matmul"]):
        return "weight"
    elif any(x in name_lower for x in ["bias", "b", "add"]):
        return "bias"
    else:
        return "param"


def sanitize_parameter_names(
    initializer_names: list[str],
) -> dict[str, str]:
    """Generate simplified parameter names for ONNX initializers.

    Maps ONNX initializer names to clean PyTorch parameter names.

    Example:
        {
            'Operation_1_MatMul_W': 'weight1',
            'Operation_1_Add_B': 'bias1',
            'Operation_2_MatMul_W': 'weight2',
            'Operation_2_Add_B': 'bias2',
            'input_AvgImg': 'param1',
        }

    :param initializer_names: List of ONNX initializer names
    :return: Dictionary mapping ONNX names to simplified names
    """
    name_mapping: dict[str, str] = {}
    counters: dict[str, int] = {"weight": 0, "bias": 0, "param": 0}

    for onnx_name in initializer_names:
        param_type = infer_parameter_type(onnx_name)

        counters[param_type] += 1
        simplified_name = f"{param_type}{counters[param_type]}"

        name_mapping[onnx_name] = simplified_name

    return name_mapping
