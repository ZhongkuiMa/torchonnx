"""Utility functions for code generation.

Helper functions for formatting tensors, arguments, and names.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "format_argument",
    "format_tensor_shape",
    "sanitize_identifier",
    "sanitize_layer_name",
    "to_camel_case",
]

import keyword
from typing import Any


def format_tensor_shape(shape: tuple[int, ...] | None) -> str:
    """Format tensor shape as Python tuple string.

    :param shape: Tensor shape tuple or None
    :return: Formatted shape string (e.g., "(1, 3, 224, 224)")
    """
    if shape is None:
        return "None"
    return f"({', '.join(map(str, shape))})"


def format_argument(value: Any) -> str:
    """Format argument value as valid Python literal.

    Handles common types: None, bool, int, float, str, list, tuple.

    :param value: Argument value to format
    :return: Python literal string suitable for code generation
    :raises TypeError: If value type is not supported
    """
    if value is None:
        return "None"
    # Handle bool before int (bool is subclass of int)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]" if isinstance(value, list) else "()"
        formatted = ", ".join(format_argument(v) for v in value)
        if isinstance(value, list):
            return f"[{formatted}]"
        if len(value) == 1:
            return f"({formatted},)"
        return f"({formatted})"
    # Fail fast with clear error instead of generating potentially invalid code
    raise TypeError(
        f"Cannot format argument of type {type(value).__name__}: {value!r}. "
        f"Supported types: None, bool, int, float, str, list, tuple"
    )


def sanitize_identifier(name: str) -> str:
    """Sanitize string to be a valid Python identifier.

    Replaces invalid characters with underscores and ensures
    the result is a valid Python identifier.

    :param name: String to sanitize
    :return: Valid Python identifier
    """
    if not name:
        return "_"

    # Replace invalid characters with underscores
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)

    # Remove leading digits (Python identifiers can't start with digits)
    while sanitized and sanitized[0].isdigit():
        sanitized = sanitized[1:]

    # If all characters were digits, use fallback
    if not sanitized:
        sanitized = "Model"

    # Ensure it's not a Python keyword
    if keyword.iskeyword(sanitized):
        sanitized = sanitized + "_"

    return sanitized


def sanitize_layer_name(name: str) -> str:
    """Sanitize ONNX layer name to a clean Python identifier.

    Removes underscores and non-alphanumeric characters, keeping only
    letters and digits. Example: "relu_1" -> "relu1"

    :param name: ONNX layer name
    :return: Clean Python identifier without underscores
    """
    if not name:
        return "layer"

    # Remove all non-alphanumeric characters (including underscores)
    sanitized = "".join(c for c in name if c.isalnum())

    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "layer" + sanitized

    # Ensure it's not empty
    if not sanitized:
        sanitized = "layer"

    # Ensure it's not a Python keyword
    if keyword.iskeyword(sanitized):
        sanitized = sanitized + "Layer"

    return sanitized


def to_camel_case(name: str) -> str:
    """Convert name to CamelCase, removing illegal characters.

    Examples:
        "acasxu_2023_ACASXU_run2a_1_1_batch_2000" -> "Acasxu2023AcasxuRun2a11Batch2000"
        "patch-1" -> "Patch1"
        "vgg16-7" -> "Vgg167"

    :param name: Original name
    :return: CamelCase name
    """
    if not name:
        return "Model"

    # Replace non-alphanumeric with spaces for splitting
    cleaned = "".join(c if c.isalnum() else " " for c in name)

    # Split on spaces and capitalize each word
    words = cleaned.split()
    camel = "".join(word.capitalize() for word in words if word)

    # Ensure it starts with uppercase letter (not digit)
    if not camel:
        return "Model"

    # Remove leading digits
    while camel and camel[0].isdigit():
        camel = camel[1:]

    if not camel:
        return "Model"

    if not camel[0].isupper():
        camel = camel[0].upper() + camel[1:]

    # Ensure it's not a Python keyword
    if keyword.iskeyword(camel):
        camel = camel + "Model"

    return camel
