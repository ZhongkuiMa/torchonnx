"""Module name sanitization for PyTorch module generation.

This module provides utilities for converting ONNX model names and benchmark names
into valid Python class identifiers.
"""

__docformat__ = "restructuredtext"
__all__ = ["sanitize_module_name"]


def sanitize_module_name(model_name: str, benchmark_name: str | None = None) -> str:
    """Sanitize model name to create valid Python class identifier.

    Always prepends benchmark name if provided to avoid naming conflicts
    and ensure class names don't start with numbers.

    Strategy:
    1. Prepend benchmark_name if provided (format: benchmark_model)
    2. Replace invalid characters (hyphens → underscores)
    3. Convert to CamelCase
    4. Ensure starts with letter (remove leading digits)
    5. Append "_Model" suffix

    Examples:
        >>> sanitize_module_name("vgg16-7", "vggnet16_2023")
        'Vggnet162023Vgg167Model'

        >>> sanitize_module_name("ACASXU_run2a_1_1_batch_2000", "acasxu_2023")
        'Acasxu2023AcasxuRun2a11Batch2000Model'

        >>> sanitize_module_name("my-model")
        'MyModelModel'

    :param model_name: Original model name from ONNX file
    :param benchmark_name: Optional benchmark name to prepend
    :return: Sanitized Python class name
    """
    if benchmark_name:
        name = f"{benchmark_name}_{model_name}"
    else:
        name = model_name

    name = name.replace("-", "_")
    name = name.replace(".", "_")
    name = name.replace("=", "_")

    parts = name.split("_")
    name = "".join(part.capitalize() for part in parts if part)

    while name and name[0].isdigit():
        name = name[1:]

    if not name:
        name = "Model"

    if not name.endswith("Model"):
        name = name + "Model"

    return name
