"""Line-by-line code optimization for Stage 6.

Transforms individual lines of generated code.
"""

__docformat__ = "restructuredtext"
__all__ = ["optimize_line"]

import re

from torchonnx.simplify._rules import (
    FUNCTION_DEFAULTS,
    LAYER_DEFAULTS,
    POSITIONAL_ONLY_ARGS,
)


def optimize_line(line: str) -> str:
    """Optimize a single line of code.

    :param line: Line of code to optimize
    :return: Optimized line
    """
    stripped = line.strip()

    # Skip empty lines, comments, class/def declarations
    if not stripped:
        return line
    if stripped.startswith("#"):
        return line
    if stripped.startswith("class "):
        return line
    if stripped.startswith("def "):
        return line
    if stripped.startswith("super()"):
        return line
    if stripped.startswith("import "):
        return line
    if stripped.startswith("from "):
        return line
    if stripped.startswith("return "):
        return line

    indent = line[: len(line) - len(line.lstrip())]

    # Apply optimizations
    optimized = stripped
    optimized = _optimize_layer_instantiation(optimized)
    optimized = _optimize_function_call(optimized)

    return indent + optimized


def _optimize_layer_instantiation(line: str) -> str:
    """Optimize layer instantiation lines.

    Handles:
    - self.xxx = nn.LayerType(args)

    :param line: Line to optimize
    :return: Optimized line
    """
    # Match pattern: self.xxx = nn.LayerType(args)
    match = re.match(r"(self\.\w+\s*=\s*nn\.)(\w+)\((.*)\)$", line)
    if not match:
        return line

    prefix = match.group(1)  # "self.xxx = nn."
    layer_type = match.group(2)  # "Conv2d", "Flatten", etc.
    args_str = match.group(3)  # "in_channels=3, out_channels=64, ..."

    # Parse arguments
    args = _parse_args(args_str)

    # Apply optimizations
    args = _convert_to_positional(layer_type, args)
    args = _remove_defaults(layer_type, args)

    # Rebuild line
    new_args_str = ", ".join(args)
    return f"{prefix}{layer_type}({new_args_str})"


def _parse_args(args_str: str) -> list[str]:
    """Parse comma-separated arguments, respecting nested parentheses.

    :param args_str: Arguments string
    :return: List of individual arguments
    """
    if not args_str.strip():
        return []

    args = []
    current = ""
    depth = 0

    for char in args_str:
        if char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += char

    if current.strip():
        args.append(current.strip())

    return args


def _convert_to_positional(layer_type: str, args: list[str]) -> list[str]:
    """Convert named arguments to positional where appropriate.

    :param layer_type: Layer type name
    :param args: List of arguments
    :return: Modified arguments list
    """
    if layer_type not in POSITIONAL_ONLY_ARGS:
        return args

    positional_names = POSITIONAL_ONLY_ARGS[layer_type]
    result = []

    for arg in args:
        if "=" in arg:
            name, value = arg.split("=", 1)
            name = name.strip()
            value = value.strip()

            if name in positional_names:
                # Convert to positional
                result.append(value)
            else:
                result.append(arg)
        else:
            result.append(arg)

    return result


def _remove_defaults(layer_type: str, args: list[str]) -> list[str]:
    """Remove arguments that match default values.

    :param layer_type: Layer type name
    :param args: List of arguments
    :return: Modified arguments list
    """
    if layer_type not in LAYER_DEFAULTS:
        return args

    defaults = LAYER_DEFAULTS[layer_type]
    result = []

    for arg in args:
        if "=" in arg:
            name, value = arg.split("=", 1)
            name = name.strip()
            value = value.strip()

            if name in defaults and value == defaults[name]:
                # Skip this argument (it's the default)
                continue

        result.append(arg)

    return result


def _optimize_function_call(line: str) -> str:
    """Optimize function call lines (F.*, torch.*).

    Handles:
    - xxx = F.function(args)
    - xxx = torch.function(args)

    :param line: Line to optimize
    :return: Optimized line
    """
    # Match pattern: xxx = F.function(args) or xxx = torch.function(args)
    match = re.match(r"(.+\s*=\s*)((F|torch)\.\w+)\((.*)\)$", line)
    if not match:
        return line

    prefix = match.group(1)  # "xxx = "
    function_name = match.group(2)  # "F.relu" or "torch.cat"
    args_str = match.group(4)  # "x, inplace=False"

    # Parse arguments
    args = _parse_args(args_str)

    # Remove default arguments
    args = _remove_function_defaults(function_name, args)

    # Rebuild line
    new_args_str = ", ".join(args)
    return f"{prefix}{function_name}({new_args_str})"


def _remove_function_defaults(function_name: str, args: list[str]) -> list[str]:
    """Remove function arguments that match default values.

    :param function_name: Function name (e.g., "F.relu", "torch.cat")
    :param args: List of arguments
    :return: Modified arguments list
    """
    if function_name not in FUNCTION_DEFAULTS:
        return args

    defaults = FUNCTION_DEFAULTS[function_name]
    result = []

    for arg in args:
        if "=" in arg:
            name, value = arg.split("=", 1)
            name = name.strip()
            value = value.strip()

            if name in defaults and value == defaults[name]:
                # Skip this argument (it's the default)
                continue

        result.append(arg)

    return result
