"""Main Stage 5 code optimizer.

Orchestrates line-by-line and whole-code optimizations on generated PyTorch code.
"""

__docformat__ = "restructuredtext"
__all__ = ["optimize_generated_code"]

import re
from typing import overload

from torch import Tensor

from torchonnx.simplify._line_optimizer import optimize_line


@overload
def optimize_generated_code(
    code: str, state_dict: dict[str, Tensor], enable: bool = ...
) -> tuple[str, dict[str, Tensor]]: ...
@overload
def optimize_generated_code(code: str, state_dict: None = ..., enable: bool = ...) -> str: ...
def optimize_generated_code(
    code: str, state_dict: dict[str, Tensor] | None = None, enable: bool = True
) -> str | tuple[str, dict[str, Tensor]]:
    """Apply code-level optimizations to generated PyTorch code.

    Overloads encode the discrimination so callers do not need an
    ``assert isinstance(result, tuple)`` to placate the type checker:
    a ``state_dict`` argument forces the tuple return, ``state_dict=None``
    (the default) returns the bare code string.

    Optimizations include:
    - Converting named arguments to positional where appropriate
    - Removing default arguments from layer constructors and functions
    - Removing unused buffer registrations

    :param code: Generated Python code from Stage 4 (the generate stage).
    :param state_dict: Optional state dict to filter based on removed buffers.
    :param enable: If False, return code unchanged.

    :return: Optimized code, or (optimized_code, filtered_state_dict) if
        ``state_dict`` is provided.
    """
    if not enable:
        return (code, state_dict) if state_dict is not None else code

    # Pass 1: Line-by-line optimizations
    lines = code.split("\n")
    optimized_lines = []

    for line in lines:
        optimized_line = optimize_line(line)
        optimized_lines.append(optimized_line)

    code = "\n".join(optimized_lines)

    # Pass 2: Remove unused buffer registrations
    code, removed_buffers = _remove_unused_buffers(code)

    # Filter state_dict if provided
    if state_dict is not None:
        filtered_state_dict = _filter_state_dict(code, state_dict, removed_buffers)
        return code, filtered_state_dict

    return code


def _filter_state_dict(
    code: str, state_dict: dict[str, Tensor], removed_buffers: set[str]
) -> dict[str, Tensor]:
    """Keep only state_dict entries the generated module actually owns.

    A module's ``load_state_dict`` (strict) rejects keys the class does
    not declare. Two sources of orphan keys exist:
    - buffers registered then pruned as unused (``removed_buffers``);
    - constants the generator inlined as literals (e.g. a Reshape shape
      emitted as ``x.reshape(-1, 784)``) yet still exported to the
      ``.pth``. These never get a ``register_buffer`` line, so the
      unused-buffer pass cannot see them.

    Keep a key when its top-level segment is referenced as ``self.<seg>``
    in the code: submodule params (``linear1.weight`` -> ``self.linear1``)
    and live buffers (``c5`` -> ``self.c5``) survive; orphan constants
    (``c0`` with no ``self.c0``) are dropped.

    :param code: Optimized generated module source.
    :param state_dict: Raw state dict from code generation.
    :param removed_buffers: Buffer names pruned by the unused-buffer pass.
    :return: State dict restricted to keys the module declares.
    """
    used_names = set(re.findall(r"\bself\.([a-zA-Z_]\w*)\b", code))
    return {
        key: value
        for key, value in state_dict.items()
        if key not in removed_buffers and key.split(".", 1)[0] in used_names
    }


def _remove_unused_buffers(code: str) -> tuple[str, set[str]]:
    """Remove buffer registrations that are never used in forward method.

    Parses __init__ to find all register_buffer() calls, parses forward()
    to find buffer usage, and removes unused registrations.

    :param code: Generated code.

    :return: Tuple of (code with unused buffers removed, set of removed buffer names)
    """
    lines = code.split("\n")

    # Extract forward method
    forward_start = None
    forward_end = None
    for i, line in enumerate(lines):
        if "def forward(" in line:
            forward_start = i
        elif forward_start is not None and re.match(r"\s{4}def ", line):
            # Found another method at class level
            forward_end = i
            break

    if forward_start is None:
        # No forward method found, return unchanged
        return code, set()

    if forward_end is None:
        forward_end = len(lines)

    forward_code = "\n".join(lines[forward_start:forward_end])

    # Find all references to self.XXX in forward method
    buffer_pattern = r"\bself\.([a-zA-Z_]\w*)\b"
    used_buffers = set(re.findall(buffer_pattern, forward_code))

    # Remove unused register_buffer() calls from __init__
    result_lines = []
    removed_buffers = set()

    for line in lines:
        # Check if this is a register_buffer call
        if "self.register_buffer(" in line:
            # Extract buffer name from: self.register_buffer("c20", ...)
            match = re.search(r"self\.register_buffer\([\"\'](\w+)[\"\']", line)
            if match:
                buffer_name = match.group(1)
                if buffer_name not in used_buffers:
                    # Skip this line - buffer is unused
                    removed_buffers.add(buffer_name)
                    continue

        result_lines.append(line)

    return "\n".join(result_lines), removed_buffers
