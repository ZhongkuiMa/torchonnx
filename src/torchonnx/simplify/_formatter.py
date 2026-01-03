"""Format generated code to follow Black rules."""

__docformat__ = "restructuredtext"
__all__ = ["format_code"]

import re

MAX_LINE_LENGTH = 88


def format_code(code: str) -> str:
    """Apply Black-compatible formatting to generated code.

    Applies the following formatting rules:
    - Max line length of 88 characters
    - Two blank lines before class/function definitions at module level
    - One blank line between methods in a class
    - Trailing commas in multi-line function calls

    :param code: Generated PyTorch code
    :return: Formatted code
    """
    # Pass 1: Normalize blank lines first
    code = _normalize_blank_lines(code)

    # Pass 2: Wrap long lines
    lines = code.split("\n")
    result = []

    for line in lines:
        if len(line) > MAX_LINE_LENGTH:
            formatted = _wrap_long_line(line)
            result.extend(formatted)
        else:
            result.append(line)

    return "\n".join(result)


def _wrap_long_line(line: str) -> list[str]:
    """Wrap a long line into multiple lines with proper indentation.

    Handles patterns like:
    - self.layer = nn.Conv2d(arg1, arg2, arg3, ...)
    - x = func(arg1, arg2, arg3, ...)

    :param line: Line to wrap
    :return: List of wrapped lines
    """
    # Get leading indentation
    indent_match = re.match(r"^(\s*)", line)
    base_indent = indent_match.group(1) if indent_match else ""
    continuation_indent = base_indent + "    "

    # Check if this is an assignment with a function/constructor call
    # Pattern: indent + var = something(args)
    assignment_match = re.match(r"^(\s*)(\S+)\s*=\s*(.+?)\((.*)\)$", line, re.DOTALL)

    if assignment_match:
        indent = assignment_match.group(1)
        var_name = assignment_match.group(2)
        func_call = assignment_match.group(3)
        args_str = assignment_match.group(4)

        # Split arguments by comma (careful with nested parens)
        args = _split_args(args_str)

        if len(args) > 1:
            # Check if wrapping would help
            first_line = f"{indent}{var_name} = {func_call}("
            # Format as multi-line
            result = [first_line]
            arg_indent = continuation_indent
            for _i, arg in enumerate(args):
                arg = arg.strip()
                comma = ","  # Always trailing comma for Black
                arg_line = f"{arg_indent}{arg}{comma}"
                result.append(arg_line)
            result.append(f"{indent})")
            return result

    # If we can't parse it or wrapping won't help, return as-is
    return [line]


def _split_args(args_str: str) -> list[str]:
    """Split arguments by comma, respecting nested parentheses and brackets.

    :param args_str: Arguments string (without outer parentheses)
    :return: List of argument strings
    """
    args = []
    current = []
    depth = 0

    for char in args_str:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        args.append("".join(current).strip())

    return args


def _normalize_blank_lines(code: str) -> str:
    """Ensure proper blank lines between definitions.

    - Two blank lines before class/function definitions at module level
    - One blank line between methods in a class

    :param code: Code to normalize
    :return: Code with normalized blank lines
    """
    lines = code.split("\n")
    result: list[str] = []
    in_class = False

    prev_was_class_def = False

    for _i, line in enumerate(lines):
        stripped = line.strip()

        # Detect entering a class
        if re.match(r"^class\s+\w+", stripped):
            in_class = True

            # Ensure 2 blank lines before class at module level
            _ensure_blank_lines_before(result, 2)
            result.append(line)
            prev_was_class_def = True
            continue

        # Detect module-level function (def at column 0)
        if re.match(r"^def\s+\w+", stripped) and not line.startswith(" "):
            in_class = False
            # Ensure 2 blank lines before module-level function
            _ensure_blank_lines_before(result, 2)
            result.append(line)
            prev_was_class_def = False
            continue

        # Detect method inside class (def with indentation)
        if re.match(r"^\s+def\s+\w+", line) and in_class:
            # First method right after class def - no blank line
            # Subsequent methods - 1 blank line before
            if not prev_was_class_def:
                _ensure_blank_lines_before(result, 1)
            result.append(line)
            prev_was_class_def = False
            continue

        # Skip blank lines immediately after class definition
        if prev_was_class_def and stripped == "":
            continue

        result.append(line)
        prev_was_class_def = False

    return "\n".join(result)


def _ensure_blank_lines_before(lines: list[str], count: int) -> None:
    """Ensure exactly `count` blank lines at end of lines list.

    Modifies the list in place.

    :param lines: List of lines to modify
    :param count: Number of blank lines to ensure
    """
    if not lines:
        return

    # Count existing trailing blank lines
    existing_blanks = 0
    for line in reversed(lines):
        if line.strip() == "":
            existing_blanks += 1
        else:
            break

    # Add or remove blank lines as needed
    if existing_blanks < count:
        # Add blank lines
        lines.extend([""] * (count - existing_blanks))
    elif existing_blanks > count:
        # Remove excess blank lines
        for _ in range(existing_blanks - count):
            lines.pop()
