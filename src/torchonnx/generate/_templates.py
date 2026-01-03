"""Code templates and constants for PyTorch code generation.

This module provides string templates and constants used throughout
the code generation process.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "BUFFER_PREFIX",
    "FORWARD_TEMPLATE",
    "INDENT",
    "INIT_TEMPLATE",
    "MODULE_TEMPLATE",
    "PARAM_PREFIX",
    "VAR_PREFIX",
]

# Naming constants
INDENT = "    "
VAR_PREFIX = "x"
PARAM_PREFIX = "p"
BUFFER_PREFIX = "c"

# Module template
MODULE_TEMPLATE = """\
{imports}
{helpers}

class {class_name}(nn.Module):
{init_method}

{forward_method}
"""

# __init__ method template
INIT_TEMPLATE = """\
{indent}def __init__(self):
{indent}{indent}super().__init__()
{body}
"""

# forward() method template
FORWARD_TEMPLATE = """\
{indent}def forward(self, {input_args}):
{body}
{indent}{indent}return {output_return}
"""
