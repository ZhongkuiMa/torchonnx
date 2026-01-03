"""Stage 6: Code-level optimization and formatting.

Post-processes generated PyTorch code to:
- Remove default arguments from layer constructors
- Convert named arguments to positional where appropriate
- Inline simple operators (conservative)
- Add file headers with metadata
- Apply Black-compatible formatting
"""

__docformat__ = "restructuredtext"
__all__ = ["add_file_header", "format_code", "optimize_generated_code"]

from torchonnx.simplify._decorations import add_file_header
from torchonnx.simplify._formatter import format_code
from torchonnx.simplify._optimizer import optimize_generated_code
