"""Stage 6: Code-level optimization and formatting.

Post-processes generated PyTorch code to:
- Remove default arguments from layer constructors
- Convert named arguments to positional where appropriate
- Inline simple operators (conservative)
- Add file headers with metadata
- Apply Black-compatible formatting
"""

__docformat__ = "restructuredtext"
__all__ = ["optimize_generated_code", "add_file_header", "format_code"]

from ._optimizer import optimize_generated_code
from ._decorations import add_file_header
from ._formatter import format_code
