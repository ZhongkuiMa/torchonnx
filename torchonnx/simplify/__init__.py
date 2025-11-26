"""Stage 6: Code-level optimization.

Post-processes generated PyTorch code to:
- Remove default arguments from layer constructors
- Convert named arguments to positional where appropriate
- Inline simple operators (conservative)
"""

__docformat__ = "restructuredtext"
__all__ = ["optimize_generated_code"]

from ._optimizer import optimize_generated_code
