"""Stage 4: IR Optimization (Pass-Through).

Currently a pass-through stage for future optimizations.
"""

__docformat__ = "restructuredtext"
__all__ = ["optimize_semantic_ir"]

from torchonnx.optimize.optimizer import optimize_semantic_ir
