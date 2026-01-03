"""Stage 2: IR Construction.

This module builds intermediate representation (IR) from normalized ONNX models.
"""

__docformat__ = "restructuredtext"
__all__ = ["NodeIR", "ModelIR", "build_model_ir"]

from .builder import build_model_ir
from .types import ModelIR, NodeIR
