"""Stage 2: IR Construction.

This module builds intermediate representation (IR) from normalized ONNX models.
"""

__docformat__ = "restructuredtext"
__all__ = ["ModelIR", "NodeIR", "build_model_ir"]

from torchonnx.build.builder import build_model_ir
from torchonnx.build.types import ModelIR, NodeIR
