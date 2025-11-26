"""Stage 3: Semantic Analysis.

This module performs semantic analysis on raw IR to classify tensors and operators.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "OperatorClass",
    "VariableInfo",
    "ParameterInfo",
    "ConstantInfo",
    "ArgumentInfo",
    "SemanticLayerIR",
    "SemanticModelIR",
    "build_semantic_ir",
    "classify_inputs",
    "classify_outputs",
]

from .builder import build_semantic_ir
from .tensor_classifier import classify_inputs, classify_outputs
from .types import (
    OperatorClass,
    VariableInfo,
    ParameterInfo,
    ConstantInfo,
    ArgumentInfo,
    SemanticLayerIR,
    SemanticModelIR,
)
