"""Stage 3: Semantic Analysis.

This module performs semantic analysis on raw IR to classify tensors and operators.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ArgumentInfo",
    "ConstantInfo",
    "OperatorClass",
    "ParameterInfo",
    "SemanticLayerIR",
    "SemanticModelIR",
    "VariableInfo",
    "build_semantic_ir",
    "classify_inputs",
    "classify_outputs",
]

from torchonnx.analyze.builder import build_semantic_ir
from torchonnx.analyze.tensor_classifier import classify_inputs, classify_outputs
from torchonnx.analyze.types import (
    ArgumentInfo,
    ConstantInfo,
    OperatorClass,
    ParameterInfo,
    SemanticLayerIR,
    SemanticModelIR,
    VariableInfo,
)
