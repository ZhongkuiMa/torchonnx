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

from torchonnx.torchonnx.analyze.builder import build_semantic_ir
from torchonnx.torchonnx.analyze.tensor_classifier import classify_inputs, classify_outputs
from torchonnx.torchonnx.analyze.types import (
    ArgumentInfo,
    ConstantInfo,
    OperatorClass,
    ParameterInfo,
    SemanticLayerIR,
    SemanticModelIR,
    VariableInfo,
)
