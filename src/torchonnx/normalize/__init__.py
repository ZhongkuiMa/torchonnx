"""Stage 1: ONNX Model Normalization.

This module normalizes ONNX models to a canonical form suitable for IR construction.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "extract_onnx_opset_version",
    "get_onnx_initializers",
    "get_onnx_model_input_names",
    "get_onnx_model_output_names",
    "get_onnx_model_shapes",
    "get_onnx_nodes",
    "load_and_preprocess_onnx_model",
]

from torchonnx.normalize.normalize import load_and_preprocess_onnx_model
from torchonnx.normalize.utils import (
    extract_onnx_opset_version,
    get_onnx_initializers,
    get_onnx_model_input_names,
    get_onnx_model_output_names,
    get_onnx_model_shapes,
    get_onnx_nodes,
)
