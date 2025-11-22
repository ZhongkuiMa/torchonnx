"""ONNX to PyTorch mapping utilities."""

__docformat__ = "restructuredtext"
__all__ = [
    "infer_pytorch_layer_type",
    "is_parametric_layer",
    "is_functional_operation",
    "get_functional_operator",
    "identify_layer_parameters",
    "map_onnx_to_pytorch_args",
    "is_functional_operation_with_args",
    "get_functional_operation_template",
    "extract_functional_args",
]

from .arguments import map_onnx_to_pytorch_args
from .functional import (
    extract_functional_args,
    get_functional_operation_template,
    is_functional_operation_with_args,
)
from .operators import (
    get_functional_operator,
    infer_pytorch_layer_type,
    is_functional_operation,
    is_parametric_layer,
)
from .parameters import identify_layer_parameters
