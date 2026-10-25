"""ONNX-to-PyTorch model conversion framework.

.. seealso:: ``torchonnx/CONVENTIONS.md`` for code conventions.
"""

__docformat__ = "restructuredtext"
__version__ = "2026.10.4"
__all__ = [
    "BENCHMARKS_WITHOUT_BATCH_DIM",
    "ONNXModelInterface",
    "ONNXTensorInterface",
    "TorchONNX",
    "has_batch_dim",
    "inspect_model_interface",
]

from torchonnx._torchonnx import TorchONNX
from torchonnx.interface import ONNXModelInterface, ONNXTensorInterface, inspect_model_interface
from torchonnx.presets import BENCHMARKS_WITHOUT_BATCH_DIM, has_batch_dim
