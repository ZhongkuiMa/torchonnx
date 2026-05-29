"""ONNX-to-PyTorch model conversion framework."""

__docformat__ = "restructuredtext"
__version__ = "2026.5.4"
__all__ = [
    "BENCHMARKS_WITHOUT_BATCH_DIM",
    "TorchONNX",
    "has_batch_dim",
]

from torchonnx._torchonnx import TorchONNX
from torchonnx.presets import BENCHMARKS_WITHOUT_BATCH_DIM, has_batch_dim
