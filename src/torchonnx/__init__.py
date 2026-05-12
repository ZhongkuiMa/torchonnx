"""ONNX-to-PyTorch model conversion framework."""

__docformat__ = "restructuredtext"
__version__ = "2026.5.1"
__all__ = [
    "BENCHMARKS_WITHOUT_BATCH_DIM",
    "TorchONNX",
    "if_has_batch_dim",
]

from torchonnx._torchonnx import TorchONNX
from torchonnx.presets import BENCHMARKS_WITHOUT_BATCH_DIM, if_has_batch_dim
