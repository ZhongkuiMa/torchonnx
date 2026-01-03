__docformat__ = "restructuredtext"
__version__ = "2026.1.0"
__all__ = [
    "BENCHMARKS_WITHOUT_BATCH_DIM",
    "TorchONNX",
    "if_has_batch_dim",
]

from torchonnx._torchonnx import TorchONNX
from torchonnx.presets import BENCHMARKS_WITHOUT_BATCH_DIM, if_has_batch_dim
