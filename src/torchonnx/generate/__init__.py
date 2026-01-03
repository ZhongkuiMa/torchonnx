"""Stage 5: PyTorch Code Generation.

Generates complete PyTorch nn.Module from semantic IR.
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_pytorch_module", "to_camel_case"]

from torchonnx.generate._utils import to_camel_case
from torchonnx.generate.code_generator import generate_pytorch_module
