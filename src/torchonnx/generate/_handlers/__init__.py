"""Operation handlers for code generation.

Handler registry and operation-specific code generators.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "HANDLERS",
    "get_handler",
    "register_handler",
    "register_layer_handlers",
    "register_operation_handlers",
    "register_operator_handlers",
]

from torchonnx.generate._handlers._layers import register_layer_handlers
from torchonnx.generate._handlers._operations import register_operation_handlers
from torchonnx.generate._handlers._operators import register_operator_handlers
from torchonnx.generate._handlers._registry import HANDLERS, get_handler, register_handler
