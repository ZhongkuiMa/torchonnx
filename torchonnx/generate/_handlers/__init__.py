"""Operation handlers for code generation.

Handler registry and operation-specific code generators.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "register_layer_handlers",
    "register_operation_handlers",
    "register_operator_handlers",
    "get_handler",
    "register_handler",
    "HANDLERS",
]

from ._layers import register_layer_handlers
from ._operations import register_operation_handlers
from ._operators import register_operator_handlers
from ._registry import get_handler, register_handler, HANDLERS
