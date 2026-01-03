"""Handler registry for operation code generation.

Provides dispatcher for operation-specific code generators.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "HANDLERS",
    "Handler",
    "get_handler",
    "register_handler",
]

from collections.abc import Callable

from torchonnx.analyze import SemanticLayerIR

# Handler type: takes SemanticLayerIR and layer_name_mapping, returns code string
Handler = Callable[[SemanticLayerIR, dict[str, str]], str]

# Global handler registry
HANDLERS: dict[str, Handler] = {}


def register_handler(pytorch_type: str, handler: Handler) -> None:
    """Register handler for a PyTorch type.

    :param pytorch_type: PyTorch type string (e.g., "nn.Conv2d", "torch.add")
    :param handler: Handler function
    """
    HANDLERS[pytorch_type] = handler


def get_handler(pytorch_type: str) -> Handler | None:
    """Get handler for PyTorch type.

    :param pytorch_type: PyTorch type string
    :return: Handler function or None if not found
    """
    return HANDLERS.get(pytorch_type)
