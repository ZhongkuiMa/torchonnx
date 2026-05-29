"""Optimization rules for the simplify stage.

Defines which constructor arguments should be promoted to positional form
and what each argument's PyTorch default is. The default tables are
**derived at import time** from ``inspect.signature(...)`` on the real
classes / functions in ``torch.nn`` and ``torch.nn.functional`` so that
the simplify stage can never drift from the actual PyTorch surface --
when PyTorch bumps a default in a minor release, this module picks it
up on the next import.

Earlier revisions of this file maintained ~200 lines of hand-copied
defaults (``"eps": "1e-05"``, ``"padding_mode": "'zeros'"``, ...) which
were guaranteed to disagree with PyTorch eventually. The R8 multi-master
audit flagged the table as a parallel source of truth that would silently
diverge from the generator's IR-level ``arg.is_default()`` check; this
rewrite collapses it to one source: PyTorch itself.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "FUNCTION_DEFAULTS",
    "LAYER_DEFAULTS",
    "POSITIONAL_ONLY_ARGS",
]

import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  -- F is the idiomatic alias

# ---------------------------------------------------------------------------
# Positional-only argument rules.
#
# These cannot be derived from inspect: PyTorch signatures do not mark which
# arguments are "naturally positional" in idiomatic source. The first N
# arguments below are stripped of their keyword form by the line optimizer
# (e.g. ``nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)`` becomes
# ``nn.Conv2d(3, 16, 3)``).
# ---------------------------------------------------------------------------

POSITIONAL_ONLY_ARGS: dict[str, list[str]] = {
    # Convolution layers
    "Conv1d": ["in_channels", "out_channels", "kernel_size"],
    "Conv2d": ["in_channels", "out_channels", "kernel_size"],
    "Conv3d": ["in_channels", "out_channels", "kernel_size"],
    "ConvTranspose1d": ["in_channels", "out_channels", "kernel_size"],
    "ConvTranspose2d": ["in_channels", "out_channels", "kernel_size"],
    "ConvTranspose3d": ["in_channels", "out_channels", "kernel_size"],
    # Linear layers
    "Linear": ["in_features", "out_features"],
    # Normalization layers
    "BatchNorm1d": ["num_features"],
    "BatchNorm2d": ["num_features"],
    "BatchNorm3d": ["num_features"],
    "LayerNorm": ["normalized_shape"],
    "GroupNorm": ["num_groups", "num_channels"],
    "InstanceNorm1d": ["num_features"],
    "InstanceNorm2d": ["num_features"],
    "InstanceNorm3d": ["num_features"],
    # Pooling layers
    "MaxPool1d": ["kernel_size"],
    "MaxPool2d": ["kernel_size"],
    "MaxPool3d": ["kernel_size"],
    "AvgPool1d": ["kernel_size"],
    "AvgPool2d": ["kernel_size"],
    "AvgPool3d": ["kernel_size"],
    "AdaptiveAvgPool1d": ["output_size"],
    "AdaptiveAvgPool2d": ["output_size"],
    "AdaptiveAvgPool3d": ["output_size"],
    "AdaptiveMaxPool1d": ["output_size"],
    "AdaptiveMaxPool2d": ["output_size"],
    "AdaptiveMaxPool3d": ["output_size"],
    # Activation layers
    "ReLU": [],
    "LeakyReLU": ["negative_slope"],
    "ELU": ["alpha"],
    "PReLU": [],
    "Sigmoid": [],
    "Tanh": [],
    "Softmax": ["dim"],
    "LogSoftmax": ["dim"],
    # Dropout layers
    "Dropout": ["p"],
    "Dropout2d": ["p"],
    "Dropout3d": ["p"],
    # Embedding layers
    "Embedding": ["num_embeddings", "embedding_dim"],
    # Recurrent layers
    "RNN": ["input_size", "hidden_size"],
    "LSTM": ["input_size", "hidden_size"],
    "GRU": ["input_size", "hidden_size"],
    # Shape operations
    "Flatten": ["start_dim"],
    "Unflatten": ["dim", "unflattened_size"],
    # Upsampling
    "Upsample": [],
    "UpsamplingNearest2d": [],
    "UpsamplingBilinear2d": [],
}


def _format_default(value: Any) -> str:
    """Format a PyTorch default value as its Python source representation.

    Mirrors what ``format_argument`` emits in the generator so the simplify
    stage's string comparison succeeds.

    :param value: Default value from ``inspect.Parameter.default``.

    :return: Source-text rendering ('True', "'zeros'", '1e-05', 'None', ...).
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def _derive_defaults(callable_obj: Callable[..., Any]) -> dict[str, str]:
    """Read the default-value table for a callable from its signature.

    Returns an empty dict if the signature cannot be inspected (typically
    a C extension without a proper ``__text_signature__``), so callers
    silently fall back to "do not strip anything" rather than crashing.
    Excludes ``*args`` / ``**kwargs`` and positional-only or self-style
    parameters that have no name to emit.

    :param callable_obj: The class / function to introspect.

    :return: Mapping of keyword-name -> formatted default value.
    """
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {}
    return {
        name: _format_default(p.default)
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


# ---------------------------------------------------------------------------
# nn.* class defaults -- auto-derived from torch.nn signatures.
# ---------------------------------------------------------------------------

_SUPPORTED_LAYERS: tuple[type, ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.Linear,
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.GELU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    nn.LogSoftmax,
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.Upsample,
    nn.Flatten,
)

LAYER_DEFAULTS: dict[str, dict[str, str]] = {
    cls.__name__: _derive_defaults(cls) for cls in _SUPPORTED_LAYERS
}

# ---------------------------------------------------------------------------
# Function defaults -- auto-derived from F.* and torch.* signatures.
# ---------------------------------------------------------------------------

_SUPPORTED_FUNCTIONS: tuple[tuple[str, Callable[..., Any]], ...] = (
    # Activation functions
    ("F.relu", F.relu),
    ("F.leaky_relu", F.leaky_relu),
    ("F.elu", F.elu),
    ("F.gelu", F.gelu),
    ("F.sigmoid", F.sigmoid),
    ("F.tanh", F.tanh),
    ("F.softmax", F.softmax),
    ("F.log_softmax", F.log_softmax),
    # Pooling
    ("F.max_pool2d", F.max_pool2d),
    ("F.avg_pool2d", F.avg_pool2d),
    ("F.adaptive_avg_pool2d", F.adaptive_avg_pool2d),
    # Convolution
    ("F.conv2d", F.conv2d),
    ("F.conv_transpose2d", F.conv_transpose2d),
    # Dropout
    ("F.dropout", F.dropout),
    # Pad
    ("F.pad", F.pad),
    # Torch tensor ops
    ("torch.cat", torch.cat),
    ("torch.concat", torch.concat),
    ("torch.flatten", torch.flatten),
    ("torch.squeeze", torch.squeeze),
    ("torch.unsqueeze", torch.unsqueeze),
    # Torch math
    ("torch.add", torch.add),
    ("torch.sub", torch.sub),
    # Torch creation
    ("torch.zeros", torch.zeros),
    ("torch.ones", torch.ones),
    ("torch.full", torch.full),
    ("torch.arange", torch.arange),
)

FUNCTION_DEFAULTS: dict[str, dict[str, str]] = {
    name: _derive_defaults(fn) for name, fn in _SUPPORTED_FUNCTIONS
}
