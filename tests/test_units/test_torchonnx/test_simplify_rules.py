"""Smoke tests for the simplify-stage auto-derived default tables.

These tables are read from ``inspect.signature`` on the real torch.nn /
torch.nn.functional surface at import time. If PyTorch ever drops or
renames the keys the simplify stage depends on, these tests fail loudly
instead of letting the converter silently emit non-canonical code.
"""

__docformat__ = "restructuredtext"

import pytest

from torchonnx.simplify._rules import (
    FUNCTION_DEFAULTS,
    LAYER_DEFAULTS,
    POSITIONAL_ONLY_ARGS,
)


class TestLayerDefaultsAutoDerivation:
    """LAYER_DEFAULTS is derived from torch.nn at import time."""

    @pytest.mark.parametrize(
        ("layer", "arg", "expected"),
        [
            ("Conv2d", "stride", "1"),
            ("Conv2d", "padding", "0"),
            ("Conv2d", "bias", "True"),
            ("Conv2d", "padding_mode", "'zeros'"),
            ("BatchNorm2d", "eps", "1e-05"),
            ("BatchNorm2d", "momentum", "0.1"),
            ("BatchNorm2d", "affine", "True"),
            ("Linear", "bias", "True"),
            ("ReLU", "inplace", "False"),
            ("LeakyReLU", "negative_slope", "0.01"),
            ("Dropout", "p", "0.5"),
            ("Flatten", "start_dim", "1"),
            ("Flatten", "end_dim", "-1"),
        ],
    )
    def test_key_defaults_match_pytorch(self, layer, arg, expected):
        """Spot-check the contract for the most-used layer defaults."""
        assert LAYER_DEFAULTS[layer][arg] == expected

    def test_unknown_layer_returns_no_defaults_for_simplify(self):
        """LAYER_DEFAULTS.get on an unsupported layer must yield an empty dict.

        The line optimizer relies on ``.get(layer_type, {})`` returning an
        empty mapping so it strips nothing for layers we have not opted in.
        """
        assert LAYER_DEFAULTS.get("ThisLayerDoesNotExist", {}) == {}


class TestFunctionDefaultsAutoDerivation:
    """FUNCTION_DEFAULTS is derived from F.* and torch.* at import time."""

    @pytest.mark.parametrize(
        ("function", "arg", "expected"),
        [
            ("F.relu", "inplace", "False"),
            ("F.leaky_relu", "negative_slope", "0.01"),
            ("F.dropout", "p", "0.5"),
            ("F.dropout", "training", "True"),
        ],
    )
    def test_key_defaults_match_pytorch(self, function, arg, expected):
        assert FUNCTION_DEFAULTS[function][arg] == expected

    def test_c_extension_fns_silently_yield_empty(self):
        """torch.* C-ext signatures may be opaque to ``inspect``.

        When a function's signature cannot be introspected we want a
        no-strip fallback rather than a crash. The simplify stage tolerates
        missing entries via ``.get(fn, {})``.
        """
        # torch.cat is a C builtin; depending on PyTorch version inspect
        # may or may not see it. Either outcome is acceptable: either we
        # get the {"dim": "0"} mapping, or we get {} (no-strip fallback).
        cat_defaults = FUNCTION_DEFAULTS["torch.cat"]
        assert isinstance(cat_defaults, dict)


class TestPositionalOnlyArgs:
    """POSITIONAL_ONLY_ARGS is hand-maintained -- spot check key layers."""

    @pytest.mark.parametrize(
        ("layer", "expected"),
        [
            ("Conv2d", ["in_channels", "out_channels", "kernel_size"]),
            ("Linear", ["in_features", "out_features"]),
            ("BatchNorm2d", ["num_features"]),
            ("MaxPool2d", ["kernel_size"]),
            ("ReLU", []),
        ],
    )
    def test_positional_args(self, layer, expected):
        assert POSITIONAL_ONLY_ARGS[layer] == expected
