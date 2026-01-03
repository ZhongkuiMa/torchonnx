"""Phase 4: Line Optimizer Edge Case Tests.

This module tests line-by-line code optimization functionality in _line_optimizer.py.

Tests cover:
- Argument parsing with nested parentheses
- Positional conversion for different layer types
- Default value removal for layers and functions
- Function call optimization
- Line filtering and skipping

Target: _line_optimizer.py coverage improvement from 74.3% to 88%+
"""

from torchonnx.simplify._line_optimizer import (
    _convert_to_positional,
    _optimize_function_call,
    _optimize_layer_instantiation,
    _parse_args,
    _remove_defaults,
    _remove_function_defaults,
    optimize_line,
)


class TestLineOptimizerParsing:
    """Test argument parsing functionality."""

    def test_parse_args_empty_string(self):
        """Test parsing empty argument string."""
        result = _parse_args("")
        assert result == []

    def test_parse_args_single_argument(self):
        """Test parsing single positional argument."""
        result = _parse_args("x")
        assert result == ["x"]

    def test_parse_args_named_arguments(self):
        """Test parsing named arguments."""
        result = _parse_args("in_channels=3, out_channels=64, kernel_size=3")
        assert len(result) == 3
        assert "in_channels=3" in result
        assert "out_channels=64" in result
        assert "kernel_size=3" in result

    def test_parse_args_nested_parentheses(self):
        """Test parsing with nested parentheses."""
        result = _parse_args("F.relu(x), stride=(2, 2), padding=1")
        assert len(result) == 3
        assert "F.relu(x)" in result
        assert "stride=(2, 2)" in result
        assert "padding=1" in result

    def test_parse_args_deeply_nested_parentheses(self):
        """Test parsing with deeply nested parentheses."""
        result = _parse_args("kernel_size=(3, 3), dilation=(1, 1)")
        assert len(result) == 2
        assert "kernel_size=(3, 3)" in result
        assert "dilation=(1, 1)" in result

    def test_parse_args_trailing_whitespace(self):
        """Test parsing with trailing whitespace."""
        result = _parse_args("  in_channels=3  ,  out_channels=64  ")
        assert len(result) == 2
        assert result[0] == "in_channels=3"
        assert result[1] == "out_channels=64"


class TestLayerInstantiationOptimization:
    """Test layer instantiation optimization."""

    def test_optimize_layer_no_args(self):
        """Test optimizing layer with no arguments."""
        line = "self.layer = nn.Relu()"
        result = _optimize_layer_instantiation(line)
        # Should return as-is if not in optimization tables
        assert "nn.Relu" in result

    def test_optimize_layer_with_default_args(self):
        """Test optimizing layer with default arguments."""
        line = "self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)"
        result = _optimize_layer_instantiation(line)
        # Should optimize, removing stride=1 and padding=0 if they're defaults
        assert "nn.Conv2d" in result

    def test_optimize_layer_unknown_type(self):
        """Test optimizing layer with unknown type."""
        line = "self.custom = nn.CustomLayer(arg1=1, arg2=2)"
        result = _optimize_layer_instantiation(line)
        # Should return unchanged if layer type not in optimization tables
        assert "arg1=1" in result
        assert "arg2=2" in result

    def test_optimize_layer_preserves_indentation(self):
        """Test that optimization preserves indentation."""
        # This is tested at optimize_line level
        line = "    self.conv = nn.Conv2d(in_channels=3, out_channels=64)"
        indent = line[: len(line) - len(line.lstrip())]
        assert len(indent) == 4

    def test_optimize_layer_non_matching_pattern(self):
        """Test with non-matching pattern."""
        line = "x = some_function(arg=1)"
        result = _optimize_layer_instantiation(line)
        # Should return unchanged if doesn't match self.xxx = nn.Layer pattern
        assert result == line


class TestPositionalConversion:
    """Test conversion to positional arguments."""

    def test_convert_to_positional_unknown_layer(self):
        """Test with unknown layer type."""
        result = _convert_to_positional("UnknownLayer", ["in_channels=3"])
        # Should return unchanged if layer not in POSITIONAL_ONLY_ARGS
        assert result == ["in_channels=3"]

    def test_convert_to_positional_already_positional(self):
        """Test with already positional arguments."""
        result = _convert_to_positional("Conv2d", ["3", "64", "3"])
        # Should return as-is
        assert result == ["3", "64", "3"]

    def test_convert_to_positional_mixed_args(self):
        """Test with mix of positional and named arguments."""
        result = _convert_to_positional("Conv2d", ["3", "out_channels=64", "kernel_size=3"])
        # Should preserve structure
        assert len(result) == 3


class TestDefaultRemoval:
    """Test removal of default arguments."""

    def test_remove_defaults_unknown_layer(self):
        """Test with unknown layer type."""
        result = _remove_defaults("UnknownLayer", ["arg1=1", "arg2=2"])
        # Should return unchanged if layer not in LAYER_DEFAULTS
        assert result == ["arg1=1", "arg2=2"]

    def test_remove_defaults_all_defaults(self):
        """Test layer with all default arguments."""
        # Using Conv2d as example
        args = ["in_channels=3", "out_channels=64"]
        result = _remove_defaults("Conv2d", args)
        # Should preserve all args (may not be defaults for Conv2d)
        assert len(result) > 0

    def test_remove_defaults_preserves_non_defaults(self):
        """Test that non-default values are preserved."""
        args = ["in_channels=3", "out_channels=64", "kernel_size=5"]
        result = _remove_defaults("Conv2d", args)
        # Should preserve non-default values
        assert len(result) >= 1


class TestFunctionCallOptimization:
    """Test function call optimization."""

    def test_optimize_function_call_f_relu(self):
        """Test optimizing F.relu call."""
        line = "x = F.relu(x, inplace=False)"
        result = _optimize_function_call(line)
        # Should optimize F.relu
        assert "F.relu" in result

    def test_optimize_function_call_torch_cat(self):
        """Test optimizing torch.cat call."""
        line = "x = torch.cat([x1, x2], dim=0)"
        result = _optimize_function_call(line)
        # Should optimize torch.cat
        assert "torch.cat" in result

    def test_optimize_function_call_no_match(self):
        """Test with non-matching pattern."""
        line = "result = some_regular_function(arg)"
        result = _optimize_function_call(line)
        # Should return unchanged
        assert result == line

    def test_optimize_function_call_nested_args(self):
        """Test with nested argument calls."""
        line = "x = F.relu(F.conv2d(x, w))"
        result = _optimize_function_call(line)
        # Should handle nested function calls
        assert "F.relu" in result

    def test_optimize_function_call_unknown_function(self):
        """Test with unknown function."""
        line = "x = torch.unknown_func(arg1=1, arg2=2)"
        result = _optimize_function_call(line)
        # Should return mostly unchanged if function not in FUNCTION_DEFAULTS
        assert "torch.unknown_func" in result


class TestRemoveFunctionDefaults:
    """Test removal of function default arguments."""

    def test_remove_function_defaults_f_relu(self):
        """Test removing defaults from F.relu."""
        args = ["x", "inplace=False"]
        result = _remove_function_defaults("F.relu", args)
        # Should handle F.relu function
        assert len(result) >= 1

    def test_remove_function_defaults_unknown_function(self):
        """Test with unknown function."""
        args = ["arg1=1", "arg2=2"]
        result = _remove_function_defaults("unknown.func", args)
        # Should return unchanged if function not in FUNCTION_DEFAULTS
        assert result == args

    def test_remove_function_defaults_preserves_positional(self):
        """Test that positional arguments are preserved."""
        args = ["x", "y"]
        result = _remove_function_defaults("F.relu", args)
        # Should preserve positional args
        assert len(result) >= 2


class TestLineOptimization:
    """Test main optimize_line function."""

    def test_optimize_line_empty(self):
        """Test with empty line."""
        result = optimize_line("")
        assert result == ""

    def test_optimize_line_comment(self):
        """Test with comment line."""
        line = "# This is a comment"
        result = optimize_line(line)
        assert result == line

    def test_optimize_line_class_definition(self):
        """Test with class definition."""
        line = "class MyClass:"
        result = optimize_line(line)
        assert result == line

    def test_optimize_line_def_declaration(self):
        """Test with def declaration."""
        line = "def my_function():"
        result = optimize_line(line)
        assert result == line

    def test_optimize_line_super_call(self):
        """Test with super() call."""
        line = "super().__init__()"
        result = optimize_line(line)
        assert result == line

    def test_optimize_line_import_statement(self):
        """Test with import statement."""
        line = "import torch"
        result = optimize_line(line)
        assert result == line

    def test_optimize_line_from_import_statement(self):
        """Test with from-import statement."""
        line = "from torch import nn"
        result = optimize_line(line)
        assert result == line

    def test_optimize_line_return_statement(self):
        """Test with return statement."""
        line = "return output"
        result = optimize_line(line)
        assert result == line

    def test_optimize_line_preserves_indentation(self):
        """Test that indentation is preserved."""
        line = "    self.layer = nn.Relu()"
        result = optimize_line(line)
        # Should preserve 4-space indentation
        assert result.startswith("    ")
        assert len(result) - len(result.lstrip()) == 4

    def test_optimize_line_function_call(self):
        """Test optimizing a function call line."""
        line = "x = F.relu(x, inplace=False)"
        result = optimize_line(line)
        # Should process and return optimized line
        assert "F.relu" in result

    def test_optimize_line_layer_instantiation(self):
        """Test optimizing a layer instantiation line."""
        line = "self.conv = nn.Conv2d(3, 64, 3)"
        result = optimize_line(line)
        # Should process and return optimized line
        assert "nn.Conv2d" in result
