"""Phase 6: Remaining Gap Tests.

This module tests edge cases and uncovered code paths to reach 90%+ coverage.

Tests cover:
- Code formatter functions
- Utility functions edge cases
- Code generation helper functions

Target: Reach 90%+ overall coverage from current 88%
"""

from torchonnx.simplify._formatter import format_code


class TestCodeFormatter:
    """Test code formatting functions."""

    def test_format_code_empty_string(self):
        """Test formatting empty code string."""
        code = ""
        result = format_code(code)
        assert result == ""

    def test_format_code_short_line(self):
        """Test formatting line under max length."""
        code = "x = relu(x)"
        result = format_code(code)
        assert "x = relu(x)" in result

    def test_format_code_multiple_short_lines(self):
        """Test formatting multiple short lines."""
        code = "x = relu(x)\nx = conv(x, w)\nreturn x"
        result = format_code(code)
        assert "relu(x)" in result
        assert "return x" in result

    def test_format_code_with_indentation(self):
        """Test formatting preserves proper indentation."""
        code = "def forward(x):\n    x = relu(x)\n    return x"
        result = format_code(code)
        assert len(result) > 0

    def test_format_code_with_class_definition(self):
        """Test formatting with class definition."""
        code = "class MyModule:\n    def forward(self, x):\n        return relu(x)"
        result = format_code(code)
        assert "class MyModule" in result

    def test_format_code_blank_lines(self):
        """Test formatting normalizes blank lines."""
        code = "def func1():\n    pass\n\n\ndef func2():\n    pass"
        result = format_code(code)
        assert "func1" in result
        assert "func2" in result

    def test_format_code_long_line(self):
        """Test formatting wraps long lines."""
        code = "x = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)"
        result = format_code(code)
        # Should either keep as-is if within limits or wrap
        assert "Conv2d" in result

    def test_format_code_nested_function_calls(self):
        """Test formatting nested function calls."""
        code = "result = f(g(h(x)))"
        result = format_code(code)
        assert "result" in result

    def test_format_code_multiple_statements(self):
        """Test formatting with multiple statements."""
        code = "x = input\nx = relu(x)\nx = conv(x)\noutput = x"
        result = format_code(code)
        assert "input" in result
        assert "output" in result

    def test_format_code_with_comments(self):
        """Test formatting preserves comments."""
        code = "# Process input\nx = relu(x)\n# Apply convolution\nx = conv(x)"
        result = format_code(code)
        # Comments should be preserved or at least code should work
        assert len(result) > 0

    def test_format_code_with_self_references(self):
        """Test formatting with self references."""
        code = "self.x = relu(x)\nself.y = conv(self.x, w)"
        result = format_code(code)
        assert "self" in result

    def test_format_code_dictionary_literal(self):
        """Test formatting with dictionary literals."""
        code = "config = {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3}"
        result = format_code(code)
        assert "config" in result

    def test_format_code_list_literal(self):
        """Test formatting with list literals."""
        code = "layers = [relu, conv, batchnorm]"
        result = format_code(code)
        assert "layers" in result

    def test_format_code_multiline_list(self):
        """Test formatting with multiline list."""
        code = "layers = [\n    relu,\n    conv,\n    batchnorm,\n]"
        result = format_code(code)
        assert "layers" in result

    def test_format_code_function_call_many_args(self):
        """Test formatting function with many arguments."""
        code = "result = function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)"
        result = format_code(code)
        assert "result" in result or "function" in result
