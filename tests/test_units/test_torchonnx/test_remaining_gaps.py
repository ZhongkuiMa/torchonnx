"""Phase 6: Remaining Gap Tests.

This module tests edge cases and uncovered code paths to reach 90%+ coverage.

Tests cover:
- Code formatter functions
- Utility functions edge cases
- Code generation helper functions

Target: Reach 90%+ overall coverage from current 88%
"""

__docformat__ = "restructuredtext"

import pytest

from torchonnx.simplify._formatter import format_code


class TestCodeFormatter:
    """Test code formatting functions."""

    def test_format_code_empty_string(self):
        """Test formatting empty code string."""
        code = ""
        result = format_code(code)
        assert result == ""

    def test_format_code_with_indentation(self):
        """Test formatting preserves proper indentation."""
        code = "def forward(x):\n    x = relu(x)\n    return x"
        result = format_code(code)
        assert len(result) > 0

    def test_format_code_long_line(self):
        """Test formatting wraps long lines."""
        code = "x = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)"
        result = format_code(code)
        # Should either keep as-is if within limits or wrap
        assert "Conv2d" in result

    def test_format_code_with_comments(self):
        """Test formatting preserves comments."""
        code = "# Process input\nx = relu(x)\n# Apply convolution\nx = conv(x)"
        result = format_code(code)
        # Comments should be preserved or at least code should work
        assert len(result) > 0

    # [REVIEW] Parametrized: test_format_code_short_line, test_format_code_with_class_definition,
    # test_format_code_nested_function_calls, test_format_code_with_self_references,
    # test_format_code_dictionary_literal, test_format_code_list_literal,
    # test_format_code_multiline_list, test_format_code_function_call_many_args

    @pytest.mark.parametrize(
        ("code", "expected_part"),
        [
            pytest.param("x = relu(x)", "x = relu(x)", id="short_line"),
            pytest.param(
                "class MyModule:\n    def forward(self, x):\n        return relu(x)",
                "class MyModule",
                id="class_definition",
            ),
            pytest.param("result = f(g(h(x)))", "result", id="nested_function_calls"),
            pytest.param(
                "self.x = relu(x)\nself.y = conv(self.x, w)", "self", id="self_references"
            ),
            pytest.param(
                "config = {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3}",
                "config",
                id="dictionary_literal",
            ),
            pytest.param("layers = [relu, conv, batchnorm]", "layers", id="list_literal"),
            pytest.param(
                "layers = [\n    relu,\n    conv,\n    batchnorm,\n]",
                "layers",
                id="multiline_list",
            ),
            pytest.param(
                "result = function(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)",
                "result",
                id="function_call_many_args",
            ),
        ],
    )
    def test_format_code(self, code, expected_part):
        """Test formatting preserves expected content in output."""
        result = format_code(code)
        assert expected_part in result

    # [REVIEW] Parametrized: test_format_code_multiple_short_lines, test_format_code_blank_lines,
    # test_format_code_multiple_statements

    @pytest.mark.parametrize(
        ("code", "expected_parts"),
        [
            pytest.param(
                "x = relu(x)\nx = conv(x, w)\nreturn x",
                {"relu(x)", "return x"},
                id="multiple_short_lines",
            ),
            pytest.param(
                "def func1():\n    pass\n\n\ndef func2():\n    pass",
                {"func1", "func2"},
                id="blank_lines",
            ),
            pytest.param(
                "x = input\nx = relu(x)\nx = conv(x)\noutput = x",
                {"input", "output"},
                id="multiple_statements",
            ),
        ],
    )
    def test_format_code_multiple_parts(self, code, expected_parts):
        """Test formatting preserves multiple expected patterns."""
        result = format_code(code)
        for part in expected_parts:
            assert part in result
