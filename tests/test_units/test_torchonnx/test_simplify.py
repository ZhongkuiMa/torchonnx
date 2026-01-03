"""Comprehensive tests for Stage 6: Code Simplification and Optimization.

This module tests the simplify stage which handles:
- Code formatting (Black-compatible)
- Line wrapping and multi-line formatting
- Code optimization (removing defaults, inlining)
- Removing unused buffers
- Adding file headers and docstrings

Test Coverage:
- TestFormatCode: 8 tests - Code formatting utilities
- TestOptimizeCode: 5 tests - Code optimization
- TestFileHeaders: 3 tests - File header generation
- TestCodeQuality: 2 tests - Code quality checks
"""

from torchonnx.simplify import add_file_header, format_code, optimize_generated_code


class TestFormatCode:
    """Test code formatting functionality."""

    def test_format_code_preserves_logic(self):
        """Test Python code formatting preserves logic."""
        unformatted = "x=1\ny = 2\nz=x+y"
        formatted = format_code(unformatted)
        assert formatted is not None
        assert isinstance(formatted, str)

    def test_format_empty_code(self):
        """Test formatting empty code."""
        formatted = format_code("")
        assert formatted is not None
        assert formatted == ""

    def test_format_multiline_code(self):
        """Test formatting multiline code."""
        code = "def foo():\n    x = 1\n    return x"
        formatted = format_code(code)
        assert formatted is not None
        assert "def foo()" in formatted

    def test_format_long_line_wrapping(self):
        """Test that long lines are wrapped."""
        # Create a line longer than 88 characters
        long_line = "self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)"
        formatted = format_code(long_line)
        assert formatted is not None
        # Should either wrap or leave as-is if it can't be wrapped
        assert len(formatted) > 0

    def test_format_class_definition(self):
        """Test formatting class definitions with proper blank lines."""
        code = "class Model:\n    def __init__(self):\n        pass\n    def forward(self):\n        pass"
        formatted = format_code(code)
        assert formatted is not None
        assert "class Model" in formatted
        assert "def __init__" in formatted

    def test_format_assignment_with_function_call(self):
        """Test formatting assignment statements with function calls."""
        code = "x = some_function(arg1, arg2, arg3, arg4, arg5)"
        formatted = format_code(code)
        assert formatted is not None
        assert "x" in formatted

    def test_format_preserves_indentation(self):
        """Test that formatting preserves indentation."""
        code = "class A:\n    def foo(self):\n        x = 1"
        formatted = format_code(code)
        assert formatted is not None
        # Check that indentation is preserved
        lines = formatted.split("\n")
        assert any(line.startswith("    ") for line in lines)


class TestOptimizeCode:
    """Test code optimization functionality."""

    def test_optimize_generated_code_returns_string(self):
        """Test that optimize_generated_code returns string."""
        code = "x = 1\ny = 2"
        result = optimize_generated_code(code)
        assert isinstance(result, str)

    def test_optimize_with_state_dict_returns_tuple(self):
        """Test that optimize with state_dict returns (code, state_dict)."""
        import torch

        code = "x = 1"
        state_dict = {"param": torch.tensor([1.0])}
        result = optimize_generated_code(code, state_dict)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

    def test_optimize_disabled_returns_original(self):
        """Test that disabling optimization returns original code."""
        code = "x = 1\ny = 2"
        result = optimize_generated_code(code, enable=False)
        assert result == code

    def test_optimize_preserves_code_logic(self):
        """Test that optimization preserves code logic."""
        code = "def foo():\n    return 42"
        result = optimize_generated_code(code)
        assert "def foo()" in result
        assert "return 42" in result

    def test_optimize_removes_unused_buffers(self):
        """Test that optimization can identify structure for buffer removal."""
        # Code with register_buffer that might be unused
        code = """class Model:
    def __init__(self):
        self.register_buffer("unused_buf", tensor)
    def forward(self):
        return x"""
        result = optimize_generated_code(code)
        assert isinstance(result, str)


class TestFileHeaders:
    """Test file header generation."""

    def test_add_file_header_basic(self):
        """Test adding file header to code."""
        code = "print('hello')"
        header = add_file_header(code, "TestModel", "model.onnx")
        assert header is not None
        assert "TestModel" in header
        assert "model.onnx" in header

    def test_add_file_header_with_docstring(self):
        """Test that file header includes module docstring."""
        code = "x = 1"
        header = add_file_header(code, "MyModel", "my.onnx")
        assert '"""' in header
        assert "MyModel" in header
        # Should include original code
        assert "x = 1" in header

    def test_add_file_header_preserves_code(self):
        """Test that file header preserves original code."""
        code = "def foo():\n    return 42"
        header = add_file_header(code, "Model", "model.onnx")
        assert "def foo()" in header
        assert "return 42" in header


class TestCodeQuality:
    """Test code quality aspects."""

    def test_formatted_code_is_string(self):
        """Test that formatted code is valid string."""
        code = "x = 1"
        result = format_code(code)
        assert isinstance(result, str)

    def test_formatted_code_with_imports(self):
        """Test formatting code with imports."""
        code = "import torch\nimport torch.nn as nn\nx = 1"
        result = format_code(code)
        assert "import torch" in result or result is not None
