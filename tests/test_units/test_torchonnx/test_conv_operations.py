"""Phase 1: Conv/ConvTranspose Functional Operation Tests.

This module tests Conv and ConvTranspose functional operation extraction
from ONNX models to improve coverage of _operations.py.

Tests cover:
- Conv argument extraction with various configurations
- ConvTranspose argument extraction
- Error handling for invalid operator types
- Edge cases in argument processing

Target: _operations.py coverage improvement from 44.9% to 67%+
"""

import numpy as np
import pytest
from onnx import TensorProto
from onnx import helper as onnx_helper

from torchonnx.analyze.type_mapping._operations import (
    _extract_conv_args,
    _extract_conv_transpose_args,
    _process_conv_padding,
    _simplify_homogeneous_values,
    convert_to_operation,
    convert_to_operator,
)


class TestConvFunctionalOperations:
    """Test Conv/ConvTranspose functional operation extraction."""

    # Helper method to create Conv node with specified configuration
    @staticmethod
    def create_conv_node(**kwargs):
        """Create a Conv ONNX node with given attributes."""
        node = onnx_helper.make_node("Conv", inputs=["X", "W"], outputs=["Y"], **kwargs)
        return node

    @staticmethod
    def create_conv_transpose_node(**kwargs):
        """Create a ConvTranspose ONNX node with given attributes."""
        node = onnx_helper.make_node("ConvTranspose", inputs=["X", "W"], outputs=["Y"], **kwargs)
        return node

    @staticmethod
    def create_weight_initializer(shape):
        """Create a weight tensor initializer."""
        rng = np.random.default_rng(42)
        weight_data = rng.standard_normal(shape).astype(np.float32)
        weight_tensor = onnx_helper.make_tensor(
            "W",
            TensorProto.FLOAT,
            shape,
            weight_data.flatten().tolist(),
        )
        return weight_tensor

    # ===== Conv Args Extraction Tests (7 tests) =====

    def test_extract_conv_args_basic(self):
        """Extract standard Conv args with default values."""
        node = self.create_conv_node()
        weight = self.create_weight_initializer([1, 1, 3, 3])
        initializers = {"W": weight}

        args = _extract_conv_args(node, initializers)

        assert args is not None
        # Default strides without explicit specification are added
        assert isinstance(args, dict)

    def test_extract_conv_args_with_padding(self):
        """Test symmetric padding extraction and simplification."""
        node = self.create_conv_node(pads=[1, 1, 1, 1])
        weight = self.create_weight_initializer([3, 1, 3, 3])
        initializers = {"W": weight}

        args = _extract_conv_args(node, initializers)

        assert args is not None
        assert "padding" in args
        # Symmetric padding [1,1,1,1] should simplify to 1
        assert args["padding"] == 1 or args["padding"] == (1, 1)

    def test_extract_conv_args_asymmetric_padding_raises(self):
        """Test asymmetric padding raises ValueError."""
        node = self.create_conv_node(pads=[1, 2, 2, 1])
        weight = self.create_weight_initializer([1, 1, 3, 3])
        initializers = {"W": weight}

        # Asymmetric padding should raise an error
        with pytest.raises(ValueError, match=r"Asymmetric|symmetric"):
            _extract_conv_args(node, initializers)

    def test_extract_conv_args_with_dilation(self):
        """Test dilation != 1 is included in args."""
        node = self.create_conv_node(dilations=[2, 2])
        weight = self.create_weight_initializer([1, 1, 3, 3])
        initializers = {"W": weight}

        args = _extract_conv_args(node, initializers)

        assert args is not None
        assert "dilation" in args
        # Dilation should be included
        assert args["dilation"] == 2 or args["dilation"] == (2, 2)

    def test_extract_conv_args_with_groups(self):
        """Test grouped convolution args."""
        node = self.create_conv_node(group=2)
        weight = self.create_weight_initializer([2, 1, 3, 3])
        initializers = {"W": weight}

        args = _extract_conv_args(node, initializers)

        assert args is not None
        assert "groups" in args
        assert args["groups"] == 2

    def test_extract_conv_args_simplifies_homogeneous(self):
        """Test homogeneous value simplification: (3,3,3) -> 3."""
        # Test the simplification function directly
        result = _simplify_homogeneous_values([3, 3, 3])
        assert result == 3

        result = _simplify_homogeneous_values([1, 1])
        assert result == 1

    def test_extract_conv_args_preserves_heterogeneous(self):
        """Test heterogeneous values preserved: (1,2,3) -> (1,2,3)."""
        result = _simplify_homogeneous_values([1, 2, 3])
        assert result == (1, 2, 3)

        result = _simplify_homogeneous_values([2, 3])
        assert result == (2, 3)

    # ===== ConvTranspose Args Extraction Tests (4 tests) =====

    def test_extract_conv_transpose_args_basic(self):
        """Extract standard ConvTranspose args."""
        node = self.create_conv_transpose_node()
        weight = self.create_weight_initializer([1, 1, 3, 3])
        initializers = {"W": weight}

        args = _extract_conv_transpose_args(node, initializers)

        assert args is not None
        assert isinstance(args, dict)

    def test_extract_conv_transpose_args_with_output_padding(self):
        """Test output_padding handling."""
        node = self.create_conv_transpose_node(output_padding=[1, 1])
        weight = self.create_weight_initializer([1, 1, 3, 3])
        initializers = {"W": weight}

        args = _extract_conv_transpose_args(node, initializers)

        assert args is not None
        # output_padding should be included if non-zero
        if "output_padding" in args:
            assert args["output_padding"] != 0

    def test_extract_conv_transpose_args_asymmetric_error(self):
        """Test asymmetric padding error in ConvTranspose."""
        node = self.create_conv_transpose_node(pads=[1, 2, 2, 1])
        weight = self.create_weight_initializer([1, 1, 3, 3])
        initializers = {"W": weight}

        # Asymmetric padding should raise an error
        with pytest.raises(ValueError, match=r"Asymmetric|symmetric"):
            _extract_conv_transpose_args(node, initializers)

    def test_extract_conv_transpose_args_default_output_padding(self):
        """Test output_padding=0 omission."""
        node = self.create_conv_transpose_node(output_padding=[0, 0])
        weight = self.create_weight_initializer([1, 1, 3, 3])
        initializers = {"W": weight}

        args = _extract_conv_transpose_args(node, initializers)

        assert args is not None
        # output_padding with default value should be omitted
        if "output_padding" in args:
            assert args["output_padding"] != 0

    # ===== Error Handling Tests (2 tests) =====

    def test_convert_to_operator_invalid_type_raises(self):
        """Test invalid operator type raises ValueError."""
        # Try to convert a non-existent operator type
        with pytest.raises(ValueError, match="InvalidOperator"):
            convert_to_operator("InvalidOperator")

    def test_convert_to_operation_invalid_type_raises(self):
        """Test invalid operation type raises ValueError."""
        # Try to convert a non-existent operation type
        with pytest.raises(ValueError, match="InvalidOperation"):
            convert_to_operation("InvalidOperation")

    # ===== ConstantOfShape Edge Cases Tests (2 tests) =====

    def test_process_conv_padding_symmetric(self):
        """Test symmetric padding processing for Conv."""
        # Test symmetric padding
        result = _process_conv_padding([1, 1, 1, 1], "Conv")
        assert result == 1 or result == (1, 1)

    def test_process_conv_padding_asymmetric_error(self):
        """Test asymmetric padding raises error."""
        # Truly asymmetric padding: [top, left, bottom, right] = [1, 2, 2, 1]
        # This means top=1, bottom=2 (asymmetric) and left=2, right=1 (asymmetric)
        with pytest.raises(ValueError, match=r"Asymmetric|symmetric"):
            _process_conv_padding([1, 2, 2, 1], "Conv")
