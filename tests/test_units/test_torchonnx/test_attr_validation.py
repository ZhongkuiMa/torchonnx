"""Phase 5: Attribute Extractor Error Handling Tests.

This module tests attribute validation and error handling in attr_extractor.py.

Tests cover:
- Unsupported attribute types (STRINGS, TENSORS, SPARSE_TENSOR)
- Operator-specific validation errors
- Required attribute checks
- Default value constraints

Target: attr_extractor.py coverage improvement from 74.5% to 90%+
"""

import onnx
import pytest
from onnx import TensorProto
from onnx import helper as onnx_helper

from torchonnx.analyze.attr_extractor import extract_onnx_attrs


class TestUnsupportedAttributeTypes:
    """Test handling of unsupported ONNX attribute types."""

    def test_attribute_type_strings_returns_none(self):
        """Test that STRINGS attribute type is handled (returns None)."""
        # Create a node with STRINGS attribute (type 8)
        node = onnx_helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        # Manually create attribute with type 8 (STRINGS)
        attr = onnx.AttributeProto()
        attr.name = "unsupported_strings"
        attr.type = 8  # STRINGS type
        node.attribute.append(attr)

        attrs = extract_onnx_attrs(node, {})
        # STRINGS type returns None
        assert attrs.get("unsupported_strings") is None

    def test_attribute_type_tensors_returns_none(self):
        """Test that TENSORS attribute type is handled (returns None)."""
        node = onnx_helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        # Manually create attribute with type 9 (TENSORS)
        attr = onnx.AttributeProto()
        attr.name = "unsupported_tensors"
        attr.type = 9  # TENSORS type
        node.attribute.append(attr)

        attrs = extract_onnx_attrs(node, {})
        # TENSORS type returns None
        assert attrs.get("unsupported_tensors") is None

    def test_attribute_type_sparse_tensor_returns_none(self):
        """Test that SPARSE_TENSOR attribute type is handled (returns None)."""
        node = onnx_helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        # Manually create attribute with type 11 (SPARSE_TENSOR)
        attr = onnx.AttributeProto()
        attr.name = "unsupported_sparse"
        attr.type = 11  # SPARSE_TENSOR type
        node.attribute.append(attr)

        attrs = extract_onnx_attrs(node, {})
        # SPARSE_TENSOR type returns None
        assert attrs.get("unsupported_sparse") is None


class TestValidationErrors:
    """Test operator-specific validation errors."""

    def test_argmax_select_last_index_not_zero_raises(self):
        """Test ArgMax with select_last_index != 0 raises error."""
        node = onnx_helper.make_node("ArgMax", inputs=["X"], outputs=["Y"], select_last_index=1)
        with pytest.raises(ValueError, match="select_last_index"):
            extract_onnx_attrs(node, {})

    def test_batchnorm_training_mode_not_zero_raises(self):
        """Test BatchNormalization with training_mode != 0 raises error."""
        node = onnx_helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y"],
            training_mode=1,
        )
        with pytest.raises(ValueError, match="training_mode"):
            extract_onnx_attrs(node, {})

    def test_batchnorm_multiple_outputs_raises(self):
        """Test BatchNormalization with multiple outputs raises error."""
        node = onnx_helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y", "mean_out", "var_out"],
            training_mode=0,
        )
        with pytest.raises(ValueError, match="outputs"):
            extract_onnx_attrs(node, {})

    def test_cast_saturate_not_one_raises(self):
        """Test Cast with saturate != 1 raises error."""
        node = onnx_helper.make_node(
            "Cast", inputs=["X"], outputs=["Y"], to=TensorProto.FLOAT, saturate=0
        )
        with pytest.raises(ValueError, match="saturate"):
            extract_onnx_attrs(node, {})

    def test_cast_missing_to_attribute_raises(self):
        """Test Cast without 'to' attribute raises error."""
        node = onnx_helper.make_node("Cast", inputs=["X"], outputs=["Y"])
        with pytest.raises(ValueError, match="to"):
            extract_onnx_attrs(node, {})

    def test_constantofshape_missing_value_raises(self):
        """Test ConstantOfShape without value raises error."""
        node = onnx_helper.make_node("ConstantOfShape", inputs=["shape"], outputs=["Y"])
        with pytest.raises(ValueError, match="value"):
            extract_onnx_attrs(node, {})

    def test_convtranspose_group_not_one_raises(self):
        """Test ConvTranspose with group != 1 raises error."""
        node = onnx_helper.make_node(
            "ConvTranspose", inputs=["X", "W"], outputs=["Y"], group=2, kernel_shape=[3, 3]
        )
        with pytest.raises(ValueError, match="group"):
            extract_onnx_attrs(node, {})

    def test_maxpool_kernel_shape_required_raises(self):
        """Test MaxPool without kernel_shape raises error."""
        node = onnx_helper.make_node("MaxPool", inputs=["X"], outputs=["Y"])
        with pytest.raises(ValueError, match="kernel_shape"):
            extract_onnx_attrs(node, {})

    def test_maxpool_storage_order_not_zero_raises(self):
        """Test MaxPool with storage_order != 0 raises error."""
        node = onnx_helper.make_node(
            "MaxPool", inputs=["X"], outputs=["Y"], kernel_shape=[3, 3], storage_order=1
        )
        with pytest.raises(ValueError, match="storage_order"):
            extract_onnx_attrs(node, {})

    def test_maxpool_multiple_outputs_raises(self):
        """Test MaxPool with multiple outputs raises error."""
        node = onnx_helper.make_node(
            "MaxPool", inputs=["X"], outputs=["Y", "indices"], kernel_shape=[3, 3]
        )
        with pytest.raises(ValueError, match="outputs"):
            extract_onnx_attrs(node, {})

    def test_reshape_allowzero_not_zero_raises(self):
        """Test Reshape with allowzero != 0 raises error."""
        node = onnx_helper.make_node("Reshape", inputs=["X", "shape"], outputs=["Y"], allowzero=1)
        with pytest.raises(ValueError, match="allowzero"):
            extract_onnx_attrs(node, {})

    def test_scatternd_reduction_not_none_raises(self):
        """Test ScatterND with reduction != 'none' raises error."""
        node = onnx_helper.make_node(
            "ScatterND", inputs=["X", "indices", "updates"], outputs=["Y"], reduction="add"
        )
        with pytest.raises(ValueError, match="reduction"):
            extract_onnx_attrs(node, {})

    def test_shape_end_not_negative_one_raises(self):
        """Test Shape with end != -1 raises error."""
        node = onnx_helper.make_node("Shape", inputs=["X"], outputs=["Y"], end=5)
        with pytest.raises(ValueError, match="end"):
            extract_onnx_attrs(node, {})

    def test_shape_start_not_zero_raises(self):
        """Test Shape with start != 0 raises error."""
        node = onnx_helper.make_node("Shape", inputs=["X"], outputs=["Y"], start=1)
        with pytest.raises(ValueError, match="start"):
            extract_onnx_attrs(node, {})


class TestRequiredAttributes:
    """Test handling of required attributes."""

    def test_concat_axis_required(self):
        """Test Concat requires axis attribute."""
        node = onnx_helper.make_node("Concat", inputs=["X1", "X2"], outputs=["Y"])
        with pytest.raises(ValueError, match="axis"):
            extract_onnx_attrs(node, {})

    def test_transpose_perm_required(self):
        """Test Transpose requires perm attribute."""
        node = onnx_helper.make_node("Transpose", inputs=["X"], outputs=["Y"])
        with pytest.raises(ValueError, match="perm"):
            extract_onnx_attrs(node, {})

    def test_averagepool_kernel_shape_required(self):
        """Test AveragePool requires kernel_shape attribute."""
        node = onnx_helper.make_node("AveragePool", inputs=["X"], outputs=["Y"])
        with pytest.raises(ValueError, match="kernel_shape"):
            extract_onnx_attrs(node, {})


class TestAsymmetricPaddingValidation:
    """Test asymmetric padding validation."""

    def test_conv_asymmetric_padding_raises(self):
        """Test Conv with asymmetric padding raises error."""
        node = onnx_helper.make_node(
            "Conv", inputs=["X", "W"], outputs=["Y"], kernel_shape=[3, 3], pads=[1, 2, 2, 1]
        )
        with pytest.raises(ValueError, match="Asymmetric"):
            extract_onnx_attrs(node, {})

    def test_avgpool_asymmetric_padding_raises(self):
        """Test AveragePool with asymmetric padding raises error."""
        node = onnx_helper.make_node(
            "AveragePool", inputs=["X"], outputs=["Y"], kernel_shape=[3, 3], pads=[1, 2, 2, 1]
        )
        with pytest.raises(ValueError, match="Asymmetric"):
            extract_onnx_attrs(node, {})

    def test_maxpool_asymmetric_padding_raises(self):
        """Test MaxPool with asymmetric padding raises error."""
        node = onnx_helper.make_node(
            "MaxPool", inputs=["X"], outputs=["Y"], kernel_shape=[3, 3], pads=[1, 2, 2, 1]
        )
        with pytest.raises(ValueError, match="Asymmetric"):
            extract_onnx_attrs(node, {})


class TestValidAttributesExtraction:
    """Test valid attribute extraction."""

    def test_extract_argmax_valid(self):
        """Test ArgMax attribute extraction with valid values."""
        node = onnx_helper.make_node(
            "ArgMax", inputs=["X"], outputs=["Y"], axis=1, select_last_index=0
        )
        attrs = extract_onnx_attrs(node, {})
        assert attrs["axis"] == 1
        assert attrs["select_last_index"] == 0

    def test_extract_cast_valid(self):
        """Test Cast attribute extraction with valid values."""
        node = onnx_helper.make_node(
            "Cast", inputs=["X"], outputs=["Y"], to=TensorProto.FLOAT, saturate=1
        )
        attrs = extract_onnx_attrs(node, {})
        assert attrs["to"] == TensorProto.FLOAT
        assert attrs["saturate"] == 1

    def test_extract_batchnorm_valid(self):
        """Test BatchNormalization attribute extraction with valid values."""
        node = onnx_helper.make_node(
            "BatchNormalization",
            inputs=["X", "scale", "bias", "mean", "var"],
            outputs=["Y"],
            epsilon=1e-5,
            momentum=0.9,
            training_mode=0,
        )
        attrs = extract_onnx_attrs(node, {})
        assert attrs["training_mode"] == 0
        assert attrs["epsilon"] == pytest.approx(1e-5)

    def test_extract_maxpool_valid(self):
        """Test MaxPool attribute extraction with valid values."""
        node = onnx_helper.make_node(
            "MaxPool",
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            storage_order=0,
        )
        attrs = extract_onnx_attrs(node, {})
        assert attrs["kernel_shape"] == (3, 3)
        assert attrs["storage_order"] == 0

    def test_extract_shape_valid(self):
        """Test Shape attribute extraction with valid values."""
        node = onnx_helper.make_node("Shape", inputs=["X"], outputs=["Y"], start=0, end=-1)
        attrs = extract_onnx_attrs(node, {})
        assert attrs["start"] == 0
        assert attrs["end"] == -1
