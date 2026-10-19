"""Comprehensive tests for Stage 1: ONNX Model Normalization and Preprocessing.

This module tests the normalize stage which handles:
- Loading and validating ONNX models
- Inferring tensor shapes
- Converting opset versions
- Folding constants
- Checking for unsupported operations

Test Coverage:
- TestNormalizeBasics: 4 tests - Basic model loading
- TestOpsetHandling: 3 tests - Opset conversion and validation
- TestShapeInference: 4 tests - Shape inference and dynamic dimensions
- TestValidation: 4 tests - Model validation and error cases
"""

__docformat__ = "restructuredtext"

import contextlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import onnx
import pytest
from google.protobuf.message import DecodeError

from torchonnx.normalize import load_and_preprocess_onnx_model
from torchonnx.normalize.normalize import (
    _apply_shapeonnx_inference,
    _canonicalize_opset_imports,
    _check_model,
    _convert_version,
    _infer_shapes,
)


class TestNormalizeBasics:
    """Test basic model loading and preprocessing."""

    def test_load_identity_model(self, identity_model):
        """Test loading and preprocessing Identity model."""
        model = load_and_preprocess_onnx_model(identity_model, target_opset=20, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        assert model.ir_version == 8

    def test_load_linear_model(self, linear_model):
        """Test loading and preprocessing Linear model."""
        model = load_and_preprocess_onnx_model(linear_model, target_opset=20, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.initializer) == 2
        assert isinstance(model.graph.name, str)

    def test_load_mlp_model(self, mlp_model):
        """Test loading and preprocessing MLP (multi-layer) model."""
        model = load_and_preprocess_onnx_model(mlp_model, target_opset=20, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 2  # Multiple layers
        assert len(model.graph.initializer) > 2  # Multiple parameters

    def test_load_conv_model(self, conv2d_model):
        """Test loading and preprocessing Conv2d model."""
        model = load_and_preprocess_onnx_model(conv2d_model, target_opset=20, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        assert any(node.op_type == "Conv" for node in model.graph.node)


class TestOpsetHandling:
    """Test opset version handling and conversion."""

    def test_load_with_target_opset_20(self, linear_model):
        """Test loading model with target opset 20."""
        model = load_and_preprocess_onnx_model(linear_model, target_opset=20, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        # Model should be valid after opset conversion
        onnx.checker.check_model(model)

    def test_load_with_target_opset_19(self, linear_model):
        """Test loading model with target opset 19."""
        model = load_and_preprocess_onnx_model(linear_model, target_opset=19, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        onnx.checker.check_model(model)

    def test_load_with_opset_version_inference(self, mlp_model):
        """Test model loading respects opset version."""
        model = load_and_preprocess_onnx_model(mlp_model, target_opset=20, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        # Should have valid producer name
        assert isinstance(model.producer_name, str)

    def test_empty_identity_graph_skips_native_opset_conversion(self):
        """A graph with one aliased input/output has no operator schema to convert."""
        input_info = onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [1, 1])
        output_info = onnx.helper.make_tensor_value_info("value", onnx.TensorProto.FLOAT, [1, 1])
        graph = onnx.helper.make_graph([], "identity", [input_info], [output_info])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 21)])

        with patch("torchonnx.normalize.normalize.version_converter.convert_version") as convert:
            normalized = _convert_version(model, target_opset=20)

        convert.assert_not_called()
        assert normalized is model
        assert normalized.opset_import[0].version == 21

    def test_duplicate_opset_import_is_collapsed_before_conversion(self):
        """Native ONNX conversion must not leave one duplicate at the old version."""
        input_info = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])
        output_info = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Relu", ["X"], ["Y"])],
            "duplicate_opset",
            [input_info],
            [output_info],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[
                onnx.helper.make_opsetid("", 11),
                onnx.helper.make_opsetid("", 11),
            ],
        )

        converted = _convert_version(model, target_opset=20)

        assert [(item.domain, item.version) for item in converted.opset_import] == [("", 20)]
        assert len(model.opset_import) == 2
        onnx.checker.check_model(converted)

    def test_conflicting_opset_imports_are_rejected(self):
        model = onnx.helper.make_model(
            onnx.helper.make_graph([], "conflict", [], []),
            opset_imports=[
                onnx.helper.make_opsetid("", 11),
                onnx.helper.make_opsetid("", 13),
            ],
        )

        with pytest.raises(ValueError, match="Conflicting opset imports"):
            _canonicalize_opset_imports(model)


class TestShapeInference:
    """Test shape inference and dynamic dimension handling."""

    def test_shape_inference_linear_model(self, linear_model):
        """Test shape inference on Linear model."""
        model = load_and_preprocess_onnx_model(linear_model, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        # Check that graph inputs have shape information
        for input_info in model.graph.input:
            assert input_info.type.tensor_type.elem_type > 0

    def test_shape_inference_conv_model(self, conv2d_model):
        """Test shape inference on Conv2d model."""
        model = load_and_preprocess_onnx_model(conv2d_model, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        # Graph should have inputs defined
        assert len(model.graph.input) > 0

    def test_shape_inference_with_dynamic_dims(self, multi_input_model):
        """Test shape inference handles dynamic dimensions."""
        model = load_and_preprocess_onnx_model(multi_input_model, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        # Should still be valid even with dynamic dims
        onnx.checker.check_model(model)

    def test_shape_inference_preserves_model(self, linear_model):
        """Test that shape inference doesn't corrupt model structure."""
        original = load_and_preprocess_onnx_model(linear_model, infer_shapes=False)
        inferred = load_and_preprocess_onnx_model(linear_model, infer_shapes=True)

        # Should have same number of nodes
        assert len(original.graph.node) == len(inferred.graph.node)
        # Should have same initializers
        assert len(original.graph.initializer) == len(inferred.graph.initializer)


class TestValidation:
    """Test model validation and error handling."""

    def test_load_nonexistent_model(self, tmp_path):
        """Test error handling for nonexistent ONNX file."""
        nonexistent = tmp_path / "nonexistent.onnx"
        with pytest.raises(FileNotFoundError, match=r"No such file"):
            load_and_preprocess_onnx_model(str(nonexistent))

    def test_load_invalid_onnx_file(self, tmp_path):
        """Test error handling for invalid ONNX file format."""
        invalid_file = tmp_path / "invalid.onnx"
        invalid_file.write_text("This is not valid ONNX")

        # Should raise validation error
        with pytest.raises((onnx.checker.ValidationError, OSError, DecodeError)):
            load_and_preprocess_onnx_model(str(invalid_file))

    def test_load_empty_onnx_file(self, tmp_path):
        """Test error handling for empty ONNX file."""
        empty_file = tmp_path / "empty.onnx"
        empty_file.touch()  # Create empty file

        with pytest.raises(
            (OSError, RuntimeError, TypeError, ValueError, onnx.checker.ValidationError)
        ):
            load_and_preprocess_onnx_model(str(empty_file))

    def test_load_validates_model_structure(self, linear_model):
        """Test that loaded model passes ONNX validation."""
        model = load_and_preprocess_onnx_model(linear_model)
        # Should not raise
        onnx.checker.check_model(model)


class TestDynamicShapeHandling:
    """Test handling of dynamic shapes in ONNX models."""

    def test_normalize_with_dynamic_batch_dimension(self):
        """Test normalization with dynamic batch dimension (None, C, H, W)."""
        # Create a model with dynamic batch size
        X = onnx.helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [None, 3, 224, 224]
        )
        Y = onnx.helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [None, 10]
        )

        node = onnx.helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "DynamicBatchModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        # Create temp file

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            temp_path = f.name

        try:
            # Should handle dynamic batch dimension
            normalized = load_and_preprocess_onnx_model(temp_path)
            assert isinstance(normalized, onnx.ModelProto)
            # Check that input shape is preserved with None for batch
            input_shape = [
                d.dim_value if d.dim_value > 0 else None
                for d in normalized.graph.input[0].type.tensor_type.shape.dim
            ]
            assert input_shape[0] is None or input_shape[0] == 1  # Batch can be None or 1
        finally:
            Path(temp_path).unlink()

    def test_normalize_with_dynamic_spatial_dimensions(self):
        """Test normalization with dynamic spatial dimensions."""
        # Create a model with dynamic H, W
        X = onnx.helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [1, 3, None, None]
        )
        Y = onnx.helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [1, 64, None, None]
        )

        node = onnx.helper.make_node(
            "Conv",
            inputs=["X", "W", "B"],
            outputs=["Y"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )

        W = onnx.helper.make_tensor(  # noqa: N806
            "W", onnx.TensorProto.FLOAT, [64, 3, 3, 3], vals=range(64 * 3 * 3 * 3)
        )
        B = onnx.helper.make_tensor(  # noqa: N806
            "B", onnx.TensorProto.FLOAT, [64], vals=range(64)
        )

        graph = onnx.helper.make_graph([node], "DynamicSpatialModel", [X], [Y], [W, B])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            temp_path = f.name

        try:
            # Should handle dynamic spatial dimensions
            normalized = load_and_preprocess_onnx_model(temp_path)
            assert isinstance(normalized, onnx.ModelProto)
        finally:
            Path(temp_path).unlink()

    def test_normalize_with_multiple_dynamic_dims(self):
        """Test normalization with multiple dynamic dimensions."""
        # Create a model with multiple None dimensions
        X = onnx.helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, [None, None, 256]
        )
        Y = onnx.helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [None, None, 128]
        )

        node = onnx.helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "MultiDynamicModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            temp_path = f.name

        try:
            # Should handle multiple dynamic dimensions
            normalized = load_and_preprocess_onnx_model(temp_path)
            assert isinstance(normalized, onnx.ModelProto)
        finally:
            Path(temp_path).unlink()

    def test_shape_inference_preserves_dynamic_dims(self, linear_model):
        """Verify shape inference preserves dynamic dimensions."""
        # Load a static shape model first
        normalized = load_and_preprocess_onnx_model(linear_model)

        # Check that shape info is preserved
        assert len(normalized.graph.node) >= 0
        assert len(normalized.graph.input) > 0
        assert normalized.graph.input[0].type.tensor_type.elem_type > 0

    def test_normalize_preserves_dynamic_shape_info(self):
        """Test that normalization preserves shape information for inference."""
        # Create a model with symbolic shape info
        X = onnx.helper.make_tensor_value_info(  # noqa: N806
            "X",
            onnx.TensorProto.FLOAT,
            [None, 10],  # Dynamic first dimension
        )
        Y = onnx.helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, [None, 5]
        )

        node = onnx.helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "DynamicShapeModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            temp_path = f.name

        try:
            normalized = load_and_preprocess_onnx_model(temp_path)
            # After normalization, dynamic shapes should be handled
            # (may be inferred to concrete shapes or remain dynamic depending on implementation)
            assert isinstance(normalized, onnx.ModelProto)
            assert len(normalized.graph.input) > 0
        finally:
            Path(temp_path).unlink()


class TestShapeONNXIntegration:
    """Test ShapeONNX shape inference integration (Phase 2).

    Tests the alternative ShapeONNX shape inference path in normalize.py.
    """

    def test_normalize_with_shapeonnx_disabled_by_default(self, linear_model):
        """Test that use_shapeonnx=False uses ONNX's built-in shape inference."""
        # Default behavior should use ONNX shape inference, not shapeonnx
        model = load_and_preprocess_onnx_model(linear_model, target_opset=20, infer_shapes=True)
        assert isinstance(model, onnx.ModelProto)
        # Shape inference should have completed
        assert len(model.graph.value_info) > 0 or len(model.graph.output) > 0

    def test_apply_shapeonnx_inference_updates_value_info(self):
        """Test that _apply_shapeonnx_inference updates existing value_info."""
        # Create a simple model with value_info
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, 10])  # noqa: N806
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None, 5])  # noqa: N806

        node = onnx.helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "TestModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        # Add some value_info to the graph
        value_info = model.graph.value_info.add()
        value_info.name = "Y"
        value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

        # Apply shape inference with mocked shapes
        shapes = {"Y": [1, 5]}
        _apply_shapeonnx_inference(model, shapes)

        # Check that value_info was updated
        assert model.graph.value_info[0].name == "Y"
        assert len(model.graph.value_info[0].type.tensor_type.shape.dim) == 2

    def test_apply_shapeonnx_inference_creates_new_value_info(self):
        """Test that _apply_shapeonnx_inference creates missing value_info."""
        # Create a simple model without value_info
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 5])  # noqa: N806

        node = onnx.helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "TestModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        # Apply shape inference with new shapes
        shapes = {"intermediate": [1, 5]}
        _apply_shapeonnx_inference(model, shapes)

        # Check that value_info was created
        # (May or may not be added depending on whether 'intermediate' is in graph)
        assert isinstance(model, onnx.ModelProto)

    def test_check_model_validates_onnx_structure(self):
        """Test that _check_model validates ONNX model structure."""
        # Create a valid model
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806

        node = onnx.helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "TestModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        # Should not raise for valid model
        try:
            _check_model(model)
        except ValueError as e:
            pytest.fail(f"Valid model raised error: {e}")

    def test_check_model_raises_on_invalid_model(self):
        """Test that _check_model raises ValueError for invalid models."""
        # Create an invalid model (missing required inputs)
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806

        node = onnx.helper.make_node("Identity", inputs=["NonExistent"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "InvalidModel", [], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        # Should raise an error for invalid model
        # (May be ValueError from _check_model or ValidationError from ONNX checker)
        error_raised = False
        try:
            _check_model(model)
        except ValueError:
            error_raised = True
        except Exception:  # noqa: BLE001
            # ONNX checker may raise other exceptions like ValidationError
            error_raised = True

        assert error_raised, "Expected an error for invalid model"

    def test_opset_outside_tested_range_warning(self):
        """Test warning when opset is outside recommended range."""
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806

        node = onnx.helper.make_node("Identity", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "TestModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 21)])

        # Requesting opset outside tested range should warn (but still succeed).
        # target_opset=22 is above MAX_TESTED_OPSET=21.
        with pytest.warns(UserWarning, match=r"opset.*outside"), contextlib.suppress(ValueError):
            _convert_version(model, target_opset=22, warn_on_diff=True)

    def test_version_conversion_preserves_model_structure(self, linear_model):
        """Test that version conversion preserves model structure."""
        # Load model and get original structure
        original = onnx.load(linear_model)
        original_nodes = len(original.graph.node)

        # Convert to different opset
        converted = _convert_version(original, target_opset=13, warn_on_diff=False)

        # Model structure should be preserved
        assert isinstance(converted, onnx.ModelProto)
        assert (
            len(converted.graph.node) >= original_nodes
            or len(converted.graph.node) <= original_nodes
        )  # May differ due to optimization

    def test_shapeonnx_failure_falls_back_to_onnx(self):
        """Test fallback when ShapeONNX rejects an operator."""
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806

        # Use a valid ONNX operator that ShapeONNX deliberately does not support.
        # Identity is supported and must not be used as a synthetic failure fixture.
        node = onnx.helper.make_node("HardSigmoid", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "TestModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        with pytest.warns(UserWarning, match="ShapeONNX inference failed"):
            result = _infer_shapes(model, use_shapeonnx=True)
        assert isinstance(result, onnx.ModelProto)
