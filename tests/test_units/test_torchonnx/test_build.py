"""Comprehensive tests for Stage 2: Structural IR Building.

This module tests the build stage which handles:
- Converting ONNX graph to structural IR
- Building computation graph with layers and operators
- Preserving input/output connections
- Capturing model parameters and initializers
- Handling complex graph structures

Test Coverage:
- TestBuildBasics: 5 tests - Basic model IR construction
- TestBuildComplexGraphs: 3 tests - Multi-input/output, branching
- TestBuildAttributes: 2 tests - Layer attributes and properties
- TestBuildEdgeCases: 2 tests - Skip connections, complex patterns
"""

from torchonnx.build import build_model_ir
from torchonnx.normalize import load_and_preprocess_onnx_model


class TestBuildBasics:
    """Test basic structural IR building."""

    def test_build_identity_ir(self, identity_model):
        """Test building structural IR from Identity model."""
        model = load_and_preprocess_onnx_model(identity_model)
        ir = build_model_ir(model)
        assert ir is not None
        assert len(ir.layers) == 1
        assert ir.layers[0].onnx_op_type == "Identity"

    def test_build_linear_ir(self, linear_model):
        """Test building structural IR from Linear model."""
        model = load_and_preprocess_onnx_model(linear_model)
        ir = build_model_ir(model)
        assert ir is not None
        assert len(ir.layers) == 1
        assert ir.layers[0].onnx_op_type == "Gemm"
        assert len(ir.layers[0].input_names) == 3

    def test_build_mlp_ir(self, mlp_model):
        """Test building structural IR from MLP model."""
        model = load_and_preprocess_onnx_model(mlp_model)
        ir = build_model_ir(model)
        assert ir is not None
        assert len(ir.layers) == 3
        assert ir.layers[0].onnx_op_type == "Gemm"
        assert ir.layers[1].onnx_op_type == "Relu"
        assert ir.layers[2].onnx_op_type == "Gemm"

    def test_ir_preserves_graph_structure(self, add_model):
        """Test that IR preserves input/output connections."""
        model = load_and_preprocess_onnx_model(add_model)
        ir = build_model_ir(model)
        assert ir.input_names == ["X", "Y"]
        assert ir.output_names == ["Z"]

    def test_build_ir_has_initializers(self, linear_model):
        """Test that IR correctly captures model parameters."""
        model = load_and_preprocess_onnx_model(linear_model)
        ir = build_model_ir(model)
        assert ir.initializers is not None
        assert len(ir.initializers) == 2


class TestBuildComplexGraphs:
    """Test building IR from complex graph structures."""

    def test_build_conv_model_ir(self, conv2d_model):
        """Test building structural IR from Conv2d model."""
        model = load_and_preprocess_onnx_model(conv2d_model)
        ir = build_model_ir(model)
        assert ir is not None
        # Should have Conv node
        assert any(layer.onnx_op_type == "Conv" for layer in ir.layers)
        # Should have input and output
        assert len(ir.input_names) > 0
        assert len(ir.output_names) > 0

    def test_build_multi_input_model_ir(self, multi_input_model):
        """Test building structural IR from multi-input model."""
        model = load_and_preprocess_onnx_model(multi_input_model)
        ir = build_model_ir(model)
        assert ir is not None
        # Should have multiple inputs
        assert len(ir.input_names) >= 2

    def test_build_multi_output_model_ir(self, multi_output_model):
        """Test building structural IR from multi-output model."""
        model = load_and_preprocess_onnx_model(multi_output_model)
        ir = build_model_ir(model)
        assert ir is not None
        # Should have multiple outputs
        assert len(ir.output_names) >= 2


class TestBuildAttributes:
    """Test that IR preserves layer attributes."""

    def test_ir_layer_has_required_attributes(self, linear_model):
        """Test that IR layers have required attributes."""
        model = load_and_preprocess_onnx_model(linear_model)
        ir = build_model_ir(model)
        layer = ir.layers[0]

        # All layers should have these attributes
        assert hasattr(layer, "onnx_op_type")
        assert hasattr(layer, "input_names")
        assert hasattr(layer, "output_names")
        assert layer.onnx_op_type is not None

    def test_ir_preserves_layer_inputs_outputs(self, add_model):
        """Test that IR preserves input/output names for each layer."""
        model = load_and_preprocess_onnx_model(add_model)
        ir = build_model_ir(model)
        layer = ir.layers[0]

        # Add layer should have 2 inputs and 1 output
        assert len(layer.input_names) == 2
        assert len(layer.output_names) == 1


class TestBuildEdgeCases:
    """Test building IR from complex and edge case models."""

    def test_build_batchnorm_model_ir(self, batchnorm_model):
        """Test building structural IR from BatchNorm model."""
        model = load_and_preprocess_onnx_model(batchnorm_model)
        ir = build_model_ir(model)
        assert ir is not None
        # Should have BatchNormalization node
        assert any(layer.onnx_op_type == "BatchNormalization" for layer in ir.layers)

    def test_build_resnet_block_ir(self, resnet_block_model):
        """Test building structural IR from ResNet block (skip connection)."""
        model = load_and_preprocess_onnx_model(resnet_block_model)
        ir = build_model_ir(model)
        assert ir is not None
        # Should have multiple layers for skip connection
        assert len(ir.layers) >= 2
        # Should preserve input/output mapping
        assert ir.input_names is not None
        assert ir.output_names is not None
