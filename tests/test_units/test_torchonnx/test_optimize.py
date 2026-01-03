"""Stage 4 (Optimize) Tests - IR Optimization and Pass-Through Validation.

This module tests the IR optimization stage:
- Semantic IR optimization (currently pass-through)
- Property preservation after optimization
- Deterministic behavior
- Pipeline integration

Currently the optimize stage is a pass-through implementation that returns the
input unchanged. These tests validate that behavior and provide a foundation for
future optimization implementations (constant folding, dead code elimination, etc).

Test Coverage:
- TestOptimizeBasics: 4 tests - Basic optimization functionality
- TestOptimizePreservation: 4 tests - Property preservation checks
"""

from torchonnx.analyze.builder import build_semantic_ir
from torchonnx.build import build_model_ir
from torchonnx.normalize import load_and_preprocess_onnx_model
from torchonnx.optimize import optimize_semantic_ir


class TestOptimizeBasics:
    """Test basic optimization functionality."""

    def test_optimize_semantic_ir_returns_semantic_ir(self, linear_model):
        """Verify optimize_semantic_ir returns a SemanticModelIR object."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        result = optimize_semantic_ir(semantic_ir)

        assert result is not None
        assert hasattr(result, "layers")
        assert hasattr(result, "variables")
        assert hasattr(result, "parameters")
        assert hasattr(result, "input_names")
        assert hasattr(result, "output_names")

    def test_optimize_semantic_ir_is_deterministic(self, linear_model):
        """Verify optimization is deterministic (same input → same output)."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        # Optimize multiple times
        result1 = optimize_semantic_ir(semantic_ir)
        result2 = optimize_semantic_ir(semantic_ir)

        # Results should be identical
        assert len(result1.layers) == len(result2.layers)
        assert len(result1.variables) == len(result2.variables)
        assert len(result1.parameters) == len(result2.parameters)
        assert result1.input_names == result2.input_names
        assert result1.output_names == result2.output_names

    def test_optimize_with_linear_model(self, linear_model):
        """Test optimization with linear model semantic IR."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        result = optimize_semantic_ir(semantic_ir)

        assert result is not None
        assert len(result.layers) >= 1
        assert len(result.input_names) == 1
        assert len(result.output_names) == 1

    def test_optimize_with_mlp_model(self, mlp_model):
        """Test optimization with multi-layer MLP semantic IR."""
        normalized = load_and_preprocess_onnx_model(mlp_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        result = optimize_semantic_ir(semantic_ir)

        assert result is not None
        assert len(result.layers) >= 2
        assert len(result.parameters) >= 2


class TestOptimizePreservation:
    """Test that optimization preserves semantic IR properties."""

    def test_optimize_semantic_ir_preserves_layers(self, linear_model):
        """Verify optimization preserves all layers."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        original_layer_count = len(semantic_ir.layers)
        result = optimize_semantic_ir(semantic_ir)

        assert len(result.layers) == original_layer_count

    def test_optimize_semantic_ir_preserves_variables(self, linear_model):
        """Verify optimization preserves all variables."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        original_var_count = len(semantic_ir.variables)
        result = optimize_semantic_ir(semantic_ir)

        assert len(result.variables) == original_var_count

    def test_optimize_semantic_ir_preserves_parameters(self, linear_model):
        """Verify optimization preserves all parameters."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        original_param_count = len(semantic_ir.parameters)
        result = optimize_semantic_ir(semantic_ir)

        assert len(result.parameters) == original_param_count

    def test_optimize_semantic_ir_preserves_input_output_mappings(self, mlp_model):
        """Verify optimization preserves input/output names and mappings."""
        normalized = load_and_preprocess_onnx_model(mlp_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        original_inputs = semantic_ir.input_names
        original_outputs = semantic_ir.output_names
        result = optimize_semantic_ir(semantic_ir)

        assert result.input_names == original_inputs
        assert result.output_names == original_outputs

    def test_optimize_preserves_layer_types(self, conv2d_model):
        """Verify optimization preserves layer type information."""
        normalized = load_and_preprocess_onnx_model(conv2d_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        original_types = [layer.pytorch_type for layer in semantic_ir.layers]
        result = optimize_semantic_ir(semantic_ir)
        result_types = [layer.pytorch_type for layer in result.layers]

        assert result_types == original_types

    def test_optimize_preserves_layer_names(self, mlp_model):
        """Verify optimization preserves layer names."""
        normalized = load_and_preprocess_onnx_model(mlp_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        original_names = [layer.name for layer in semantic_ir.layers]
        result = optimize_semantic_ir(semantic_ir)
        result_names = [layer.name for layer in result.layers]

        assert result_names == original_names
