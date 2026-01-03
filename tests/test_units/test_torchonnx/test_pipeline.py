"""Tests for pipeline integration and stage-to-stage communication.

This module tests how different stages of the compilation pipeline work together:
- Normalize → Build stage integration
- Build → Analyze stage (when semantic IR generation is fixed)
- Analyze → Generate stage (when semantic IR generation is fixed)
- Full pipeline end-to-end
- Main convert() API end-to-end conversion

Test Coverage:
- TestTorchONNXInitialization: 3 tests - Main converter class
- TestNormalizeToBuild: 3 tests - Stage 1 to Stage 2
- TestBuildProperties: 2 tests - IR property preservation
- TestPipelineErrorPropagation: 2 tests - Error handling across stages
- TestConvertAPI: 10 tests - Full convert() pipeline validation
"""

import importlib.util

import numpy as np
import onnx
import pytest
import torch

from torchonnx._torchonnx import TorchONNX
from torchonnx.build import build_model_ir
from torchonnx.normalize import load_and_preprocess_onnx_model


class TestTorchONNXInitialization:
    """Test TorchONNX main converter class."""

    def test_initialization_default(self):
        """Test TorchONNX initialization with defaults."""
        converter = TorchONNX()
        assert converter.verbose is False
        assert converter.use_shapeonnx is False

    def test_initialization_with_options(self):
        """Test TorchONNX initialization with options."""
        converter = TorchONNX(verbose=True, use_shapeonnx=True)
        assert converter.verbose is True
        assert converter.use_shapeonnx is True

    def test_initialization_options_independent(self):
        """Test that converter options don't interfere with each other."""
        c1 = TorchONNX(verbose=True, use_shapeonnx=False)
        c2 = TorchONNX(verbose=False, use_shapeonnx=True)
        assert c1.verbose is True
        assert c1.use_shapeonnx is False
        assert c2.verbose is False
        assert c2.use_shapeonnx is True


class TestNormalizeToBuild:
    """Test integration between Normalize and Build stages."""

    def test_normalize_to_build_pipeline(self, identity_model):
        """Test pipeline from normalize to build stages."""
        model = load_and_preprocess_onnx_model(identity_model)
        assert model is not None

        ir = build_model_ir(model)
        assert ir is not None
        assert len(ir.layers) == 1

    def test_build_stage_handles_multiple_layers(self, mlp_model):
        """Test build stage correctly structures multi-layer models."""
        model = load_and_preprocess_onnx_model(mlp_model)
        ir = build_model_ir(model)

        assert len(ir.layers) == 3
        assert ir.input_names
        assert ir.output_names

    def test_normalize_opset_affects_build(self, linear_model):
        """Test that normalize opset version is preserved in IR."""
        model = load_and_preprocess_onnx_model(linear_model, target_opset=20)
        ir = build_model_ir(model)

        assert ir is not None
        assert len(ir.layers) == 1


class TestBuildProperties:
    """Test that build stage preserves model properties."""

    def test_ir_preserves_model_properties(self, linear_model):
        """Test that IR preserves essential model properties."""
        model = load_and_preprocess_onnx_model(linear_model)
        ir = build_model_ir(model)

        assert ir.input_names == ["X"]
        assert ir.output_names == ["Y"]
        assert len(ir.layers) == 1

    def test_ir_preserves_initializers_for_multiple_layers(self, mlp_model):
        """Test that IR preserves initializers across multiple layers."""
        model = load_and_preprocess_onnx_model(mlp_model)
        ir = build_model_ir(model)

        # MLP should have parameters for 3 layers (2 Gemm + 1 for ReLU)
        assert ir.initializers is not None
        # Should have multiple initializers for multi-layer model
        assert len(ir.initializers) > 0


class TestPipelineErrorPropagation:
    """Test error handling across pipeline stages."""

    def test_normalize_error_blocks_pipeline(self, tmp_path):
        """Test that normalize errors prevent build from running."""
        nonexistent = tmp_path / "nonexistent.onnx"
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_onnx_model(str(nonexistent))

    def test_build_error_from_invalid_ir(self):
        """Test that build errors are raised for invalid input."""
        # Passing None to build_model_ir should raise AttributeError
        with pytest.raises((AttributeError, TypeError)):
            build_model_ir(None)


class TestConvertAPI:
    """Test the main convert() API end-to-end."""

    def test_convert_linear_model_full_pipeline(self, linear_model, tmp_path):
        """Test complete convert() pipeline with linear model."""
        converter = TorchONNX()
        output_py = tmp_path / "linear_converted.py"
        output_pth = tmp_path / "linear_converted.pth"

        converter.convert(
            linear_model,
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
        )

        assert output_py.exists()
        assert output_pth.exists()
        assert output_py.stat().st_size > 0
        assert output_pth.stat().st_size > 0

    def test_convert_returns_valid_code_and_state_dict(self, linear_model, tmp_path):
        """Verify convert() output format is valid."""
        converter = TorchONNX()
        output_py = tmp_path / "linear.py"
        output_pth = tmp_path / "linear.pth"

        converter.convert(
            linear_model,
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
        )

        # Verify Python code is valid
        code = output_py.read_text()
        assert "class " in code
        assert "def forward" in code
        assert "torch" in code

        # Verify state dict is valid
        state_dict = torch.load(output_pth)
        assert isinstance(state_dict, dict)

    def test_convert_with_module_name_option(self, linear_model, tmp_path):
        """Test convert() with custom module name via benchmark_name."""
        converter = TorchONNX()
        output_py = tmp_path / "test.py"
        output_pth = tmp_path / "test.pth"

        converter.convert(
            linear_model,
            benchmark_name="MyModel",
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
        )

        code = output_py.read_text()
        # Module name should include benchmark_name (CamelCased)
        assert "mymodel" in code.lower()
        assert output_py.exists()

    def test_convert_with_vmap_mode_enabled(self, linear_model, tmp_path):
        """Test convert() with vmap mode enabled."""
        converter = TorchONNX()
        output_py = tmp_path / "vmap.py"
        output_pth = tmp_path / "vmap.pth"

        converter.convert(
            linear_model,
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
            vmap_mode=True,
        )

        code = output_py.read_text()
        assert output_py.exists()
        # Verify code is generated (linear model may not have visible vmap differences)
        assert "class " in code
        assert "def forward" in code

    def test_convert_with_vmap_mode_disabled(self, linear_model, tmp_path):
        """Test convert() with vmap mode disabled."""
        converter = TorchONNX()
        output_py = tmp_path / "no_vmap.py"
        output_pth = tmp_path / "no_vmap.pth"

        converter.convert(
            linear_model,
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
            vmap_mode=False,
        )

        code = output_py.read_text()
        assert output_py.exists()
        # Non-vmap mode should use nn.Module layers
        assert "self." in code

    def test_convert_code_is_executable(self, linear_model, tmp_path):
        """Verify generated code can be imported and instantiated."""
        converter = TorchONNX()
        output_py = tmp_path / "executable.py"
        output_pth = tmp_path / "executable.pth"

        converter.convert(
            linear_model,
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
        )

        # Load the generated module
        spec = importlib.util.spec_from_file_location("generated_module", output_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the class (should be the only class in the module)
        classes = [
            getattr(module, name)
            for name in dir(module)
            if not name.startswith("_") and isinstance(getattr(module, name), type)
        ]
        assert len(classes) > 0, "No classes found in generated module"

        # Instantiate the class
        model_class = classes[0]
        model_instance = model_class()
        assert model_instance is not None

    def test_convert_numerical_accuracy(self, linear_model, tmp_path):
        """Compare numerical outputs between ONNX and generated PyTorch."""
        import onnxruntime as ort

        converter = TorchONNX()
        output_py = tmp_path / "numeric.py"
        output_pth = tmp_path / "numeric.pth"

        converter.convert(
            linear_model,
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
        )

        # Load generated module
        spec = importlib.util.spec_from_file_location("gen_module", output_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Instantiate and load state dict
        classes = [
            getattr(module, name)
            for name in dir(module)
            if not name.startswith("_") and isinstance(getattr(module, name), type)
        ]
        model = classes[0]()
        model.load_state_dict(torch.load(output_pth))
        model.eval()

        # Get input shape from ONNX model
        onnx_model = onnx.load(linear_model)
        input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        # Create test input with correct shape
        test_input = torch.randn(*input_shape, dtype=torch.float32)

        # ONNX inference
        onnx_session = ort.InferenceSession(linear_model)
        onnx_output = onnx_session.run(None, {"X": test_input.numpy()})[0]

        # PyTorch inference
        with torch.no_grad():
            torch_output = model(test_input).numpy()

        # Compare outputs (allow some numerical tolerance)
        np.testing.assert_allclose(onnx_output, torch_output, rtol=1e-4, atol=1e-5)

    def test_convert_state_dict_compatibility(self, linear_model, tmp_path):
        """Verify state dict is compatible with generated module."""
        converter = TorchONNX()
        output_py = tmp_path / "stdict.py"
        output_pth = tmp_path / "stdict.pth"

        converter.convert(
            linear_model,
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
        )

        # Load generated module
        spec = importlib.util.spec_from_file_location("stdict_module", output_py)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Instantiate model
        classes = [
            getattr(module, name)
            for name in dir(module)
            if not name.startswith("_") and isinstance(getattr(module, name), type)
        ]
        model = classes[0]()

        # Load state dict
        state_dict = torch.load(output_pth)
        model.load_state_dict(state_dict)

        # Verify all parameters were loaded
        assert len(model.state_dict()) > 0

    def test_convert_with_mlp_model(self, mlp_model, tmp_path):
        """Test convert() with multi-layer MLP model."""
        converter = TorchONNX()
        output_py = tmp_path / "mlp.py"
        output_pth = tmp_path / "mlp.pth"

        converter.convert(
            mlp_model,
            target_py_path=str(output_py),
            target_pth_path=str(output_pth),
        )

        code = output_py.read_text()
        assert output_py.exists()
        assert output_pth.exists()
        assert "class " in code
        assert "def forward" in code
