"""Shared pytest configuration and fixtures for torchonnx unit tests.

This module provides:
- Model fixtures (40+ ONNX models for testing)
- Validation utilities
- Test helpers
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from tests.test_units.test_torchonnx.fixtures.synthetic_models import SyntheticONNXModels

# ===== Basic Model Fixtures (Migrated) =====


@pytest.fixture
def identity_model(tmp_path):
    """Create and save Identity ONNX model."""
    model = SyntheticONNXModels.create_identity_model()
    path = tmp_path / "identity.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def linear_model(tmp_path):
    """Create and save Linear ONNX model."""
    model = SyntheticONNXModels.create_linear_model()
    path = tmp_path / "linear.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def add_model(tmp_path):
    """Create and save Add ONNX model."""
    model = SyntheticONNXModels.create_add_model()
    path = tmp_path / "add.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def mlp_model(tmp_path):
    """Create and save 2-layer MLP ONNX model."""
    model = SyntheticONNXModels.create_mlp_model()
    path = tmp_path / "mlp.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Operator Model Fixtures (New - Phase 23) =====


@pytest.fixture
def sub_model(tmp_path):
    """Create and save Sub ONNX model."""
    model = SyntheticONNXModels.create_sub_model()
    path = tmp_path / "sub.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def div_model(tmp_path):
    """Create and save Div ONNX model."""
    model = SyntheticONNXModels.create_div_model()
    path = tmp_path / "div.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def pow_model(tmp_path):
    """Create and save Pow ONNX model."""
    model = SyntheticONNXModels.create_pow_model()
    path = tmp_path / "pow.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def neg_model(tmp_path):
    """Create and save Neg ONNX model."""
    model = SyntheticONNXModels.create_neg_model()
    path = tmp_path / "neg.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def equal_model(tmp_path):
    """Create and save Equal ONNX model."""
    model = SyntheticONNXModels.create_equal_model()
    path = tmp_path / "equal.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def add_with_vector_constant_model(tmp_path):
    """Create and save Add ONNX model with vector constant."""
    model = SyntheticONNXModels.create_add_with_vector_constant()
    path = tmp_path / "add_vector_constant.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def mul_with_scalar_constant_model(tmp_path):
    """Create and save Mul ONNX model with scalar constant."""
    model = SyntheticONNXModels.create_mul_with_scalar_constant()
    path = tmp_path / "mul_scalar_constant.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def chained_operators_model(tmp_path):
    """Create and save model with chained operators."""
    model = SyntheticONNXModels.create_chained_operators_model()
    path = tmp_path / "chained_operators.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Operation Handler Edge Cases (Phase 25) =====


@pytest.fixture
def reshape_with_shape_tensor_model(tmp_path):
    """Create and save Reshape model with shape tensor."""
    model = SyntheticONNXModels.create_reshape_with_shape_tensor()
    path = tmp_path / "reshape_shape_tensor.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def gather_with_axis_model(tmp_path):
    """Create and save Gather model with axis."""
    model = SyntheticONNXModels.create_gather_with_axis()
    path = tmp_path / "gather_axis.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def multi_concat_model(tmp_path):
    """Create and save multi-input Concat model."""
    model = SyntheticONNXModels.create_multi_concat_model()
    path = tmp_path / "multi_concat.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def reduce_with_keepdims_model(tmp_path):
    """Create and save ReduceMean model with keepdims."""
    model = SyntheticONNXModels.create_reduce_with_keepdims()
    path = tmp_path / "reduce_keepdims.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def expand_with_runtime_shape_model(tmp_path):
    """Create and save Expand model with runtime shape."""
    model = SyntheticONNXModels.create_expand_with_runtime_shape()
    path = tmp_path / "expand_runtime_shape.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def split_equal_parts_model(tmp_path):
    """Create and save Split model with equal parts."""
    model = SyntheticONNXModels.create_split_equal_parts_model()
    path = tmp_path / "split_equal.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Convolutional Model Fixtures (New) =====


@pytest.fixture
def conv2d_model(tmp_path):
    """Create and save Conv2d ONNX model."""
    model = SyntheticONNXModels.create_conv2d_model()
    path = tmp_path / "conv2d.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def batchnorm_model(tmp_path):
    """Create and save BatchNorm2d ONNX model."""
    model = SyntheticONNXModels.create_batchnorm_model()
    path = tmp_path / "batchnorm.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def batchnorm3d_model(tmp_path):
    """Create and save BatchNorm3d ONNX model."""
    model = SyntheticONNXModels.create_batchnorm_model(spatial_dims=3)
    path = tmp_path / "batchnorm3d.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Pooling Model Fixtures (New) =====


@pytest.fixture
def maxpool_model(tmp_path):
    """Create and save MaxPool2d ONNX model."""
    model = SyntheticONNXModels.create_maxpool_model()
    path = tmp_path / "maxpool.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def avgpool_model(tmp_path):
    """Create and save AvgPool ONNX model."""
    model = SyntheticONNXModels.create_avgpool_model()
    path = tmp_path / "avgpool.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Shape Operation Model Fixtures (New) =====


@pytest.fixture
def reshape_model(tmp_path):
    """Create and save Reshape ONNX model."""
    model = SyntheticONNXModels.create_reshape_model()
    path = tmp_path / "reshape.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def concat_model(tmp_path):
    """Create and save Concat ONNX model."""
    model = SyntheticONNXModels.create_concat_model()
    path = tmp_path / "concat.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def transpose_model(tmp_path):
    """Create and save Transpose ONNX model."""
    model = SyntheticONNXModels.create_transpose_model()
    path = tmp_path / "transpose.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Reduction Model Fixtures (New) =====


@pytest.fixture
def reduce_mean_model(tmp_path):
    """Create and save ReduceMean ONNX model."""
    model = SyntheticONNXModels.create_reduce_mean_model()
    path = tmp_path / "reduce_mean.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def reduce_sum_model(tmp_path):
    """Create and save ReduceSum ONNX model."""
    model = SyntheticONNXModels.create_reduce_sum_model()
    path = tmp_path / "reduce_sum.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Operator Model Fixtures (New) =====


@pytest.fixture
def arithmetic_model(tmp_path):
    """Create and save Arithmetic operations ONNX model."""
    model = SyntheticONNXModels.create_arithmetic_model()
    path = tmp_path / "arithmetic.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def matmul_model(tmp_path):
    """Create and save MatMul ONNX model."""
    model = SyntheticONNXModels.create_matmul_model()
    path = tmp_path / "matmul.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Complex Model Fixtures (New) =====


@pytest.fixture
def multi_input_model(tmp_path):
    """Create and save model with multiple inputs."""
    model = SyntheticONNXModels.create_multi_input_model()
    path = tmp_path / "multi_input.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def multi_output_model(tmp_path):
    """Create and save model with multiple outputs."""
    model = SyntheticONNXModels.create_multi_output_model()
    path = tmp_path / "multi_output.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def resnet_block_model(tmp_path):
    """Create and save simplified ResNet block model."""
    model = SyntheticONNXModels.create_resnet_block()
    path = tmp_path / "resnet_block.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Error Test Model Fixtures (New) =====


@pytest.fixture
def asymmetric_padding_model(tmp_path):
    """Create and save Conv2d with asymmetric padding (error test)."""
    model = SyntheticONNXModels.create_asymmetric_padding_model()
    path = tmp_path / "asymmetric_padding.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def unsupported_op_model(tmp_path):
    """Create and save model with unsupported operator."""
    model = SyntheticONNXModels.create_unsupported_op_model()
    path = tmp_path / "unsupported_op.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def constant_node_model(tmp_path):
    """Create and save model with Constant nodes."""
    model = SyntheticONNXModels.create_constant_node_model()
    path = tmp_path / "constant_node.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Phase 3: Additional Operator Model Fixtures =====


@pytest.fixture
def cast_model(tmp_path):
    """Create and save Cast operator model."""
    model = SyntheticONNXModels.create_cast_model()
    path = tmp_path / "cast.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def argmax_model(tmp_path):
    """Create and save ArgMax operator model."""
    model = SyntheticONNXModels.create_argmax_model()
    path = tmp_path / "argmax.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def convtranspose_model(tmp_path):
    """Create and save ConvTranspose operator model."""
    model = SyntheticONNXModels.create_convtranspose_model()
    path = tmp_path / "convtranspose.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def constantofshape_model(tmp_path):
    """Create and save ConstantOfShape operator model."""
    model = SyntheticONNXModels.create_constantofshape_model()
    path = tmp_path / "constantofshape.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Phase 1: Slice Operation Model Fixtures =====


@pytest.fixture
def slice_static_model(tmp_path):
    """Create and save Slice model with all static parameters."""
    model = SyntheticONNXModels.create_slice_static_model()
    path = tmp_path / "slice_static.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def slice_dynamic_starts_model(tmp_path):
    """Create and save Slice model with dynamic starts parameter."""
    model = SyntheticONNXModels.create_slice_dynamic_starts_model()
    path = tmp_path / "slice_dynamic_starts.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def slice_dynamic_ends_model(tmp_path):
    """Create and save Slice model with dynamic ends parameter."""
    model = SyntheticONNXModels.create_slice_dynamic_ends_model()
    path = tmp_path / "slice_dynamic_ends.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def slice_narrow_compatible_model(tmp_path):
    """Create and save Slice model optimizable to narrow operation."""
    model = SyntheticONNXModels.create_slice_narrow_compatible_model()
    path = tmp_path / "slice_narrow_compatible.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def slice_multi_axis_model(tmp_path):
    """Create and save Slice model slicing multiple axes."""
    model = SyntheticONNXModels.create_slice_multi_axis_model()
    path = tmp_path / "slice_multi_axis.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def slice_int64max_model(tmp_path):
    """Create and save Slice model with INT64_MAX end value."""
    model = SyntheticONNXModels.create_slice_int64max_model()
    path = tmp_path / "slice_int64max.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Phase 2: Expand & Pad Operation Model Fixtures =====


@pytest.fixture
def expand_constant_shape_model(tmp_path):
    """Create and save Expand model with constant output shape."""
    model = SyntheticONNXModels.create_expand_constant_shape_model()
    path = tmp_path / "expand_constant_shape.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def expand_runtime_shape_model(tmp_path):
    """Create and save Expand model with dynamic output shape."""
    model = SyntheticONNXModels.create_expand_runtime_shape_model()
    path = tmp_path / "expand_runtime_shape.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def expand_broadcast_model(tmp_path):
    """Create and save Expand model testing broadcasting semantics."""
    model = SyntheticONNXModels.create_expand_broadcast_model()
    path = tmp_path / "expand_broadcast.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def pad_constant_pads_model(tmp_path):
    """Create and save Pad model with constant padding values."""
    model = SyntheticONNXModels.create_pad_constant_pads_model()
    path = tmp_path / "pad_constant_pads.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def pad_dynamic_pads_model(tmp_path):
    """Create and save Pad model with dynamic padding values."""
    model = SyntheticONNXModels.create_pad_dynamic_pads_model()
    path = tmp_path / "pad_dynamic_pads.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def pad_with_value_model(tmp_path):
    """Create and save Pad model with non-zero pad value."""
    model = SyntheticONNXModels.create_pad_with_value_model()
    path = tmp_path / "pad_with_value.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Phase 3: Indexing Operation Model Fixtures =====


@pytest.fixture
def gather_scalar_index_model(tmp_path):
    """Create and save Gather model with scalar index."""
    model = SyntheticONNXModels.create_gather_scalar_index_model()
    path = tmp_path / "gather_scalar_index.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def gather_vector_indices_model(tmp_path):
    """Create and save Gather model with vector indices."""
    model = SyntheticONNXModels.create_gather_vector_indices_model()
    path = tmp_path / "gather_vector_indices.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def scatter_nd_model(tmp_path):
    """Create and save ScatterND model."""
    model = SyntheticONNXModels.create_scatter_nd_model()
    path = tmp_path / "scatter_nd.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def concat_batch_expand_model(tmp_path):
    """Create and save Concat model for batch expansion."""
    model = SyntheticONNXModels.create_concat_batch_expand_model()
    path = tmp_path / "concat_batch_expand.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def split_equal_model(tmp_path):
    """Create and save Split model with equal split sizes."""
    model = SyntheticONNXModels.create_split_equal_model()
    path = tmp_path / "split_equal.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def split_unequal_model(tmp_path):
    """Create and save Split model with unequal split sizes."""
    model = SyntheticONNXModels.create_split_unequal_model()
    path = tmp_path / "split_unequal.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Utility Fixtures =====


@pytest.fixture
def random_input_generator():
    """Generate random test inputs.

    Returns a function that creates random numpy arrays.
    """

    def _generate(shape, dtype=np.float32, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape).astype(dtype)

    return _generate


@pytest.fixture
def onnx_runner():
    """Create ONNX Runtime inference sessions.

    Returns a function that runs ONNX models.
    """

    def _run(model_path, inputs):
        """Run ONNX model with given inputs.

        :param model_path: Path to ONNX model file
        :param inputs: Dictionary of input names to numpy arrays
        :return: List of output arrays
        """
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return session.run(None, inputs)

    return _run


@pytest.fixture
def model_input_generator(onnx_runner):
    """Generate random inputs matching model requirements.

    Returns a function that inspects an ONNX model and generates appropriate inputs.
    """

    def _generate(model_path, seed=42):
        """Generate inputs for an ONNX model.

        :param model_path: Path to ONNX model file
        :param seed: Random seed
        :return: Dictionary of input names to numpy arrays
        """
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inputs = {}
        rng = np.random.default_rng(seed)

        for input_info in session.get_inputs():
            input_name = input_info.name
            input_shape = input_info.shape
            # Replace dynamic dimensions with actual values
            input_shape = [dim if isinstance(dim, int) else 1 for dim in input_shape]
            inputs[input_name] = rng.standard_normal(input_shape).astype(np.float32)

        return inputs

    return _generate


# ===== Code Validation Utilities =====


@pytest.fixture
def code_validator():
    """Validate generated PyTorch code.

    Returns a function that validates generated modules.
    """

    def _validate_syntax(code):
        """Validate Python syntax.

        :param code: Python code as string
        :return: True if valid, raises SyntaxError otherwise
        """
        compile(code, "<string>", "exec")
        return True

    def _import_module(py_path):
        """Import a Python module from file path.

        :param py_path: Path to Python file
        :return: Imported module
        """
        spec = importlib.util.spec_from_file_location("generated_model", py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from {py_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _instantiate_model(module):
        """Instantiate model class from module.

        :param module: Imported module
        :return: Model instance
        """
        if not hasattr(module, "__all__"):
            raise ValueError("Module must have __all__ defined")
        class_name = module.__all__[0]
        model_class = getattr(module, class_name)
        return model_class()

    return {
        "validate_syntax": _validate_syntax,
        "import_module": _import_module,
        "instantiate_model": _instantiate_model,
    }


@pytest.fixture
def numerical_validator():
    """Validate numerical equivalence between ONNX and PyTorch.

    Returns a function that compares outputs with configurable tolerances.
    """

    def _compare(onnx_output, pytorch_output, rtol=1e-5, atol=1e-6, name=""):
        """Compare ONNX and PyTorch outputs.

        :param onnx_output: Output from ONNX model
        :param pytorch_output: Output from PyTorch model
        :param rtol: Relative tolerance
        :param atol: Absolute tolerance
        :param name: Name of the test (for error messages)
        :return: True if outputs match within tolerance
        """
        if isinstance(pytorch_output, torch.Tensor):
            pytorch_output = pytorch_output.detach().numpy()

        try:
            np.testing.assert_allclose(
                pytorch_output,
                onnx_output,
                rtol=rtol,
                atol=atol,
                err_msg=f"Output mismatch for {name}",
            )
            return True
        except AssertionError as e:
            print(f"Numerical validation failed: {e}")
            print(f"Max difference: {np.max(np.abs(pytorch_output - onnx_output))}")
            raise

    return _compare


@pytest.fixture
def state_dict_validator():
    """Validate state_dict correctness.

    Returns a function that checks state_dict properties.
    """

    def _validate(model, state_dict):
        """Validate state_dict for a PyTorch model.

        :param model: PyTorch model
        :param state_dict: State dictionary loaded from file
        :return: True if valid, raises AssertionError otherwise
        """
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())

        assert model_keys == loaded_keys, f"State dict keys mismatch: {model_keys ^ loaded_keys}"

        # Check shapes
        for key in model_keys:
            model_shape = model.state_dict()[key].shape
            loaded_shape = state_dict[key].shape
            assert model_shape == loaded_shape, (
                f"Shape mismatch for {key}: {model_shape} vs {loaded_shape}"
            )

        # Check dtypes
        for key in model_keys:
            model_dtype = model.state_dict()[key].dtype
            loaded_dtype = state_dict[key].dtype
            assert model_dtype == loaded_dtype, (
                f"Dtype mismatch for {key}: {model_dtype} vs {loaded_dtype}"
            )

        return True

    return _validate


# ===== Model Compilation Utilities =====


@pytest.fixture
def model_executor():
    """Execute PyTorch models.

    Returns a function that runs models safely.
    """

    def _run(model, input_data, eval_mode=True):
        """Run a PyTorch model.

        :param model: PyTorch model
        :param input_data: Input tensor or dictionary of tensors
        :param eval_mode: If True, set model to eval mode
        :return: Model output
        """
        if eval_mode:
            model.eval()

        with torch.no_grad():
            if isinstance(input_data, dict):
                output = model(**input_data)
            else:
                output = model(input_data)

        return output

    return _run


@pytest.fixture
def temp_file_manager():
    """Manage temporary files and directories.

    Returns functions for creating and cleaning up temporary files.
    """
    temp_files = []

    def _create_temp_file(suffix=""):
        """Create a temporary file.

        :param suffix: File suffix
        :return: Path to temporary file
        """
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            temp_files.append(f.name)
            return f.name

    def _create_temp_dir():
        """Create a temporary directory.

        :return: Path to temporary directory
        """
        temp_dir = tempfile.mkdtemp()
        temp_files.append(temp_dir)
        return temp_dir

    def _cleanup():
        """Clean up all temporary files and directories."""
        from shutil import rmtree

        for path in temp_files:
            try:
                if Path(path).is_dir():
                    rmtree(path)
                else:
                    Path(path).unlink()
            except OSError:
                pass

    yield {"create_file": _create_temp_file, "create_dir": _create_temp_dir, "cleanup": _cleanup}

    _cleanup()


# ===== Phase 4: Convolution and Linear Operation Fixtures =====


@pytest.fixture
def conv1d_model(tmp_path):
    """Create and save 1D Convolution model."""
    model = SyntheticONNXModels.create_conv1d_model()
    path = tmp_path / "conv1d.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def conv3d_model(tmp_path):
    """Create and save 3D Convolution model."""
    model = SyntheticONNXModels.create_conv3d_model()
    path = tmp_path / "conv3d.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def linear_transposed_model(tmp_path):
    """Create and save Linear model with transposed weights."""
    model = SyntheticONNXModels.create_linear_transposed_model()
    path = tmp_path / "linear_transposed.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def interpolate_model(tmp_path):
    """Create and save Interpolate (Resize) model."""
    model = SyntheticONNXModels.create_interpolate_model()
    path = tmp_path / "interpolate.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def conv_transpose_model(tmp_path):
    """Create and save Transpose Convolution model."""
    model = SyntheticONNXModels.create_conv_transpose_model()
    path = tmp_path / "conv_transpose.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Phase 5: Reduction and Utility Operation Fixtures =====


@pytest.fixture
def clip_constant_bounds_model(tmp_path):
    """Create and save Clip model with constant bounds."""
    model = SyntheticONNXModels.create_clip_constant_bounds_model()
    path = tmp_path / "clip_constant_bounds.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def clip_tensor_bounds_model(tmp_path):
    """Create and save Clip model with tensor bounds."""
    model = SyntheticONNXModels.create_clip_tensor_bounds_model()
    path = tmp_path / "clip_tensor_bounds.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def arange_literal_model(tmp_path):
    """Create and save Range (Arange) model with literal parameters."""
    model = SyntheticONNXModels.create_arange_literal_model()
    path = tmp_path / "arange_literal.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def arange_runtime_model(tmp_path):
    """Create and save Range (Arange) model with runtime parameters."""
    model = SyntheticONNXModels.create_arange_runtime_model()
    path = tmp_path / "arange_runtime.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def reshape_infer_dim_model(tmp_path):
    """Create and save Reshape model with dimension inference."""
    model = SyntheticONNXModels.create_reshape_infer_dim_model()
    path = tmp_path / "reshape_infer_dim.onnx"
    onnx.save(model, str(path))
    return str(path)


# ===== Phase 6: Simple Operation Fixtures =====


@pytest.fixture
def squeeze_model(tmp_path):
    """Create and save Squeeze model."""
    model = SyntheticONNXModels.create_squeeze_model()
    path = tmp_path / "squeeze.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def unsqueeze_model(tmp_path):
    """Create and save Unsqueeze model."""
    model = SyntheticONNXModels.create_unsqueeze_model()
    path = tmp_path / "unsqueeze.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def shape_model(tmp_path):
    """Create and save Shape model."""
    model = SyntheticONNXModels.create_shape_model()
    path = tmp_path / "shape.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def sign_model(tmp_path):
    """Create and save Sign model."""
    model = SyntheticONNXModels.create_sign_model()
    path = tmp_path / "sign.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def trigonometric_model(tmp_path):
    """Create and save Trigonometric (Sin) model."""
    model = SyntheticONNXModels.create_trigonometric_model()
    path = tmp_path / "trigonometric.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def floor_model(tmp_path):
    """Create and save Floor model."""
    model = SyntheticONNXModels.create_floor_model()
    path = tmp_path / "floor.onnx"
    onnx.save(model, str(path))
    return str(path)


# ============================================================================
# PHASE 15: Pattern-Specific Models for Deep Analysis Testing
# ============================================================================


@pytest.fixture
def slice_with_add_pattern_model(tmp_path):
    """Create Slice model where ends = Add(starts, constant).

    This triggers code_generator._try_add_pattern_case pattern matching.

    Graph:
    - starts (input constant)
    - const_offset (constant)
    - ends = Add(starts, const_offset)
    - Slice(data, starts, ends)
    """
    # Input data tensor
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20])  # noqa: N806

    # Slice parameters
    starts = onnx.helper.make_tensor("starts", onnx.TensorProto.INT64, [1], vals=[0])
    const_offset = onnx.helper.make_tensor("const_offset", onnx.TensorProto.INT64, [1], vals=[10])
    axes = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])

    # ends = Add(starts, const_offset)
    add_node = onnx.helper.make_node(
        "Add",
        inputs=["starts", "const_offset"],
        outputs=["ends"],
    )

    # Slice using computed ends
    slice_node = onnx.helper.make_node(
        "Slice",
        inputs=["X", "starts", "ends", "axes"],
        outputs=["Y"],
    )

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806

    graph = onnx.helper.make_graph(
        [add_node, slice_node],
        "SliceWithAddPattern",
        [X],
        [Y],
        [starts, const_offset, axes],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "slice_add_pattern.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def slice_through_shape_ops_model(tmp_path):
    """Create Slice model where slice params are traced through shape ops.

    This triggers code_generator._find_producer_through_shape_ops.

    Graph:
    - starts_init (constant)
    - starts_unsqueezed = Unsqueeze(starts_init)
    - starts_squeezed = Squeeze(starts_unsqueezed)  (back to original shape)
    - ends = Add(starts_squeezed, 10)
    - Slice(X, starts_squeezed, ends)
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20])  # noqa: N806

    # Create constants
    starts_init = onnx.helper.make_tensor("starts_init", onnx.TensorProto.INT64, [1], vals=[0])
    const_10 = onnx.helper.make_tensor("const_10", onnx.TensorProto.INT64, [1], vals=[10])
    axes = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])
    unsqueeze_axes = onnx.helper.make_tensor(
        "unsqueeze_axes", onnx.TensorProto.INT64, [1], vals=[0]
    )
    squeeze_axes = onnx.helper.make_tensor("squeeze_axes", onnx.TensorProto.INT64, [1], vals=[0])

    # starts_unsqueezed = Unsqueeze(starts_init, unsqueeze_axes)
    unsqueeze_node = onnx.helper.make_node(
        "Unsqueeze",
        inputs=["starts_init", "unsqueeze_axes"],
        outputs=["starts_unsqueezed"],
    )

    # starts_squeezed = Squeeze(starts_unsqueezed, squeeze_axes)
    squeeze_node = onnx.helper.make_node(
        "Squeeze",
        inputs=["starts_unsqueezed", "squeeze_axes"],
        outputs=["starts_squeezed"],
    )

    # ends = Add(starts_squeezed, const_10)
    add_node = onnx.helper.make_node(
        "Add",
        inputs=["starts_squeezed", "const_10"],
        outputs=["ends"],
    )

    # Slice
    slice_node = onnx.helper.make_node(
        "Slice",
        inputs=["X", "starts_squeezed", "ends", "axes"],
        outputs=["Y"],
    )

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806

    graph = onnx.helper.make_graph(
        [unsqueeze_node, squeeze_node, add_node, slice_node],
        "SliceThroughShapeOps",
        [X],
        [Y],
        [starts_init, const_10, axes, unsqueeze_axes, squeeze_axes],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "slice_shape_ops.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def expand_through_computation_model(tmp_path):
    """Create Expand model where shape is computed through Add.

    Graph:
    - X (input)
    - base_shape (constant [2, 1])
    - offset (constant [0, 3])
    - target_shape = Add(base_shape, offset) = [2, 4]
    - Y = Expand(X, target_shape)
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 1])  # noqa: N806

    base_shape = onnx.helper.make_tensor("base_shape", onnx.TensorProto.INT64, [2], vals=[2, 1])
    offset = onnx.helper.make_tensor("offset", onnx.TensorProto.INT64, [2], vals=[0, 3])

    # target_shape = Add(base_shape, offset)
    add_node = onnx.helper.make_node(
        "Add",
        inputs=["base_shape", "offset"],
        outputs=["target_shape"],
    )

    # Y = Expand(X, target_shape)
    expand_node = onnx.helper.make_node(
        "Expand",
        inputs=["X", "target_shape"],
        outputs=["Y"],
    )

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 4])  # noqa: N806

    graph = onnx.helper.make_graph(
        [add_node, expand_node],
        "ExpandThroughComputation",
        [X],
        [Y],
        [base_shape, offset],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "expand_computation.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def scatter_nd_with_runtime_indices_model(tmp_path):
    """Create ScatterND model with runtime indices.

    Tests helper generation for ScatterND with dynamic indices.
    """
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [10, 10])
    indices = onnx.helper.make_tensor_value_info("indices", onnx.TensorProto.INT64, [2, 2])
    updates = onnx.helper.make_tensor_value_info("updates", onnx.TensorProto.FLOAT, [2])

    scatter_node = onnx.helper.make_node(
        "ScatterND",
        inputs=["data", "indices", "updates"],
        outputs=["output"],
    )

    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [10, 10])

    graph = onnx.helper.make_graph(
        [scatter_node],
        "ScatterNDRuntime",
        [data, indices, updates],
        [output],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "scatter_nd_runtime.onnx"
    onnx.save(model, str(path))
    return str(path)


# ============================================================================
# PHASE 17: Deep Edge Cases and Missing Branches
# ============================================================================


@pytest.fixture
def model_with_functional_ops(tmp_path):
    """Create model with F.* functional operations.

    Tests conditional import of torch.nn.functional (F).
    Includes Relu which is typically handled as F.relu.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3, 4])  # noqa: N806

    relu_node = onnx.helper.make_node(
        "Relu",
        inputs=["X"],
        outputs=["Y"],
    )

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2, 3, 4])  # noqa: N806

    graph = onnx.helper.make_graph(
        [relu_node],
        "FunctionalOps",
        [X],
        [Y],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "functional_ops.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def reshape_with_inferred_dim_model(tmp_path):
    """Create Reshape with -1 dimension (inferred).

    Tests _compute_inferred_dim for reshape dimension inference.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2, 3, 4])  # noqa: N806

    shape = onnx.helper.make_tensor("shape", onnx.TensorProto.INT64, [2], vals=[6, -1])

    reshape_node = onnx.helper.make_node(
        "Reshape",
        inputs=["X", "shape"],
        outputs=["Y"],
    )

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [6, 4])  # noqa: N806

    graph = onnx.helper.make_graph(
        [reshape_node],
        "ReshapeInferred",
        [X],
        [Y],
        [shape],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "reshape_inferred.onnx"
    onnx.save(model, str(path))
    return str(path)


# ============================================================================
# PHASE 18: Boundary Conditions and Numeric Edge Cases
# ============================================================================


@pytest.fixture
def expand_minimal_inputs_model(tmp_path):
    """Create Expand with minimal inputs (tests len(layer.inputs) < 2 edge case).

    Tests _check_expand_needs_helper edge case at line 194-195.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 4])  # noqa: N806

    # Expand with one input - will be caught at validation stage
    expand_node = onnx.helper.make_node(
        "Expand",
        inputs=["X"],
        outputs=["Y"],
    )

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3, 4])  # noqa: N806

    graph = onnx.helper.make_graph(
        [expand_node],
        "ExpandMinimal",
        [X],
        [Y],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "expand_minimal.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def slice_minimal_inputs_model(tmp_path):
    """Create Slice with minimal inputs (tests len(layer.inputs) < 3 edge case).

    Tests _check_slice_needs_helper edge case at line 118.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 20, 10])  # noqa: N806

    # Slice with only input tensor (no starts/ends) - caught at validation
    slice_node = onnx.helper.make_node(
        "Slice",
        inputs=["X"],
        outputs=["Y"],
    )

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 20, 10])  # noqa: N806

    graph = onnx.helper.make_graph(
        [slice_node],
        "SliceMinimal",
        [X],
        [Y],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "slice_minimal.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def expand_dynamic_shape_model(tmp_path):
    """Create Expand with dynamic output shape.

    Tests _check_expand_needs_helper with dynamic shape requiring helper.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1, 4])  # noqa: N806

    # Dynamic shape tensor for expand
    shape = onnx.helper.make_tensor_value_info("shape", onnx.TensorProto.INT64, [None])

    expand_node = onnx.helper.make_node(
        "Expand",
        inputs=["X", "shape"],
        outputs=["Y"],
    )

    # Output shape is dynamic (not fully known)
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None, None, 4])  # noqa: N806

    graph = onnx.helper.make_graph(
        [expand_node],
        "ExpandDynamicShape",
        [X, shape],
        [Y],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "expand_dynamic.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def slice_large_indices_model(tmp_path):
    """Create Slice with large indices (tests numeric boundary conditions).

    Tests handling of large index values approaching INT64_MAX.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 1000, 100])  # noqa: N806

    starts = onnx.helper.make_tensor("starts", onnx.TensorProto.INT64, [1], vals=[999])
    ends = onnx.helper.make_tensor("ends", onnx.TensorProto.INT64, [1], vals=[1000])
    axes = onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [1], vals=[1])

    slice_node = onnx.helper.make_node(
        "Slice",
        inputs=["X", "starts", "ends", "axes"],
        outputs=["Y"],
    )

    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 1, 100])  # noqa: N806

    graph = onnx.helper.make_graph(
        [slice_node],
        "SLiceLargeIndices",
        [X],
        [Y],
        [starts, ends, axes],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    path = tmp_path / "slice_large_indices.onnx"
    onnx.save(model, str(path))
    return str(path)
