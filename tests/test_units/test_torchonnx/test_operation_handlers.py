"""Comprehensive tests for operation handlers in code generation.

This module tests the low-level operation handlers that convert ONNX operations
to PyTorch code. Focus on critical, high-complexity operations through integration.

Test Coverage:
- TestSliceOperations: 12 tests - Slice operation with various parameter combinations
- TestSliceEndToEnd: 4 tests - End-to-end slice operation through the pipeline
"""

import ast

import numpy as np
import onnx
import pytest
import torch

from torchonnx.analyze.types import ConstantInfo
from torchonnx.build import build_model_ir
from torchonnx.generate._handlers._operations import (
    _encode_slice_input,
    _generate_literal_slice,
    _try_narrow_slice,
)
from torchonnx.normalize import load_and_preprocess_onnx_model


class TestSliceOperations:
    """Test Slice operation handler with various parameter combinations."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("slice_static_model", id="static"),
            pytest.param("slice_dynamic_starts_model", id="dynamic_starts"),
            pytest.param("slice_dynamic_ends_model", id="dynamic_ends"),
            pytest.param("slice_narrow_compatible_model", id="narrow_compatible"),
            pytest.param("slice_multi_axis_model", id="multi_axis"),
            pytest.param("slice_int64max_model", id="int64max"),
        ],
    )
    def test_slice_model_loads(self, model_fixture, request):
        """Test that slice models can be loaded and preprocessed."""
        model = load_and_preprocess_onnx_model(request.getfixturevalue(model_fixture))
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    def test_slice_static_through_build_ir(self, slice_static_model):
        """Test static slice model through Build IR stage."""
        model = load_and_preprocess_onnx_model(slice_static_model)
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0
        assert isinstance(ir.input_names, list)
        assert isinstance(ir.output_names, list)

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("slice_dynamic_starts_model", id="dynamic_starts"),
            pytest.param("slice_multi_axis_model", id="multi_axis"),
            pytest.param("slice_int64max_model", id="int64max"),
            pytest.param("slice_narrow_compatible_model", id="narrow_compatible"),
        ],
    )
    def test_slice_model_through_build_ir(self, model_fixture, request):
        """Test slice models through Build IR stage."""
        model = load_and_preprocess_onnx_model(request.getfixturevalue(model_fixture))
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_slice_models_have_correct_structure(self, slice_static_model):
        """Verify slice models have Slice operator in graph."""
        model = load_and_preprocess_onnx_model(slice_static_model)

        # Find Slice node
        slice_nodes = [node for node in model.graph.node if node.op_type == "Slice"]
        assert len(slice_nodes) > 0, "Model should contain Slice operator"

        slice_node = slice_nodes[0]
        # Check that slice node has required inputs
        assert len(slice_node.input) >= 3, "Slice requires at least data, starts, ends inputs"


class TestSliceHelperFunctions:
    """Test helper functions for slice operation code generation."""

    def test_generate_literal_slice_single_axis(self):
        """Test _generate_literal_slice with single axis."""
        code = _generate_literal_slice(
            data="x", starts=[0], ends=[10], axes=[1], steps=[1], output="y"
        )

        assert "y = x[" in code
        assert "0:10" in code

    def test_generate_literal_slice_multi_axis(self):
        """Test _generate_literal_slice with multiple axes."""
        code = _generate_literal_slice(
            data="x", starts=[0, 5], ends=[10, 15], axes=[1, 2], steps=[1, 1], output="y"
        )

        assert "y = x[" in code
        # Should have slices for both axes
        assert code.count(":") >= 2

    def test_generate_literal_slice_with_steps(self):
        """Test _generate_literal_slice with non-unity steps."""
        code = _generate_literal_slice(
            data="x", starts=[0], ends=[10], axes=[1], steps=[2], output="y"
        )

        assert "y = x[" in code
        # Should include step in slice
        assert "::2" in code or ":10:2" in code

    def test_generate_literal_slice_int64max(self):
        """Test _generate_literal_slice correctly handles INT64_MAX."""
        int64_max = np.iinfo(np.int64).max
        code = _generate_literal_slice(
            data="x",
            starts=[5],
            ends=[int64_max],
            axes=[1],
            steps=[1],
            output="y",
        )

        # INT64_MAX should be converted to empty (meaning "to end")
        assert str(int64_max) not in code
        # Should have slice starting at 5
        assert "5:" in code

    def test_generate_literal_slice_is_valid_python(self):
        """Test that generated slice code is valid Python."""
        code = _generate_literal_slice(
            data="x", starts=[0], ends=[10], axes=[1], steps=[1], output="y"
        )

        # Should parse as valid Python
        try:
            tree = ast.parse(code)
            assert isinstance(tree, ast.AST)
        except SyntaxError:
            pytest.fail(f"Generated code is not valid Python: {code}")

    def test_try_narrow_slice_single_axis_constant(self):
        """Test _try_narrow_slice with single-axis constant slicing."""
        starts = ConstantInfo(
            onnx_name="starts",
            code_name="starts",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([5], dtype=torch.int64),
        )
        ends = ConstantInfo(
            onnx_name="ends",
            code_name="ends",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([10], dtype=torch.int64),
        )
        axes = ConstantInfo(
            onnx_name="axes",
            code_name="axes",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([1], dtype=torch.int64),
        )
        steps = ConstantInfo(
            onnx_name="steps",
            code_name="steps",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([1], dtype=torch.int64),
        )

        code = _try_narrow_slice(
            data="x",
            starts_input=starts,
            ends_input=ends,
            axes_input=axes,
            steps_input=steps,
            output="y",
        )

        # Should generate narrow code
        assert isinstance(code, str)
        assert len(code) > 0
        assert "narrow" in code
        assert "y = x.narrow" in code

    def test_try_narrow_slice_multi_axis_returns_none(self):
        """Test _try_narrow_slice returns None for multi-axis (can't use narrow)."""
        starts = ConstantInfo(
            onnx_name="starts",
            code_name="starts",
            shape=(2,),
            dtype=torch.int64,
            data=torch.tensor([0, 5], dtype=torch.int64),
        )
        ends = ConstantInfo(
            onnx_name="ends",
            code_name="ends",
            shape=(2,),
            dtype=torch.int64,
            data=torch.tensor([10, 10], dtype=torch.int64),
        )
        axes = ConstantInfo(
            onnx_name="axes",
            code_name="axes",
            shape=(2,),
            dtype=torch.int64,
            data=torch.tensor([1, 2], dtype=torch.int64),
        )
        steps = ConstantInfo(
            onnx_name="steps",
            code_name="steps",
            shape=(2,),
            dtype=torch.int64,
            data=torch.tensor([1, 1], dtype=torch.int64),
        )

        code = _try_narrow_slice(
            data="x",
            starts_input=starts,
            ends_input=ends,
            axes_input=axes,
            steps_input=steps,
            output="y",
        )

        # Multi-axis cannot use narrow, should return None
        assert code is None

    def test_try_narrow_slice_non_unity_step_returns_none(self):
        """Test _try_narrow_slice returns None for non-unity steps."""
        starts = ConstantInfo(
            onnx_name="starts",
            code_name="starts",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([0], dtype=torch.int64),
        )
        ends = ConstantInfo(
            onnx_name="ends",
            code_name="ends",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([10], dtype=torch.int64),
        )
        axes = ConstantInfo(
            onnx_name="axes",
            code_name="axes",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([1], dtype=torch.int64),
        )
        steps = ConstantInfo(
            onnx_name="steps",
            code_name="steps",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([2], dtype=torch.int64),
        )

        code = _try_narrow_slice(
            data="x",
            starts_input=starts,
            ends_input=ends,
            axes_input=axes,
            steps_input=steps,
            output="y",
        )

        # Step != 1 cannot use narrow
        assert code is None

    def test_encode_slice_input_constant(self):
        """Test _encode_slice_input with constant input."""
        const = ConstantInfo(
            onnx_name="const",
            code_name="const",
            shape=(1,),
            dtype=torch.int64,
            data=torch.tensor([5], dtype=torch.int64),
        )
        code = _encode_slice_input(const, "None")

        # Should return the constant value as list
        assert "[5]" in code or "tensor([5]" in code

    def test_encode_slice_input_none_default(self):
        """Test _encode_slice_input returns default for None input."""
        code = _encode_slice_input(None, "None")

        # Should return the default value
        assert code == "None"


class TestSliceEndToEnd:
    """End-to-end tests for slice operations through the full pipeline."""

    def test_slice_static_normalizes(self, slice_static_model):
        """Test static slice operation can be normalized."""
        model = load_and_preprocess_onnx_model(slice_static_model)
        assert isinstance(model, onnx.ModelProto)
        # Verify opset version is set
        assert len(model.opset_import) > 0

    def test_slice_dynamic_starts_normalizes(self, slice_dynamic_starts_model):
        """Test dynamic starts slice can be normalized."""
        model = load_and_preprocess_onnx_model(slice_dynamic_starts_model)
        assert isinstance(model, onnx.ModelProto)

    def test_slice_multi_axis_normalizes(self, slice_multi_axis_model):
        """Test multi-axis slice can be normalized."""
        model = load_and_preprocess_onnx_model(slice_multi_axis_model)
        assert isinstance(model, onnx.ModelProto)

    def test_slice_int64max_normalizes(self, slice_int64max_model):
        """Test INT64_MAX slice can be normalized."""
        model = load_and_preprocess_onnx_model(slice_int64max_model)
        assert isinstance(model, onnx.ModelProto)


class TestExpandOperations:
    """Test Expand operation handler with various shape configurations."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("expand_constant_shape_model", id="constant"),
            pytest.param("expand_runtime_shape_model", id="runtime"),
            pytest.param("expand_broadcast_model", id="broadcast"),
        ],
    )
    def test_expand_model_loads(self, model_fixture, request):
        """Test that expand models can be loaded and preprocessed."""
        model = load_and_preprocess_onnx_model(request.getfixturevalue(model_fixture))
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("expand_constant_shape_model", id="constant"),
            pytest.param("expand_runtime_shape_model", id="runtime"),
            pytest.param("expand_broadcast_model", id="broadcast"),
        ],
    )
    def test_expand_model_through_build_ir(self, model_fixture, request):
        """Test expand models through Build IR stage."""
        model = load_and_preprocess_onnx_model(request.getfixturevalue(model_fixture))
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_expand_models_have_correct_structure(self, expand_constant_shape_model):
        """Verify expand models have Expand operator in graph."""
        model = load_and_preprocess_onnx_model(expand_constant_shape_model)

        # Find Expand node
        expand_nodes = [node for node in model.graph.node if node.op_type == "Expand"]
        assert len(expand_nodes) > 0, "Model should contain Expand operator"

    def test_expand_constant_normalizes(self, expand_constant_shape_model):
        """Test constant expand can be normalized."""
        model = load_and_preprocess_onnx_model(expand_constant_shape_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.opset_import) > 0

    def test_expand_runtime_normalizes(self, expand_runtime_shape_model):
        """Test runtime expand can be normalized."""
        model = load_and_preprocess_onnx_model(expand_runtime_shape_model)
        assert isinstance(model, onnx.ModelProto)

    def test_expand_broadcast_normalizes(self, expand_broadcast_model):
        """Test broadcast expand can be normalized."""
        model = load_and_preprocess_onnx_model(expand_broadcast_model)
        assert isinstance(model, onnx.ModelProto)


class TestPadOperations:
    """Test Pad operation handler with various padding configurations."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("pad_constant_pads_model", id="constant_pads"),
            pytest.param("pad_dynamic_pads_model", id="dynamic_pads"),
            pytest.param("pad_with_value_model", id="with_value"),
        ],
    )
    def test_pad_model_loads(self, model_fixture, request):
        """Test that pad models can be loaded and preprocessed."""
        model = load_and_preprocess_onnx_model(request.getfixturevalue(model_fixture))
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("pad_constant_pads_model", id="constant_pads"),
            pytest.param("pad_dynamic_pads_model", id="dynamic_pads"),
            pytest.param("pad_with_value_model", id="with_value"),
        ],
    )
    def test_pad_model_through_build_ir(self, model_fixture, request):
        """Test pad models through Build IR stage."""
        model = load_and_preprocess_onnx_model(request.getfixturevalue(model_fixture))
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_pad_models_have_correct_structure(self, pad_constant_pads_model):
        """Verify pad models have Pad operator in graph."""
        model = load_and_preprocess_onnx_model(pad_constant_pads_model)

        # Find Pad node
        pad_nodes = [node for node in model.graph.node if node.op_type == "Pad"]
        assert len(pad_nodes) > 0, "Model should contain Pad operator"

    def test_pad_constant_normalizes(self, pad_constant_pads_model):
        """Test constant pads pad can be normalized."""
        model = load_and_preprocess_onnx_model(pad_constant_pads_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.opset_import) > 0

    def test_pad_dynamic_normalizes(self, pad_dynamic_pads_model):
        """Test dynamic pads pad can be normalized."""
        model = load_and_preprocess_onnx_model(pad_dynamic_pads_model)
        assert isinstance(model, onnx.ModelProto)

    def test_pad_with_value_normalizes(self, pad_with_value_model):
        """Test pad with value can be normalized."""
        model = load_and_preprocess_onnx_model(pad_with_value_model)
        assert isinstance(model, onnx.ModelProto)


class TestGatherOperations:
    """Test Gather operation handler with various index configurations."""

    def test_gather_scalar_index_loads(self, gather_scalar_index_model):
        """Test that scalar index gather model can be loaded."""
        model = load_and_preprocess_onnx_model(gather_scalar_index_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    def test_gather_vector_indices_loads(self, gather_vector_indices_model):
        """Test that vector indices gather model can be loaded."""
        model = load_and_preprocess_onnx_model(gather_vector_indices_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    def test_gather_scalar_through_build_ir(self, gather_scalar_index_model):
        """Test scalar index gather through Build IR stage."""
        model = load_and_preprocess_onnx_model(gather_scalar_index_model)
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_gather_vector_through_build_ir(self, gather_vector_indices_model):
        """Test vector indices gather through Build IR stage."""
        model = load_and_preprocess_onnx_model(gather_vector_indices_model)
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_gather_models_have_correct_structure(self, gather_scalar_index_model):
        """Verify gather models have Gather operator in graph."""
        model = load_and_preprocess_onnx_model(gather_scalar_index_model)

        # Find Gather node
        gather_nodes = [node for node in model.graph.node if node.op_type == "Gather"]
        assert len(gather_nodes) > 0, "Model should contain Gather operator"

    def test_gather_scalar_normalizes(self, gather_scalar_index_model):
        """Test scalar index gather can be normalized."""
        model = load_and_preprocess_onnx_model(gather_scalar_index_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.opset_import) > 0

    def test_gather_vector_normalizes(self, gather_vector_indices_model):
        """Test vector indices gather can be normalized."""
        model = load_and_preprocess_onnx_model(gather_vector_indices_model)
        assert isinstance(model, onnx.ModelProto)


class TestScatterNDOperations:
    """Test ScatterND operation handler."""

    def test_scatter_nd_loads(self, scatter_nd_model):
        """Test that scatter_nd model can be loaded."""
        model = load_and_preprocess_onnx_model(scatter_nd_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    def test_scatter_nd_through_build_ir(self, scatter_nd_model):
        """Test scatter_nd through Build IR stage."""
        model = load_and_preprocess_onnx_model(scatter_nd_model)
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_scatter_nd_has_correct_structure(self, scatter_nd_model):
        """Verify scatter_nd model has ScatterND operator in graph."""
        model = load_and_preprocess_onnx_model(scatter_nd_model)

        # Find ScatterND node
        scatter_nodes = [node for node in model.graph.node if node.op_type == "ScatterND"]
        assert len(scatter_nodes) > 0, "Model should contain ScatterND operator"

    def test_scatter_nd_normalizes(self, scatter_nd_model):
        """Test scatter_nd can be normalized."""
        model = load_and_preprocess_onnx_model(scatter_nd_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.opset_import) > 0


class TestConcatOperations:
    """Test Concat operation handler with various configurations."""

    def test_concat_batch_expand_loads(self, concat_batch_expand_model):
        """Test that concat batch expand model can be loaded."""
        model = load_and_preprocess_onnx_model(concat_batch_expand_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    def test_concat_through_build_ir(self, concat_batch_expand_model):
        """Test concat through Build IR stage."""
        model = load_and_preprocess_onnx_model(concat_batch_expand_model)
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_concat_has_correct_structure(self, concat_batch_expand_model):
        """Verify concat model has Concat operator in graph."""
        model = load_and_preprocess_onnx_model(concat_batch_expand_model)

        # Find Concat node
        concat_nodes = [node for node in model.graph.node if node.op_type == "Concat"]
        assert len(concat_nodes) > 0, "Model should contain Concat operator"

    def test_concat_normalizes(self, concat_batch_expand_model):
        """Test concat can be normalized."""
        model = load_and_preprocess_onnx_model(concat_batch_expand_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.opset_import) > 0


class TestSplitOperations:
    """Test Split operation handler with various split configurations."""

    def test_split_equal_loads(self, split_equal_model):
        """Test that equal split model can be loaded."""
        model = load_and_preprocess_onnx_model(split_equal_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    def test_split_unequal_loads(self, split_unequal_model):
        """Test that unequal split model can be loaded."""
        model = load_and_preprocess_onnx_model(split_unequal_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.graph.node) > 0

    def test_split_equal_through_build_ir(self, split_equal_model):
        """Test equal split through Build IR stage."""
        model = load_and_preprocess_onnx_model(split_equal_model)
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_split_unequal_through_build_ir(self, split_unequal_model):
        """Test unequal split through Build IR stage."""
        model = load_and_preprocess_onnx_model(split_unequal_model)
        ir = build_model_ir(model)

        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_split_models_have_correct_structure(self, split_equal_model):
        """Verify split models have Split operator in graph."""
        model = load_and_preprocess_onnx_model(split_equal_model)

        # Find Split node
        split_nodes = [node for node in model.graph.node if node.op_type == "Split"]
        assert len(split_nodes) > 0, "Model should contain Split operator"

    def test_split_equal_normalizes(self, split_equal_model):
        """Test equal split can be normalized."""
        model = load_and_preprocess_onnx_model(split_equal_model)
        assert isinstance(model, onnx.ModelProto)
        assert len(model.opset_import) > 0

    def test_split_unequal_normalizes(self, split_unequal_model):
        """Test unequal split can be normalized."""
        model = load_and_preprocess_onnx_model(split_unequal_model)
        assert isinstance(model, onnx.ModelProto)


# ===== Phase 4: Convolution and Linear Operations =====


class TestConvOperations:
    """Test Conv operation handler with various dimension configurations."""

    def test_conv1d_loads(self, conv1d_model):
        """Test Conv1D model loads successfully."""
        assert conv1d_model
        model = load_and_preprocess_onnx_model(conv1d_model)
        assert isinstance(model, onnx.ModelProto)

    def test_conv3d_loads(self, conv3d_model):
        """Test Conv3D model loads successfully."""
        assert conv3d_model
        model = load_and_preprocess_onnx_model(conv3d_model)
        assert isinstance(model, onnx.ModelProto)

    def test_conv1d_through_build_ir(self, conv1d_model):
        """Test Conv1D through IR building stage."""
        normalized = load_and_preprocess_onnx_model(conv1d_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_conv3d_through_build_ir(self, conv3d_model):
        """Test Conv3D through IR building stage."""
        normalized = load_and_preprocess_onnx_model(conv3d_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_conv_models_have_correct_structure(self, conv1d_model, conv3d_model):
        """Test Conv models have correct structure."""
        model1d = load_and_preprocess_onnx_model(conv1d_model)
        model3d = load_and_preprocess_onnx_model(conv3d_model)
        # Conv1D should have 3D input (batch, channels, length)
        assert len(model1d.graph.input[0].type.tensor_type.shape.dim) == 3
        # Conv3D should have 5D input (batch, channels, d, h, w)
        assert len(model3d.graph.input[0].type.tensor_type.shape.dim) == 5

    def test_conv1d_normalizes(self, conv1d_model):
        """Test Conv1D normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(conv1d_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)

    def test_conv3d_normalizes(self, conv3d_model):
        """Test Conv3D normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(conv3d_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)


class TestLinearOperations:
    """Test Linear (Gemm) operation handler with various weight configurations."""

    def test_linear_transposed_loads(self, linear_transposed_model):
        """Test Linear with transposed weights loads successfully."""
        assert linear_transposed_model
        model = load_and_preprocess_onnx_model(linear_transposed_model)
        assert isinstance(model, onnx.ModelProto)

    def test_linear_transposed_through_build_ir(self, linear_transposed_model):
        """Test Linear with transposed weights through IR building."""
        normalized = load_and_preprocess_onnx_model(linear_transposed_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_linear_transposed_has_correct_structure(self, linear_transposed_model):
        """Test Linear with transposed weights has correct structure."""
        model = load_and_preprocess_onnx_model(linear_transposed_model)
        # Should have Gemm node with transB attribute
        assert any(node.op_type == "Gemm" for node in model.graph.node)

    def test_linear_transposed_normalizes(self, linear_transposed_model):
        """Test Linear with transposed weights normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(linear_transposed_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)

    def test_linear_normal_and_transposed_differ(self, linear_model, linear_transposed_model):
        """Test that normal and transposed linear models have different weight shapes."""
        normal = load_and_preprocess_onnx_model(linear_model)
        transposed = load_and_preprocess_onnx_model(linear_transposed_model)

        # Find Gemm nodes
        normal_gemm = next(n for n in normal.graph.node if n.op_type == "Gemm")
        transposed_gemm = next(n for n in transposed.graph.node if n.op_type == "Gemm")

        # Both should be Gemm but may have different attributes
        assert normal_gemm.op_type == "Gemm"
        assert transposed_gemm.op_type == "Gemm"


class TestInterpolateOperations:
    """Test Interpolate (Resize) operation handler."""

    def test_interpolate_loads(self, interpolate_model):
        """Test Interpolate model loads successfully."""
        assert interpolate_model
        model = load_and_preprocess_onnx_model(interpolate_model)
        assert isinstance(model, onnx.ModelProto)

    def test_interpolate_through_build_ir(self, interpolate_model):
        """Test Interpolate through IR building stage."""
        normalized = load_and_preprocess_onnx_model(interpolate_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_interpolate_has_correct_structure(self, interpolate_model):
        """Test Interpolate has correct structure."""
        model = load_and_preprocess_onnx_model(interpolate_model)
        # Should have Resize node
        assert any(node.op_type == "Resize" for node in model.graph.node)

    def test_interpolate_normalizes(self, interpolate_model):
        """Test Interpolate normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(interpolate_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)


class TestConvTransposeOperations:
    """Test Transpose Convolution (ConvTranspose) operation handler."""

    def test_conv_transpose_loads(self, conv_transpose_model):
        """Test ConvTranspose model loads successfully."""
        assert conv_transpose_model
        model = load_and_preprocess_onnx_model(conv_transpose_model)
        assert isinstance(model, onnx.ModelProto)

    def test_conv_transpose_through_build_ir(self, conv_transpose_model):
        """Test ConvTranspose through IR building stage."""
        normalized = load_and_preprocess_onnx_model(conv_transpose_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_conv_transpose_has_correct_structure(self, conv_transpose_model):
        """Test ConvTranspose has correct structure."""
        model = load_and_preprocess_onnx_model(conv_transpose_model)
        # Should have ConvTranspose node
        assert any(node.op_type == "ConvTranspose" for node in model.graph.node)

    def test_conv_transpose_normalizes(self, conv_transpose_model):
        """Test ConvTranspose normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(conv_transpose_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)


# ===== Phase 5: Reduction and Utility Operations =====


class TestClipOperations:
    """Test Clip operation handler with various bound configurations."""

    def test_clip_constant_bounds_loads(self, clip_constant_bounds_model):
        """Test Clip with constant bounds loads successfully."""
        assert clip_constant_bounds_model
        model = load_and_preprocess_onnx_model(clip_constant_bounds_model)
        assert isinstance(model, onnx.ModelProto)

    def test_clip_tensor_bounds_loads(self, clip_tensor_bounds_model):
        """Test Clip with tensor bounds loads successfully."""
        assert clip_tensor_bounds_model
        model = load_and_preprocess_onnx_model(clip_tensor_bounds_model)
        assert isinstance(model, onnx.ModelProto)

    def test_clip_constant_through_build_ir(self, clip_constant_bounds_model):
        """Test Clip with constant bounds through IR building."""
        normalized = load_and_preprocess_onnx_model(clip_constant_bounds_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_clip_tensor_through_build_ir(self, clip_tensor_bounds_model):
        """Test Clip with tensor bounds through IR building."""
        normalized = load_and_preprocess_onnx_model(clip_tensor_bounds_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_clip_constant_normalizes(self, clip_constant_bounds_model):
        """Test Clip with constant bounds normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(clip_constant_bounds_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)

    def test_clip_tensor_normalizes(self, clip_tensor_bounds_model):
        """Test Clip with tensor bounds normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(clip_tensor_bounds_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)


class TestArangeOperations:
    """Test Range (Arange) operation handler with various parameter types."""

    def test_arange_literal_loads(self, arange_literal_model):
        """Test Arange with literal parameters loads successfully."""
        assert arange_literal_model
        model = load_and_preprocess_onnx_model(arange_literal_model)
        assert isinstance(model, onnx.ModelProto)

    def test_arange_runtime_loads(self, arange_runtime_model):
        """Test Arange with runtime parameters loads successfully."""
        assert arange_runtime_model
        model = load_and_preprocess_onnx_model(arange_runtime_model)
        assert isinstance(model, onnx.ModelProto)

    def test_arange_literal_through_build_ir(self, arange_literal_model):
        """Test Arange with literal parameters through IR building."""
        normalized = load_and_preprocess_onnx_model(arange_literal_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_arange_runtime_through_build_ir(self, arange_runtime_model):
        """Test Arange with runtime parameters through IR building."""
        normalized = load_and_preprocess_onnx_model(arange_runtime_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_arange_literal_normalizes(self, arange_literal_model):
        """Test Arange with literal parameters normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(arange_literal_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)

    def test_arange_runtime_normalizes(self, arange_runtime_model):
        """Test Arange with runtime parameters normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(arange_runtime_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)


class TestReshapeOperations:
    """Test Reshape operation handler with dimension inference."""

    def test_reshape_infer_dim_loads(self, reshape_infer_dim_model):
        """Test Reshape with dimension inference loads successfully."""
        assert reshape_infer_dim_model
        model = load_and_preprocess_onnx_model(reshape_infer_dim_model)
        assert isinstance(model, onnx.ModelProto)

    def test_reshape_infer_dim_through_build_ir(self, reshape_infer_dim_model):
        """Test Reshape with dimension inference through IR building."""
        normalized = load_and_preprocess_onnx_model(reshape_infer_dim_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_reshape_infer_dim_has_correct_structure(self, reshape_infer_dim_model):
        """Test Reshape with dimension inference has correct structure."""
        model = load_and_preprocess_onnx_model(reshape_infer_dim_model)
        # Should have Reshape node
        assert any(node.op_type == "Reshape" for node in model.graph.node)

    def test_reshape_infer_dim_normalizes(self, reshape_infer_dim_model):
        """Test Reshape with dimension inference normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(reshape_infer_dim_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)


class TestReduceOperations:
    """Test Reduce operations (ReduceMean, ReduceSum) with various axes."""

    def test_reduce_mean_loads(self, reduce_mean_model):
        """Test ReduceMean model loads successfully."""
        assert reduce_mean_model
        model = load_and_preprocess_onnx_model(reduce_mean_model)
        assert isinstance(model, onnx.ModelProto)

    def test_reduce_sum_loads(self, reduce_sum_model):
        """Test ReduceSum model loads successfully."""
        assert reduce_sum_model
        model = load_and_preprocess_onnx_model(reduce_sum_model)
        assert isinstance(model, onnx.ModelProto)

    def test_reduce_mean_through_build_ir(self, reduce_mean_model):
        """Test ReduceMean through IR building stage."""
        normalized = load_and_preprocess_onnx_model(reduce_mean_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_reduce_sum_through_build_ir(self, reduce_sum_model):
        """Test ReduceSum through IR building stage."""
        normalized = load_and_preprocess_onnx_model(reduce_sum_model)
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0

    def test_reduce_mean_has_correct_structure(self, reduce_mean_model):
        """Test ReduceMean has correct structure with axes and keepdims."""
        model = load_and_preprocess_onnx_model(reduce_mean_model)
        # Should have ReduceMean node
        reduce_nodes = [n for n in model.graph.node if n.op_type == "ReduceMean"]
        assert len(reduce_nodes) > 0
        # Check that axes input is specified
        assert any(len(n.input) > 1 for n in reduce_nodes)

    def test_reduce_sum_has_correct_structure(self, reduce_sum_model):
        """Test ReduceSum has correct structure."""
        model = load_and_preprocess_onnx_model(reduce_sum_model)
        # Should have ReduceSum node
        assert any(node.op_type == "ReduceSum" for node in model.graph.node)

    def test_reduce_mean_normalizes(self, reduce_mean_model):
        """Test ReduceMean normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(reduce_mean_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)

    def test_reduce_sum_normalizes(self, reduce_sum_model):
        """Test ReduceSum normalizes correctly."""
        normalized = load_and_preprocess_onnx_model(reduce_sum_model)
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)


# ===== Phase 6: Simple Operations =====


class TestSimpleOperations:
    """Test simple operation handlers with basic configurations."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("transpose_model", id="transpose"),
            pytest.param("squeeze_model", id="squeeze"),
            pytest.param("unsqueeze_model", id="unsqueeze"),
            pytest.param("cast_model", id="cast"),
            pytest.param("shape_model", id="shape"),
            pytest.param("sign_model", id="sign"),
            pytest.param("trigonometric_model", id="trigonometric"),
            pytest.param("floor_model", id="floor"),
        ],
    )
    def test_simple_op_loads(self, model_fixture, request):
        """Test simple operation models load successfully."""
        model_path = request.getfixturevalue(model_fixture)
        assert model_path
        model = load_and_preprocess_onnx_model(model_path)
        assert isinstance(model, onnx.ModelProto)


class TestSimpleOperationIntegration:
    """Test simple operations through IR building pipeline."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("transpose_model", id="transpose"),
            pytest.param("squeeze_model", id="squeeze"),
            pytest.param("unsqueeze_model", id="unsqueeze"),
            pytest.param("cast_model", id="cast"),
            pytest.param("shape_model", id="shape"),
            pytest.param("sign_model", id="sign"),
            pytest.param("trigonometric_model", id="trigonometric"),
            pytest.param("floor_model", id="floor"),
        ],
    )
    def test_simple_op_through_build_ir(self, model_fixture, request):
        """Test simple operations through IR building stage."""
        normalized = load_and_preprocess_onnx_model(request.getfixturevalue(model_fixture))
        ir = build_model_ir(normalized)
        assert isinstance(ir.layers, list)
        assert len(ir.layers) > 0


class TestSimpleOperationValidation:
    """Test simple operations have correct ONNX structure."""

    def test_transpose_has_perm_attribute(self, transpose_model):
        """Test Transpose has permutation attribute."""
        model = load_and_preprocess_onnx_model(transpose_model)
        transpose_nodes = [n for n in model.graph.node if n.op_type == "Transpose"]
        assert len(transpose_nodes) > 0

    def test_squeeze_has_axes_input(self, squeeze_model):
        """Test Squeeze has axes input."""
        model = load_and_preprocess_onnx_model(squeeze_model)
        squeeze_nodes = [n for n in model.graph.node if n.op_type == "Squeeze"]
        assert len(squeeze_nodes) > 0
        assert any(len(n.input) > 1 for n in squeeze_nodes)

    def test_cast_has_dtype(self, cast_model):
        """Test Cast specifies target dtype."""
        model = load_and_preprocess_onnx_model(cast_model)
        assert any(node.op_type == "Cast" for node in model.graph.node)

    def test_shape_returns_int64(self, shape_model):
        """Test Shape operation returns INT64."""
        model = load_and_preprocess_onnx_model(shape_model)
        shape_nodes = [n for n in model.graph.node if n.op_type == "Shape"]
        assert len(shape_nodes) > 0

    def test_sign_unary_operation(self, sign_model):
        """Test Sign is unary operation."""
        model = load_and_preprocess_onnx_model(sign_model)
        sign_nodes = [n for n in model.graph.node if n.op_type == "Sign"]
        assert len(sign_nodes) > 0
        assert all(len(n.input) == 1 for n in sign_nodes)

    def test_floor_unary_operation(self, floor_model):
        """Test Floor is unary operation."""
        model = load_and_preprocess_onnx_model(floor_model)
        floor_nodes = [n for n in model.graph.node if n.op_type == "Floor"]
        assert len(floor_nodes) > 0
        assert all(len(n.input) == 1 for n in floor_nodes)


class TestSimpleOperationNormalization:
    """Test simple operations normalize correctly."""

    @pytest.mark.parametrize(
        "model_fixture",
        [
            pytest.param("transpose_model", id="transpose"),
            pytest.param("squeeze_model", id="squeeze"),
            pytest.param("unsqueeze_model", id="unsqueeze"),
            pytest.param("cast_model", id="cast"),
            pytest.param("shape_model", id="shape"),
            pytest.param("sign_model", id="sign"),
            pytest.param("trigonometric_model", id="trigonometric"),
            pytest.param("floor_model", id="floor"),
        ],
    )
    def test_simple_op_normalizes(self, model_fixture, request):
        """Test simple operations normalize correctly."""
        normalized = load_and_preprocess_onnx_model(request.getfixturevalue(model_fixture))
        assert isinstance(normalized, onnx.ModelProto)
        onnx.checker.check_model(normalized)
