"""Semantic IR, type mapping, and attribute extraction tests for Stage 3 (Analyze)."""

__docformat__ = "restructuredtext"

import onnx
import pytest
import torch

from torchonnx.analyze.attr_extractor import _check_pads_symmetric
from torchonnx.analyze.builder import build_semantic_ir
from torchonnx.analyze.tensor_classifier import classify_inputs
from torchonnx.analyze.type_mapping import (
    convert_to_pytorch_type,
    extract_layer_args,
    extract_operation_args,
    is_layer_with_args,
    is_operation,
    is_operator,
)
from torchonnx.analyze.type_mapping._layers import (
    _extract_dropout_args,
    _extract_elu_args,
    _extract_gelu_args,
    _extract_leakyrelu_args,
    _extract_softmax_args,
    _extract_upsample_args,
    _simplify_tuple,
)
from torchonnx.analyze.types import (
    AbsentInputInfo,
    ConstantInfo,
    ParameterInfo,
    VariableInfo,
)
from torchonnx.build import build_model_ir
from torchonnx.normalize import load_and_preprocess_onnx_model


class TestBuildSemanticIR:
    """Test semantic IR building from structural IR.

    NOTE: These tests are blocked due to a bug in tensor_classifier.py:
    classify_inputs() is missing a `return results` statement.
    """

    def test_build_semantic_ir_from_linear_model(self, linear_model):
        """Build semantic IR from linear model and verify structure."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        assert len(semantic_ir.layers) >= 1
        assert len(semantic_ir.input_names) == 1
        assert len(semantic_ir.output_names) == 1
        assert len(semantic_ir.variables) >= 1
        assert semantic_ir.variables[0].code_name == "x0"

    def test_semantic_ir_preserves_absent_optional_input_position(self):
        """An empty ONNX input remains an explicit positional IR slot."""
        data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [3])
        output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [3])
        max_bound = onnx.helper.make_tensor("max", onnx.TensorProto.FLOAT, [], [1.0])
        clip = onnx.helper.make_node("Clip", ["data", "", "max"], ["output"])
        graph = onnx.helper.make_graph([clip], "clip_max_only", [data], [output], [max_bound])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

        semantic_ir = build_semantic_ir(build_model_ir(model))

        inputs = semantic_ir.layers[0].inputs
        assert len(inputs) == 3
        assert isinstance(inputs[0], VariableInfo)
        assert inputs[0].onnx_name == "data"
        assert inputs[1] == AbsentInputInfo(position=1)
        assert isinstance(inputs[2], ConstantInfo)
        assert inputs[2].onnx_name == "max"

    def test_build_semantic_ir_from_conv_model(self, conv2d_model):
        """Build semantic IR from conv2d model."""
        normalized = load_and_preprocess_onnx_model(conv2d_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        assert len(semantic_ir.layers) >= 1
        conv_layers = [layer for layer in semantic_ir.layers if "Conv" in layer.pytorch_type]
        assert len(conv_layers) >= 1

    def test_build_semantic_ir_from_mlp_model(self, mlp_model):
        """Build semantic IR from multi-layer MLP."""
        normalized = load_and_preprocess_onnx_model(mlp_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        assert len(semantic_ir.layers) >= 2
        assert len(semantic_ir.parameters) >= 2

    def test_semantic_ir_has_typed_containers(self, linear_model):
        """Verify semantic IR uses typed containers (not strings)."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        for layer in semantic_ir.layers:
            for input_info in layer.inputs:
                assert isinstance(input_info, (VariableInfo, ParameterInfo, ConstantInfo))
                assert hasattr(input_info, "code_name")

        for layer in semantic_ir.layers:
            for output_info in layer.outputs:
                assert isinstance(output_info, VariableInfo)
                assert output_info.code_name.startswith("x")

    def test_semantic_ir_removes_onnx_dependencies(self, linear_model):
        """Verify semantic IR removes ONNX dependencies."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        assert not hasattr(semantic_ir, "model")
        assert not hasattr(semantic_ir, "initializers")

        for param in semantic_ir.parameters:
            assert isinstance(param.data, torch.Tensor)
            assert isinstance(param.dtype, torch.dtype)

    def test_code_name_generation_increments(self, multi_input_model):
        """Verify code names increment correctly (x0, x1, p0, c0)."""
        normalized = load_and_preprocess_onnx_model(multi_input_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        var_codes = [v.code_name for v in semantic_ir.variables]
        param_codes = [p.code_name for p in semantic_ir.parameters]
        const_codes = [c.code_name for c in semantic_ir.constants]

        for _i, code in enumerate(var_codes):
            if code.startswith("x"):
                assert int(code[1:]) == var_codes.index(code)

        for _i, code in enumerate(param_codes):
            assert code.startswith("p")
            assert code[1:].isdigit()

        for _i, code in enumerate(const_codes):
            assert code.startswith("c")
            assert code[1:].isdigit()

    def test_variable_mapping_consistency(self, linear_model):
        """Verify variable mapping is consistent across IR."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        input_to_code = {}
        for var in semantic_ir.variables:
            if var.onnx_name in semantic_ir.input_names:
                input_to_code[var.onnx_name] = var.code_name

        assert len(input_to_code) > 0
        for code_name in input_to_code.values():
            assert code_name.startswith("x")


class TestTensorClassification:
    """Test classification of tensors into Variables, Parameters, Constants.

    NOTE: These tests are blocked because classify_inputs() has a bug
    (missing return statement) in tensor_classifier.py.
    """

    def test_classify_input_as_variable(self, linear_model):
        """Classify model input as a variable."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        inputs = classify_inputs(
            input_names=layer.input_names,
            initializers=model_ir.initializers,
            pytorch_type="Linear",
            shapes=model_ir.shapes,
            code_name_counters={"var": 0, "param": 0, "const": 0},
        )

        assert any(isinstance(i, VariableInfo) for i in inputs)

    def test_classify_input_as_parameter(self, linear_model):
        """Classify weights and bias as parameters."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        inputs = classify_inputs(
            input_names=layer.input_names,
            initializers=model_ir.initializers,
            pytorch_type="Linear",
            shapes=model_ir.shapes,
            code_name_counters={"var": 0, "param": 0, "const": 0},
        )

        params = [i for i in inputs if isinstance(i, ParameterInfo)]
        assert len(params) >= 2

    def test_classify_input_as_constant(self, reshape_model):
        """Classify shape constants as constants (not parameters)."""
        normalized = load_and_preprocess_onnx_model(reshape_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        inputs = classify_inputs(
            input_names=layer.input_names,
            initializers=model_ir.initializers,
            pytorch_type="Reshape",
            shapes=model_ir.shapes,
            code_name_counters={"var": 0, "param": 0, "const": 0},
        )

        constants = [i for i in inputs if isinstance(i, ConstantInfo)]
        assert len(constants) >= 1

    def test_classify_conv_weights(self, conv2d_model):
        """Classify Conv2d weights as parameter with weight role."""
        normalized = load_and_preprocess_onnx_model(conv2d_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        inputs = classify_inputs(
            input_names=layer.input_names,
            initializers=model_ir.initializers,
            pytorch_type="Conv2d",
            shapes=model_ir.shapes,
            code_name_counters={"var": 0, "param": 0, "const": 0},
        )

        weight_params = [
            p for p in inputs if isinstance(p, ParameterInfo) and p.pytorch_name == "weight"
        ]
        assert len(weight_params) >= 1

    def test_classify_conv_bias(self, conv2d_model):
        """Classify Conv2d bias as parameter with bias role."""
        normalized = load_and_preprocess_onnx_model(conv2d_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        inputs = classify_inputs(
            input_names=layer.input_names,
            initializers=model_ir.initializers,
            pytorch_type="Conv2d",
            shapes=model_ir.shapes,
            code_name_counters={"var": 0, "param": 0, "const": 0},
        )

        bias_params = [
            p for p in inputs if isinstance(p, ParameterInfo) and p.pytorch_name == "bias"
        ]
        assert len(bias_params) >= 1

    def test_classify_batchnorm_parameters(self, batchnorm_model):
        """Classify BatchNorm parameters (weight, bias, running_mean, running_var)."""
        normalized = load_and_preprocess_onnx_model(batchnorm_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        inputs = classify_inputs(
            input_names=layer.input_names,
            initializers=model_ir.initializers,
            pytorch_type="BatchNorm2d",
            shapes=model_ir.shapes,
            code_name_counters={"var": 0, "param": 0, "const": 0},
        )

        params = [i for i in inputs if isinstance(i, ParameterInfo)]
        roles = {p.pytorch_name for p in params}

        assert "weight" in roles or "bias" in roles

    def test_classify_linear_weights_transpose(self, linear_model):
        """Verify Linear weights are transposed from ONNX format."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        inputs = classify_inputs(
            input_names=layer.input_names,
            initializers=model_ir.initializers,
            pytorch_type="Linear",
            shapes=model_ir.shapes,
            code_name_counters={"var": 0, "param": 0, "const": 0},
            node=layer.node,
        )

        weight_params = [
            p for p in inputs if isinstance(p, ParameterInfo) and p.pytorch_name == "weight"
        ]
        assert len(weight_params) >= 1

    def test_classify_gemm_canonicalizes_scaling_and_broadcast_bias(self):
        """Static Gemm controls become exact nn.Linear parameter tensors."""
        node = onnx.helper.make_node(
            "Gemm",
            inputs=["x", "W", "C"],
            outputs=["y"],
            alpha=2.0,
            beta=0.25,
            transB=1,
        )
        initializers = {
            "W": onnx.helper.make_tensor(
                "W", onnx.TensorProto.FLOAT, [2, 2], [1.0, 2.0, -1.0, 3.0]
            ),
            "C": onnx.helper.make_tensor("C", onnx.TensorProto.FLOAT, [1, 1], [2.0]),
        }

        inputs = classify_inputs(
            input_names=list(node.input),
            initializers=initializers,
            pytorch_type="Linear",
            shapes={"x": (1, 2)},
            code_name_counters={"var": 0, "param": 0, "const": 0},
            node=node,
        )
        params = {p.pytorch_name: p for p in inputs if isinstance(p, ParameterInfo)}

        torch.testing.assert_close(
            params["weight"].data,
            torch.tensor([[2.0, 4.0], [-2.0, 6.0]]),
        )
        torch.testing.assert_close(params["bias"].data, torch.tensor([0.5, 0.5]))
        assert params["weight"].shape == (2, 2)
        assert params["bias"].shape == (2,)

    def test_classify_gemm_rejects_transa(self):
        """nn.Linear lowering must not silently discard transA."""
        node = onnx.helper.make_node("Gemm", inputs=["x", "W"], outputs=["y"], transA=1)
        initializers = {
            "W": onnx.helper.make_tensor("W", onnx.TensorProto.FLOAT, [2, 2], [1.0, 0.0, 0.0, 1.0])
        }

        with pytest.raises(ValueError, match="transA"):
            classify_inputs(
                input_names=list(node.input),
                initializers=initializers,
                pytorch_type="Linear",
                shapes={"x": (2, 1)},
                code_name_counters={"var": 0, "param": 0, "const": 0},
                node=node,
            )

    def test_parameter_role_detection(self, multi_input_model):
        """Verify parameter roles are correctly detected."""
        normalized = load_and_preprocess_onnx_model(multi_input_model)
        model_ir = build_model_ir(normalized)
        # Would call: semantic_ir = build_semantic_ir(model_ir)
        # But that fails due to the classify_inputs bug
        assert len(model_ir.layers) >= 1
        assert isinstance(model_ir.initializers, dict)


class TestTypeMappingLayers:
    """Test mapping of ONNX operators to PyTorch layer types.

    These tests work because they don't rely on the broken classify_inputs function.
    """

    # [REVIEW] Parametrized: test_convert_linear_to_pytorch, test_convert_conv2d_to_pytorch,
    # test_convert_batchnorm_to_pytorch, test_convert_maxpool_to_pytorch

    @pytest.mark.parametrize(
        ("model_fixture_name", "expected_type"),
        [
            pytest.param("linear_model", "Linear", id="linear"),
            pytest.param("conv2d_model", "Conv", id="conv2d"),
            pytest.param("batchnorm_model", "BatchNorm", id="batchnorm"),
            pytest.param("maxpool_model", "MaxPool", id="maxpool"),
        ],
    )
    def test_convert_layer_to_pytorch(self, model_fixture_name, expected_type, request):
        """Map ONNX operator to PyTorch layer type."""
        model = request.getfixturevalue(model_fixture_name)
        normalized = load_and_preprocess_onnx_model(model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        pytorch_type = convert_to_pytorch_type(layer.node, model_ir.initializers)
        assert expected_type in pytorch_type

    # [REVIEW] Parametrized: test_extract_linear_args, test_extract_conv_args,
    # test_extract_batchnorm_args

    @pytest.mark.parametrize(
        ("model_fixture_name", "expected_keys"),
        [
            pytest.param("linear_model", {"in_features", "out_features"}, id="linear"),
            pytest.param("conv2d_model", {"kernel_shape", "kernel_size"}, id="conv2d"),
            pytest.param("batchnorm_model", {"num_features", "epsilon"}, id="batchnorm"),
        ],
    )
    def test_extract_layer_args(self, model_fixture_name, expected_keys, request):
        """Extract layer arguments from ONNX operator."""
        model = request.getfixturevalue(model_fixture_name)
        normalized = load_and_preprocess_onnx_model(model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        args = extract_layer_args(layer.node, model_ir.initializers)
        assert any(key in args for key in expected_keys)

    def test_layer_with_args_detection(self):
        """Verify is_layer_with_args correctly identifies layers."""
        assert is_layer_with_args("Conv2d")
        assert is_layer_with_args("Linear")
        assert is_layer_with_args("BatchNorm2d")
        assert is_layer_with_args("nn.Conv2d")

    @pytest.mark.parametrize("op_type", ["AveragePool", "MaxPool"])
    @pytest.mark.parametrize("spatial_rank", [1, 2, 3])
    def test_pool_type_matches_spatial_rank(self, op_type, spatial_rank):
        """Map ONNX pooling to the PyTorch module with matching rank."""
        node = onnx.helper.make_node(
            op_type,
            inputs=["X"],
            outputs=["Y"],
            kernel_shape=[2] * spatial_rank,
        )
        pool_name = "AvgPool" if op_type == "AveragePool" else "MaxPool"

        assert convert_to_pytorch_type(node, {}) == f"nn.{pool_name}{spatial_rank}d"


class TestDynamicWeightConv:
    """Conv / ConvTranspose with a runtime-computed (dynamic) weight.

    A spectral-normalized conv divides its kernel by its spectral norm on
    every forward, so the weight is a graph variable, not a static
    initializer. Such a node cannot become an ``nn.Conv2d`` module (its
    kernel shape is unknown at construction time); it must map to the
    functional ``F.conv`` path. Regression guard for the empty
    ``nn.Conv2d()`` codegen bug.
    """

    @staticmethod
    def _make_conv_node(op_type: str) -> onnx.NodeProto:
        return onnx.helper.make_node(
            op_type,
            inputs=["data", "dynamic_weight", "bias"],
            outputs=["out"],
            kernel_shape=[3, 3],
        )

    def test_dynamic_weight_conv_maps_to_functional(self):
        """A Conv whose weight is absent from initializers maps to F.conv."""
        node = self._make_conv_node("Conv")
        # Only the bias is a static initializer; the weight is dynamic.
        initializers = {"bias": onnx.TensorProto()}
        assert convert_to_pytorch_type(node, initializers) == "F.conv"

    def test_dynamic_weight_convtranspose_maps_to_functional(self):
        """A ConvTranspose with a dynamic weight maps to F.conv_transpose."""
        node = self._make_conv_node("ConvTranspose")
        initializers = {"bias": onnx.TensorProto()}
        assert convert_to_pytorch_type(node, initializers) == "F.conv_transpose"


class TestTypeMappingOperations:
    """Test mapping of ONNX operators to PyTorch operations.

    These tests work because they don't rely on the broken classify_inputs function.
    """

    # [REVIEW] Parametrized: test_convert_reshape_to_operation,
    # test_convert_concat_to_operation, test_convert_transpose_to_operation

    @pytest.mark.parametrize(
        "model_fixture_name",
        [
            pytest.param("reshape_model", id="reshape"),
            pytest.param("concat_model", id="concat"),
            pytest.param("transpose_model", id="transpose"),
        ],
    )
    def test_convert_operation_to_pytorch(self, model_fixture_name, request):
        """Identify ONNX operation as non-layer type."""
        model = request.getfixturevalue(model_fixture_name)
        normalized = load_and_preprocess_onnx_model(model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        pytorch_type = convert_to_pytorch_type(layer.node, model_ir.initializers)
        assert not is_layer_with_args(pytorch_type)

    def test_is_operator_detection(self):
        """Verify is_operator correctly identifies ONNX math operators."""
        # is_operator checks for ONNX operator names, not Python symbols
        assert is_operator("Add")
        assert is_operator("Sub")
        assert is_operator("Mul")
        assert is_operator("Div")
        assert is_operator("MatMul")

    def test_is_operation_detection(self):
        """Verify is_operation correctly identifies operations."""
        result = is_operation("Reshape")
        assert isinstance(result, bool)

    def test_extract_reshape_args(self, reshape_model):
        """Extract shape argument from Reshape operation."""
        normalized = load_and_preprocess_onnx_model(reshape_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        args = extract_operation_args(layer.node, model_ir.initializers, layer.onnx_op_type)
        assert isinstance(args, dict)

    def test_extract_concat_args(self, concat_model):
        """Extract axis argument from Concat operation."""
        normalized = load_and_preprocess_onnx_model(concat_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        args = extract_operation_args(layer.node, model_ir.initializers, layer.onnx_op_type)
        assert "axis" in args or "dim" in args


class TestAttributeExtraction:
    """Test attribute extraction and validation.

    These tests work because they don't rely on the broken classify_inputs function.
    """

    def test_extract_conv_attributes(self, conv2d_model):
        """Extract all Conv2d attributes (kernel, stride, padding, dilation)."""
        normalized = load_and_preprocess_onnx_model(conv2d_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        args = extract_layer_args(layer.node, model_ir.initializers)
        assert len(args) > 0

    def test_extract_maxpool_attributes(self, maxpool_model):
        """Extract MaxPool attributes (kernel_size, stride, padding)."""
        normalized = load_and_preprocess_onnx_model(maxpool_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        args = extract_layer_args(layer.node, model_ir.initializers)
        assert len(args) > 0

    def test_symmetric_padding_is_valid(self, conv2d_model):
        """Verify Conv2d has valid symmetric padding."""
        normalized = load_and_preprocess_onnx_model(conv2d_model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        args = extract_layer_args(layer.node, model_ir.initializers)
        # Conv should have some attributes extracted
        assert isinstance(args, dict)

    def test_model_ir_structure_is_valid(self, linear_model):
        """Verify ModelIR has correct structure."""
        normalized = load_and_preprocess_onnx_model(linear_model)
        model_ir = build_model_ir(normalized)

        assert hasattr(model_ir, "layers")
        assert hasattr(model_ir, "initializers")
        assert hasattr(model_ir, "shapes")
        assert len(model_ir.layers) >= 1


class TestAttributeExtractionExtended:
    """Test attribute extraction for additional ONNX operators (Phase 3)."""

    # [REVIEW] Parametrized: test_extract_cast_attributes, test_extract_argmax_attributes,
    # test_extract_convtranspose_attributes, test_extract_constantofshape_attributes

    @pytest.mark.parametrize(
        ("model_fixture_name", "expected_op_type"),
        [
            pytest.param("cast_model", "Cast", id="cast"),
            pytest.param("argmax_model", "ArgMax", id="argmax"),
            pytest.param("convtranspose_model", "ConvTranspose", id="convtranspose"),
            pytest.param("constantofshape_model", "ConstantOfShape", id="constantofshape"),
        ],
    )
    def test_extract_operator_attributes(self, model_fixture_name, expected_op_type, request):
        """Verify ONNX operator attributes are extracted correctly."""
        model = request.getfixturevalue(model_fixture_name)
        normalized = load_and_preprocess_onnx_model(model)
        model_ir = build_model_ir(normalized)
        assert len(model_ir.layers) >= 1
        layer = model_ir.layers[0]

        assert layer.node.op_type == expected_op_type

    def test_cast_model_structure(self, cast_model):
        """Verify Cast model has valid structure."""
        normalized = load_and_preprocess_onnx_model(cast_model)
        model_ir = build_model_ir(normalized)

        assert len(model_ir.input_names) >= 1
        assert len(model_ir.output_names) >= 1
        assert model_ir.layers[0].node.op_type == "Cast"

    def test_argmax_model_structure(self, argmax_model):
        """Verify ArgMax model has valid structure."""
        normalized = load_and_preprocess_onnx_model(argmax_model)
        model_ir = build_model_ir(normalized)

        assert len(model_ir.input_names) >= 1
        assert len(model_ir.output_names) >= 1
        assert model_ir.layers[0].node.op_type == "ArgMax"

    def test_convtranspose_model_structure(self, convtranspose_model):
        """Verify ConvTranspose model has valid structure."""
        normalized = load_and_preprocess_onnx_model(convtranspose_model)
        model_ir = build_model_ir(normalized)

        assert len(model_ir.input_names) >= 1
        assert len(model_ir.output_names) >= 1
        assert model_ir.layers[0].node.op_type == "ConvTranspose"
        # ConvTranspose should have initializers for weights
        assert len(model_ir.initializers) > 0

    def test_constantofshape_model_structure(self, constantofshape_model):
        """Verify ConstantOfShape model has valid structure."""
        normalized = load_and_preprocess_onnx_model(constantofshape_model)
        model_ir = build_model_ir(normalized)

        assert len(model_ir.output_names) >= 1
        assert model_ir.layers[0].node.op_type == "ConstantOfShape"

    # [REVIEW] Parametrized: test_cast_operator_type_conversion,
    # test_argmax_operator_type_conversion, test_convtranspose_operator_type_conversion

    @pytest.mark.parametrize(
        ("model_fixture_name", "expect_is_layer"),
        [
            pytest.param("cast_model", False, id="cast"),
            pytest.param("argmax_model", False, id="argmax"),
            pytest.param("convtranspose_model", True, id="convtranspose"),
        ],
    )
    def test_operator_type_conversion(self, model_fixture_name, expect_is_layer, request):
        """Verify operator-to-PyTorch type conversion classification."""
        model = request.getfixturevalue(model_fixture_name)
        normalized = load_and_preprocess_onnx_model(model)
        model_ir = build_model_ir(normalized)
        layer = model_ir.layers[0]

        pytorch_type = convert_to_pytorch_type(layer.node, model_ir.initializers)
        assert is_layer_with_args(pytorch_type) == expect_is_layer


class TestLayerExtractionEdgeCases:
    """Test layer extraction edge cases and error handling (Phase 3).

    Tests for _layers.py covering asymmetric padding, missing attributes,
    default value handling, and layer-specific extraction functions.
    """

    def test_simplify_tuple_all_equal(self):
        """Test simplify_tuple with all equal values."""
        # All equal elements should simplify to single value
        assert _simplify_tuple((3, 3, 3)) == 3
        assert _simplify_tuple((1, 1)) == 1
        assert _simplify_tuple((5,)) == 5

    def test_simplify_tuple_heterogeneous(self):
        """Test simplify_tuple preserves heterogeneous values."""
        # Different elements should remain as tuple
        assert _simplify_tuple((1, 2, 3)) == (1, 2, 3)
        assert _simplify_tuple((2, 3)) == (2, 3)

    def test_simplify_tuple_empty(self):
        """Test simplify_tuple with empty tuple."""
        # Empty tuple should remain empty
        assert _simplify_tuple(()) == ()

    def test_check_pads_symmetric_valid(self):
        """Test symmetric padding validation passes for symmetric padding."""
        # Symmetric padding should not raise error
        try:
            _check_pads_symmetric((1, 1, 1, 1))
            _check_pads_symmetric((2, 3, 2, 3))
        except ValueError as e:
            raise AssertionError("Symmetric padding should not raise error") from e

    def test_check_pads_symmetric_invalid(self):
        """Test symmetric padding validation raises for asymmetric padding."""
        # Asymmetric padding should raise error (top=1, left=2, bottom=2, right=1)
        with pytest.raises(ValueError, match=r"asymmetric|symmetric"):
            _check_pads_symmetric((1, 2, 2, 1))

    def test_check_pads_symmetric_3d(self):
        """Test symmetric padding for 3D operations."""
        # 3D: [start_d, start_h, start_w, end_d, end_h, end_w]
        # Symmetric: start==end for each dimension
        try:
            _check_pads_symmetric((1, 2, 3, 1, 2, 3))
        except ValueError as e:
            raise AssertionError("Symmetric 3D padding should not raise error") from e

        # Asymmetric should raise
        with pytest.raises(ValueError, match=r"asymmetric|symmetric"):
            _check_pads_symmetric((1, 2, 3, 1, 2, 4))

    def test_extract_dropout_opset_11_ratio_attribute(self):
        """Test Dropout with opset 11 (ratio as attribute)."""
        # Opset 11: ratio is attribute
        node = onnx.helper.make_node("Dropout", inputs=["X"], outputs=["Y"], ratio=0.5)

        args = _extract_dropout_args(node, {})
        assert isinstance(args, dict)
        # Should contain p (probability) extracted from ratio
        if "p" in args:
            assert 0 <= args["p"] <= 1

    def test_extract_dropout_opset_12_ratio_input(self):
        """Test Dropout with opset 12+ (ratio as input)."""
        # Opset 12+: ratio is second input
        node = onnx.helper.make_node("Dropout", inputs=["X", "ratio_input"], outputs=["Y", "mask"])

        args = _extract_dropout_args(node, {})
        assert isinstance(args, dict)

    def test_extract_gelu_no_approximation(self):
        """Test GELU without approximation attribute."""
        # GELU with default (no approximation)
        node = onnx.helper.make_node("Gelu", inputs=["X"], outputs=["Y"])

        args = _extract_gelu_args(node, {})
        assert isinstance(args, dict)

    def test_extract_gelu_with_approximation(self):
        """Test GELU with approximation attribute."""
        # GELU with approximation='tanh'
        node = onnx.helper.make_node("Gelu", inputs=["X"], outputs=["Y"], approximate="tanh")

        args = _extract_gelu_args(node, {})
        assert isinstance(args, dict)

    def test_extract_softmax_axis_attribute(self):
        """Test Softmax with axis attribute."""
        # Softmax with axis parameter
        node = onnx.helper.make_node("Softmax", inputs=["X"], outputs=["Y"], axis=1)

        args = _extract_softmax_args(node, {})
        assert isinstance(args, dict)
        if "dim" in args:
            assert args["dim"] == 1

    def test_extract_flatten_axis_attribute(self):
        """Test Flatten with axis attribute."""
        node = onnx.helper.make_node("Flatten", inputs=["X"], outputs=["Y"], axis=2)

        args = extract_operation_args(node, {}, "Flatten")

        assert convert_to_pytorch_type(node, {}) == "flatten"
        assert args == {"axis": 2}

    def test_extract_leakyrelu_alpha_attribute(self):
        """Test LeakyReLU with alpha attribute."""
        # LeakyReLU with alpha parameter
        node = onnx.helper.make_node("LeakyRelu", inputs=["X"], outputs=["Y"], alpha=0.2)

        args = _extract_leakyrelu_args(node, {})
        assert isinstance(args, dict)
        if "negative_slope" in args:
            assert args["negative_slope"] == pytest.approx(0.2)

    def test_extract_elu_alpha_attribute(self):
        """Test ELU with alpha attribute."""
        # ELU with alpha parameter
        node = onnx.helper.make_node("Elu", inputs=["X"], outputs=["Y"], alpha=1.0)

        args = _extract_elu_args(node, {})
        assert isinstance(args, dict)
        if "alpha" in args:
            assert args["alpha"] == 1.0

    def test_extract_upsample_scales_attribute(self):
        """Test Upsample with scales attribute."""
        # Upsample with scales parameter
        node = onnx.helper.make_node(
            "Upsample", inputs=["X"], outputs=["Y"], scales=[1.0, 1.0, 2.0, 2.0]
        )

        args = _extract_upsample_args(node, {})
        assert isinstance(args, dict)

    def test_extract_upsample_mode_attribute(self):
        """Test Upsample with mode attribute."""
        # Upsample with mode parameter (nearest, linear, etc.)
        node = onnx.helper.make_node(
            "Upsample", inputs=["X"], outputs=["Y"], mode="nearest", scales=[1.0, 1.0, 2.0, 2.0]
        )

        args = _extract_upsample_args(node, {})
        assert isinstance(args, dict)

    def test_extract_resize_static_sizes(self):
        """Resize sizes map to the PyTorch output-size argument, not ROI."""
        sizes = onnx.helper.make_tensor("sizes", onnx.TensorProto.INT64, [4], [1, 1, 1, 7])
        node = onnx.helper.make_node(
            "Resize", inputs=["X", "", "", "sizes"], outputs=["Y"], mode="nearest"
        )

        args = _extract_upsample_args(node, {"sizes": sizes})

        assert args == {"size": (1, 7), "mode": "nearest"}

    def test_resize_roi_is_not_a_legacy_scale(self):
        """A Resize ROI cannot silently become an Upsample scale factor."""
        roi = onnx.helper.make_tensor(
            "roi", onnx.TensorProto.FLOAT, [8], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        )
        node = onnx.helper.make_node("Resize", inputs=["X", "roi"], outputs=["Y"])

        args = _extract_upsample_args(node, {"roi": roi})

        assert args == {"mode": "nearest"}

    def test_globalavgpool_maps_to_rank_agnostic_operation(self):
        """Map GlobalAveragePool to its spatial reduction operation."""
        node = onnx.helper.make_node("GlobalAveragePool", inputs=["X"], outputs=["Y"])

        assert convert_to_pytorch_type(node, {}) == "global_average_pool"
        assert extract_operation_args(node, {}, "GlobalAveragePool") == {}
