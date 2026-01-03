"""Performance benchmark tests for torchonnx compilation pipeline.

Slow performance tests marked as benchmarks.
Run with: pytest tests/test_benchmarks/ -v or pytest -m benchmark
"""

import pytest

from torchonnx.build import build_model_ir
from torchonnx.normalize import load_and_preprocess_onnx_model
from torchonnx.simplify import format_code


class TestSynthesisONNXModels:
    """Fixtures for creating small synthetic ONNX models for testing."""

    @staticmethod
    def create_identity_model(input_shape=(1, 3), output_shape=(1, 3)):
        """Create minimal Identity ONNX model (pass-through).

        :param input_shape: Input tensor shape
        :param output_shape: Output tensor shape
        :return: ONNX ModelProto
        """
        import onnx
        import onnx.helper as onnx_helper

        X = onnx_helper.make_tensor_value_info(  # noqa: N806
            "X", onnx.TensorProto.FLOAT, input_shape
        )
        Y = onnx_helper.make_tensor_value_info(  # noqa: N806
            "Y", onnx.TensorProto.FLOAT, output_shape
        )

        node = onnx_helper.make_node("Identity", inputs=["X"], outputs=["Y"])

        graph = onnx_helper.make_graph(
            [node],
            "IdentityModel",
            [X],
            [Y],
        )

        return onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 13)])


@pytest.mark.benchmark
class TestPerformance:
    """Performance tests for core operations (marked as benchmark)."""

    def test_normalize_performance(self, identity_model):
        """Benchmark model loading."""
        for _ in range(3):
            model = load_and_preprocess_onnx_model(identity_model)
            assert model is not None

    def test_build_performance(self, linear_model):
        """Benchmark IR building."""
        model = load_and_preprocess_onnx_model(linear_model)
        for _ in range(3):
            ir = build_model_ir(model)
            assert ir is not None

    def test_format_code_performance(self):
        """Benchmark code formatting."""
        code = "x=1\ny=2\nz=x+y\n" * 10
        for _ in range(3):
            formatted = format_code(code)
            assert formatted is not None

    def test_model_creation_performance(self):
        """Benchmark synthetic model creation."""
        for _ in range(5):
            model = TestSynthesisONNXModels.create_identity_model()
            assert model is not None
