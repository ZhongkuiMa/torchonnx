"""Pytest configuration and shared fixtures for torchonnx tests."""

import onnx
import onnx.helper as onnx_helper
import pytest


@pytest.fixture
def identity_model(tmp_path):
    """Create and save Identity ONNX model."""
    X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, (1, 3))  # noqa: N806
    Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, (1, 3))  # noqa: N806
    node = onnx_helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = onnx_helper.make_graph([node], "IdentityModel", [X], [Y])
    model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
    model.ir_version = 8

    path = tmp_path / "identity.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def linear_model(tmp_path):
    """Create and save Linear ONNX model."""
    import numpy as np

    rng = np.random.default_rng(42)
    input_size = 3
    output_size = 2
    X = onnx_helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, input_size])  # noqa: N806
    W_init = onnx_helper.make_tensor(  # noqa: N806
        "W",
        onnx.TensorProto.FLOAT,
        [output_size, input_size],
        vals=rng.standard_normal((output_size, input_size)).astype(np.float32).flatten().tolist(),
    )
    B_init = onnx_helper.make_tensor(  # noqa: N806
        "B",
        onnx.TensorProto.FLOAT,
        [output_size],
        vals=rng.standard_normal(output_size).astype(np.float32).tolist(),
    )
    Y = onnx_helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, output_size])  # noqa: N806

    node = onnx_helper.make_node(
        "Gemm",
        inputs=["X", "W", "B"],
        outputs=["Y"],
        alpha=1.0,
        beta=1.0,
        transB=1,
    )

    graph = onnx_helper.make_graph(
        [node],
        "LinearModel",
        [X],
        [Y],
        [W_init, B_init],
    )

    model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 20)])
    model.ir_version = 8

    path = tmp_path / "linear.onnx"
    onnx.save(model, str(path))
    return str(path)
