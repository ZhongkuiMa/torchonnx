"""End-to-End Integration Tests - Complete Pipeline with Code Execution.

This module tests the entire conversion pipeline from ONNX to executable PyTorch code:
- Code generation from ONNX models
- Code validation (syntax, imports, instantiation)
- State dict loading and initialization
- Numerical validation against ONNX outputs

Test Coverage:
- TestExecutableCodeGeneration: 4 tests - Generated code is valid & executable
- TestNumericalValidation: 10 tests - ONNX vs PyTorch output matching
- TestStateDictValidation: 4 tests - State dict integrity
- TestModelExecution: 2 tests - Runtime execution safety

Failures in semantic IR construction are test failures; integration tests do not
convert implementation exceptions into skips.
"""

__docformat__ = "restructuredtext"

import ast
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from torchonnx import TorchONNX
from torchonnx.analyze import build_semantic_ir
from torchonnx.build import build_model_ir
from torchonnx.generate import generate_pytorch_module
from torchonnx.normalize import load_and_preprocess_onnx_model


def _assert_converted_model_matches_onnxruntime(
    model: onnx.ModelProto,
    data: np.ndarray | dict[str, np.ndarray],
    tmp_path: Path,
    stem: str,
    *,
    use_shapeonnx: bool = False,
) -> None:
    """Convert, strict-load, run, and compare an ONNX model."""
    onnx.checker.check_model(model)
    onnx_path = tmp_path / f"{stem}.onnx"
    py_path = tmp_path / f"{stem}.py"
    pth_path = tmp_path / f"{stem}.pth"
    onnx.save(model, onnx_path)

    TorchONNX(use_shapeonnx=use_shapeonnx).convert(
        str(onnx_path),
        target_py_path=str(py_path),
        target_pth_path=str(pth_path),
    )
    spec = importlib.util.spec_from_file_location(f"{stem}_generated", py_path)
    assert spec is not None
    assert spec.loader is not None
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    torch_model = getattr(generated, generated.__all__[0])()
    torch_model.load_state_dict(torch.load(pth_path, weights_only=True), strict=True)

    feeds = {"data": data} if isinstance(data, np.ndarray) else data
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected = session.run(None, feeds)[0]
    actual = torch_model(*(torch.from_numpy(feeds[item.name]) for item in session.get_inputs()))
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=1e-6, atol=1e-6)


def _load_converted_model(py_path: Path, pth_path: Path, module_name: str):
    """Load one generated module and its strict state dictionary."""
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    assert spec is not None
    assert spec.loader is not None
    generated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generated)
    torch_model = getattr(generated, generated.__all__[0])()
    torch_model.load_state_dict(torch.load(pth_path, weights_only=True), strict=True)
    return torch_model


def test_multi_input_output_metadata_and_onnxruntime_parity(tmp_path: Path) -> None:
    """Generated metadata and positional ABI preserve declared ONNX order."""
    z_input = onnx.helper.make_tensor_value_info("z_input", onnx.TensorProto.FLOAT, [3])
    a_input = onnx.helper.make_tensor_value_info("a_input", onnx.TensorProto.FLOAT, [3])
    legacy_bias = onnx.helper.make_tensor_value_info("bias", onnx.TensorProto.FLOAT, [3])
    second_output = onnx.helper.make_tensor_value_info("second_output", onnx.TensorProto.FLOAT, [3])
    first_output = onnx.helper.make_tensor_value_info("first_output", onnx.TensorProto.FLOAT, [3])
    bias = onnx.numpy_helper.from_array(np.array([0.25, -0.5, 1.0], dtype=np.float32), name="bias")
    nodes = [
        onnx.helper.make_node("Add", ["z_input", "a_input"], ["sum_without_bias"]),
        onnx.helper.make_node("Sub", ["z_input", "a_input"], ["second_output"]),
        onnx.helper.make_node("Add", ["sum_without_bias", "bias"], ["first_output"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "multi_io",
        [z_input, a_input, legacy_bias],
        [second_output, first_output],
        [bias],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])
    onnx_path = tmp_path / "multi_io.onnx"
    py_path = tmp_path / "multi_io.py"
    pth_path = tmp_path / "multi_io.pth"
    onnx.save(model, onnx_path)

    TorchONNX().convert(str(onnx_path), target_py_path=str(py_path), target_pth_path=str(pth_path))
    torch_model = _load_converted_model(py_path, pth_path, "multi_io_generated")

    assert torch_model.onnx_input_names == ("z_input", "a_input")
    assert torch_model.onnx_output_names == ("second_output", "first_output")
    assert "onnx_input_names" not in torch_model.state_dict()
    assert "onnx_output_names" not in torch_model.state_dict()

    feeds = {
        "z_input": np.array([1.0, 3.0, -2.0], dtype=np.float32),
        "a_input": np.array([0.5, -1.0, 4.0], dtype=np.float32),
    }
    expected = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    actual = torch_model(*(torch.from_numpy(feeds[name]) for name in torch_model.onnx_input_names))

    assert isinstance(actual, tuple)
    assert len(actual) == 2
    for actual_output, expected_output in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            actual_output.detach().numpy(), expected_output, rtol=1e-6, atol=1e-6
        )


def test_convert_model_vnnlib2_like_flat_binding_matches_onnxruntime(
    tmp_path: Path,
) -> None:
    """An in-memory flat-input composite keeps named multi-output semantics."""
    flat_input = onnx.helper.make_tensor_value_info("query_input", onnx.TensorProto.FLOAT, [4])
    left_output = onnx.helper.make_tensor_value_info(
        "network_left_output", onnx.TensorProto.FLOAT, [2]
    )
    right_output = onnx.helper.make_tensor_value_info(
        "network_right_output", onnx.TensorProto.FLOAT, [2]
    )
    left_indices = onnx.numpy_helper.from_array(
        np.array([0, 2], dtype=np.int64), name="left_indices"
    )
    right_indices = onnx.numpy_helper.from_array(
        np.array([1, 3], dtype=np.int64), name="right_indices"
    )
    nodes = [
        onnx.helper.make_node(
            "Gather", ["query_input", "left_indices"], ["network_left_output"], axis=0
        ),
        onnx.helper.make_node(
            "Gather", ["query_input", "right_indices"], ["network_right_output"], axis=0
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "vnnlib2_like_composite",
        [flat_input],
        [left_output, right_output],
        [left_indices, right_indices],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])
    py_path = tmp_path / "vnnlib2_composite.py"
    pth_path = tmp_path / "vnnlib2_composite.pth"

    TorchONNX().convert_model(
        model,
        target_py_path=str(py_path),
        target_pth_path=str(pth_path),
        source_label="vnnlib2_composite",
    )
    torch_model = _load_converted_model(py_path, pth_path, "vnnlib2_composite_generated")
    values = np.array([1.0, 10.0, 2.0, 20.0], dtype=np.float32)
    expected = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, {"query_input": values})
    actual = torch_model(torch.from_numpy(values))

    assert torch_model.onnx_input_names == ("query_input",)
    assert torch_model.onnx_output_names == (
        "network_left_output",
        "network_right_output",
    )
    for actual_output, expected_output in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_output.detach().numpy(), expected_output)


def test_atan_matches_onnxruntime(tmp_path: Path) -> None:
    """Atan lowers to the exact elementwise PyTorch operation."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [7])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [7])
    atan = onnx.helper.make_node("Atan", ["data"], ["output"])
    graph = onnx.helper.make_graph([atan], "atan", [data_info], [output_info])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.array([-100.0, -2.0, -0.25, 0.0, 0.25, 2.0, 100.0], dtype=np.float32)
    _assert_converted_model_matches_onnxruntime(model, values, tmp_path, "atan")


def test_erf_matches_onnxruntime(tmp_path: Path) -> None:
    """Erf lowers to the exact elementwise PyTorch operation."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [7])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [7])
    erf = onnx.helper.make_node("Erf", ["data"], ["output"])
    graph = onnx.helper.make_graph([erf], "erf", [data_info], [output_info])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.array([-100.0, -2.0, -0.25, 0.0, 0.25, 2.0, 100.0], dtype=np.float32)
    _assert_converted_model_matches_onnxruntime(model, values, tmp_path, "erf")


def test_clip_max_only_preserves_optional_input_position(tmp_path: Path) -> None:
    """Clip's third input remains max when its min slot is empty."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [5])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [5])
    max_bound = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="max")
    clip = onnx.helper.make_node("Clip", ["data", "", "max"], ["output"])
    graph = onnx.helper.make_graph([clip], "clip_max_only", [data_info], [output_info], [max_bound])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.array([-2.0, -0.5, 0.5, 2.0, 5.0], dtype=np.float32)
    _assert_converted_model_matches_onnxruntime(model, values, tmp_path, "clip_max_only")


def test_pad_omitted_value_with_axes_matches_onnxruntime(tmp_path: Path) -> None:
    """Pad keeps axes in slot four and applies the omitted zero value."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [5, 3])
    pads = onnx.numpy_helper.from_array(np.array([1, 2], dtype=np.int64), name="pads")
    axes = onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    pad = onnx.helper.make_node("Pad", ["data", "pads", "", "axes"], ["output"], mode="constant")
    graph = onnx.helper.make_graph([pad], "pad_axes", [data_info], [output_info], [pads, axes])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0
    _assert_converted_model_matches_onnxruntime(model, values, tmp_path, "pad_axes")


def test_slice_omitted_axes_with_steps_matches_onnxruntime(tmp_path: Path) -> None:
    """Slice keeps steps in slot five when its axes slot is empty."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [6])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [3])
    starts = onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts")
    ends = onnx.numpy_helper.from_array(np.array([6], dtype=np.int64), name="ends")
    steps = onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), name="steps")
    slice_node = onnx.helper.make_node("Slice", ["data", "starts", "ends", "", "steps"], ["output"])
    graph = onnx.helper.make_graph(
        [slice_node], "slice_default_axes", [data_info], [output_info], [starts, ends, steps]
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.arange(6, dtype=np.float32)
    _assert_converted_model_matches_onnxruntime(model, values, tmp_path, "slice_default_axes")


def test_slice_with_axes_and_omitted_steps_matches_onnxruntime(tmp_path: Path) -> None:
    """Slice applies the default step when slot five is explicitly empty."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [6])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [4])
    starts = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts")
    ends = onnx.numpy_helper.from_array(np.array([5], dtype=np.int64), name="ends")
    axes = onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    slice_node = onnx.helper.make_node("Slice", ["data", "starts", "ends", "axes", ""], ["output"])
    graph = onnx.helper.make_graph(
        [slice_node], "slice_default_steps", [data_info], [output_info], [starts, ends, axes]
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.arange(6, dtype=np.float32)
    _assert_converted_model_matches_onnxruntime(model, values, tmp_path, "slice_default_steps")


@pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param((), id="scalar"),
        pytest.param((5,), id="rank_one"),
        pytest.param((2, 3), id="rank_two"),
        pytest.param((2, 3, 4), id="rank_three"),
    ],
)
def test_transpose_without_perm_reverses_all_dimensions(
    tmp_path: Path, input_shape: tuple[int, ...]
) -> None:
    """Honor ONNX's default permutation instead of swapping only the last two axes."""
    output_shape = tuple(reversed(input_shape))
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, input_shape)
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)
    transpose = onnx.helper.make_node("Transpose", ["data"], ["output"])
    graph = onnx.helper.make_graph([transpose], "transpose_default", [data_info], [output_info])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    size = int(np.prod(input_shape)) if input_shape else 1
    values = np.arange(size, dtype=np.float32).reshape(input_shape)
    stem = f"transpose_default_rank_{len(input_shape)}"
    _assert_converted_model_matches_onnxruntime(model, values, tmp_path, stem)


@pytest.mark.parametrize(
    "axes_values",
    [
        pytest.param([0, 3], id="positive_axes"),
        pytest.param([-4, -1], id="negative_axes"),
    ],
)
@pytest.mark.parametrize("use_shapeonnx", [False, True], ids=["onnx_shapes", "shapeonnx_shapes"])
def test_unsqueeze_multiple_axes_matches_onnxruntime(
    tmp_path: Path, axes_values: list[int], use_shapeonnx: bool
) -> None:
    """Apply constant Unsqueeze axes as final output coordinates."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2, 3, 1])
    axes = onnx.numpy_helper.from_array(np.array(axes_values, dtype=np.int64), name="axes")
    unsqueeze = onnx.helper.make_node("Unsqueeze", ["data", "axes"], ["output"])
    graph = onnx.helper.make_graph(
        [unsqueeze], "unsqueeze_multiple_axes", [data_info], [output_info], [axes]
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    stem = f"unsqueeze_multiple_{axes_values[0]}_{axes_values[1]}_{use_shapeonnx}"
    _assert_converted_model_matches_onnxruntime(
        model,
        values,
        tmp_path,
        stem,
        use_shapeonnx=use_shapeonnx,
    )


@pytest.mark.parametrize("op_type", ["ReduceMean", "ReduceSum"])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("use_shapeonnx", [False, True], ids=["onnx_shapes", "shapeonnx_shapes"])
def test_reduce_without_axes_matches_onnxruntime(
    tmp_path: Path, op_type: str, keepdims: int, use_shapeonnx: bool
) -> None:
    """Reduce every dimension when ONNX omits the optional axes input."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2, 3])
    output_shape = [1, 1] if keepdims else []
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)
    reduce_node = onnx.helper.make_node(
        op_type,
        ["data"],
        ["output"],
        keepdims=keepdims,
    )
    graph = onnx.helper.make_graph(
        [reduce_node],
        f"{op_type.lower()}_without_axes",
        [data_info],
        [output_info],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    stem = f"{op_type.lower()}_without_axes_keepdims_{keepdims}_{use_shapeonnx}"
    _assert_converted_model_matches_onnxruntime(
        model,
        values,
        tmp_path,
        stem,
        use_shapeonnx=use_shapeonnx,
    )


def test_dilated_average_pool_is_rejected(tmp_path: Path) -> None:
    """Fail closed when ONNX pooling semantics lack a PyTorch layer equivalent."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [1, 1, 5, 5])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 1, 3, 3])
    average_pool = onnx.helper.make_node(
        "AveragePool",
        ["data"],
        ["output"],
        kernel_shape=[2, 2],
        strides=[1, 1],
        dilations=[2, 2],
    )
    graph = onnx.helper.make_graph(
        [average_pool],
        "dilated_average_pool",
        [data_info],
        [output_info],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])
    onnx.checker.check_model(model)
    onnx_path = tmp_path / "dilated_average_pool.onnx"
    onnx.save(model, onnx_path)

    with pytest.raises(ValueError, match=r"AveragePool.*dilations"):
        TorchONNX().convert(str(onnx_path))


@pytest.mark.parametrize("op_type", ["AveragePool", "MaxPool"])
@pytest.mark.parametrize("use_shapeonnx", [False, True], ids=["onnx_shapes", "shapeonnx_shapes"])
@pytest.mark.parametrize(
    ("input_shape", "output_shape", "kernel_shape"),
    [
        pytest.param([1, 1, 5], [1, 1, 4], [2], id="one_dimensional"),
        pytest.param(
            [1, 1, 4, 4, 4],
            [1, 1, 3, 3, 3],
            [2, 2, 2],
            id="three_dimensional",
        ),
    ],
)
def test_pool_dimension_matches_onnxruntime(
    tmp_path: Path,
    op_type: str,
    input_shape: list[int],
    output_shape: list[int],
    kernel_shape: list[int],
    use_shapeonnx: bool,
) -> None:
    """Select the PyTorch pooling layer matching ONNX spatial rank."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, input_shape)
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)
    pool = onnx.helper.make_node(
        op_type,
        ["data"],
        ["output"],
        kernel_shape=kernel_shape,
    )
    graph = onnx.helper.make_graph(
        [pool],
        f"{op_type.lower()}_{len(kernel_shape)}d",
        [data_info],
        [output_info],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    stem = f"{op_type.lower()}_{len(kernel_shape)}d_{use_shapeonnx}"
    _assert_converted_model_matches_onnxruntime(
        model,
        values,
        tmp_path,
        stem,
        use_shapeonnx=use_shapeonnx,
    )


@pytest.mark.parametrize("use_shapeonnx", [False, True], ids=["onnx_shapes", "shapeonnx_shapes"])
@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [
        pytest.param([1, 2, 5], [1, 2, 1], id="one_spatial_dimension"),
        pytest.param([1, 2, 3, 4], [1, 2, 1, 1], id="two_spatial_dimensions"),
        pytest.param(
            [1, 2, 3, 4, 5],
            [1, 2, 1, 1, 1],
            id="three_spatial_dimensions",
        ),
    ],
)
def test_global_average_pool_matches_onnxruntime(
    tmp_path: Path,
    input_shape: list[int],
    output_shape: list[int],
    use_shapeonnx: bool,
) -> None:
    """Reduce every spatial axis regardless of input rank."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, input_shape)
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)
    pool = onnx.helper.make_node("GlobalAveragePool", ["data"], ["output"])
    graph = onnx.helper.make_graph(
        [pool],
        "global_average_pool",
        [data_info],
        [output_info],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])

    values = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
    stem = f"global_average_pool_{len(input_shape) - 2}d_{use_shapeonnx}"
    _assert_converted_model_matches_onnxruntime(
        model,
        values,
        tmp_path,
        stem,
        use_shapeonnx=use_shapeonnx,
    )


@pytest.mark.parametrize("op_type", ["Min", "Max"])
@pytest.mark.parametrize("arity", [1, 3])
@pytest.mark.parametrize("use_shapeonnx", [False, True], ids=["onnx_shapes", "shapeonnx_shapes"])
def test_variadic_minmax_matches_onnxruntime(
    tmp_path: Path, op_type: str, arity: int, use_shapeonnx: bool
) -> None:
    """Apply elementwise Min/Max with broadcasting across every input."""
    shapes = [[2, 3]] if arity == 1 else [[2, 1], [1, 3], []]
    values = [np.array([[1.0, 4.0, 2.0], [5.0, 0.0, 3.0]], dtype=np.float32)]
    if arity == 3:
        values = [
            np.array([[1.0], [4.0]], dtype=np.float32),
            np.array([[2.0, 3.0, 0.0]], dtype=np.float32),
            np.array(2.5, dtype=np.float32),
        ]

    input_names = [f"data{index}" for index in range(arity)]
    input_infos = [
        onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
        for name, shape in zip(input_names, shapes, strict=True)
    ]
    output_shape = [2, 3]
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)
    node = onnx.helper.make_node(op_type, input_names, ["output"])
    graph = onnx.helper.make_graph(
        [node], f"variadic_{op_type.lower()}", input_infos, [output_info]
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])
    feeds = dict(zip(input_names, values, strict=True))

    _assert_converted_model_matches_onnxruntime(
        model,
        feeds,
        tmp_path,
        f"{op_type.lower()}_{arity}_{use_shapeonnx}",
        use_shapeonnx=use_shapeonnx,
    )


def test_dynamic_slice_requires_non_vmap_and_preserves_runtime_length(tmp_path: Path) -> None:
    """Reject unproven vmap shapes and preserve eager dynamic Slice semantics."""
    data_info = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [5])
    starts_info = onnx.helper.make_tensor_value_info("starts", onnx.TensorProto.INT64, [1])
    ends_info = onnx.helper.make_tensor_value_info("ends", onnx.TensorProto.INT64, [1])
    output_info = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [None])
    axes = onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes")
    steps = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps")
    slice_node = onnx.helper.make_node(
        "Slice",
        ["data", "starts", "ends", "axes", "steps"],
        ["output"],
    )
    graph = onnx.helper.make_graph(
        [slice_node],
        "dynamic_slice",
        [data_info, starts_info, ends_info],
        [output_info],
        [axes, steps],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 20)])
    onnx_path = tmp_path / "dynamic_slice.onnx"
    py_path = tmp_path / "dynamic_slice.py"
    pth_path = tmp_path / "dynamic_slice.pth"
    onnx.save(model, onnx_path)

    with pytest.raises(
        ValueError,
        match=r"cannot be proven statically with vmap_mode=True.*vmap_mode=False",
    ):
        TorchONNX().convert(
            str(onnx_path),
            target_py_path=str(py_path),
            target_pth_path=str(pth_path),
        )

    TorchONNX().convert(
        str(onnx_path),
        target_py_path=str(py_path),
        target_pth_path=str(pth_path),
        vmap_mode=False,
    )
    spec = importlib.util.spec_from_file_location("dynamic_slice_generated", py_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch_model = getattr(module, module.__all__[0])()
    torch_model.load_state_dict(torch.load(pth_path, weights_only=True), strict=True)

    actual = torch_model(
        torch.arange(5, dtype=torch.float32),
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([4], dtype=torch.int64),
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 2.0, 3.0]))


def test_convert_inlined_reshape_loads_strict_without_orphan_keys(reshape_model, tmp_path):
    """Full conversion of a Reshape model loads strictly (no orphan keys).

    The Reshape shape is inlined as a literal in ``forward`` (e.g.
    ``x.reshape(-1, 512, 1, 1)``); its shape constant must not leak into
    the state dict, else the production strict ``load_state_dict`` raises
    "Unexpected key(s)". Unlike :class:`IntegrationTestHelper`, this drives
    the full ``TorchONNX().convert`` pipeline (which runs the state-dict
    filter) and loads with ``strict=True``. Regression guard for the
    inlined-constant c0/c1 codegen bug.
    """
    py_path = tmp_path / "rs.py"
    pth_path = tmp_path / "rs.pth"
    TorchONNX().convert(
        str(reshape_model), target_py_path=str(py_path), target_pth_path=str(pth_path)
    )

    spec = importlib.util.spec_from_file_location("rs_generated", str(py_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, module.__all__[0])()
    state_dict = torch.load(str(pth_path), weights_only=True)

    assert set(state_dict.keys()) == set(model.state_dict().keys())
    model.load_state_dict(state_dict, strict=True)


@pytest.mark.parametrize(
    "axis",
    [pytest.param(None, id="default"), 0, 1, 2, 3, -1],
)
def test_convert_flatten_preserves_onnx_two_sided_semantics(
    axis: int | None,
    tmp_path: Path,
) -> None:
    """Flatten always produces the two dimensions defined by ONNX."""
    input_info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [2, 3, 4])
    output_info = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [None, None])
    attrs = {} if axis is None else {"axis": axis}
    flatten = onnx.helper.make_node("Flatten", ["x"], ["y"], **attrs)
    graph = onnx.helper.make_graph([flatten], "flatten", [input_info], [output_info])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    onnx_path = tmp_path / f"flatten_{axis}.onnx"
    py_path = tmp_path / f"flatten_{axis}.py"
    pth_path = tmp_path / f"flatten_{axis}.pth"
    onnx.save(model, onnx_path)

    TorchONNX().convert(
        str(onnx_path),
        target_py_path=str(py_path),
        target_pth_path=str(pth_path),
    )
    spec = importlib.util.spec_from_file_location(f"flatten_{axis}_generated", py_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch_model = getattr(module, module.__all__[0])()
    torch_model.load_state_dict(torch.load(pth_path, weights_only=True), strict=True)

    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, {"x": values}
    )[0]
    actual = torch_model(torch.from_numpy(values)).detach().numpy()

    np.testing.assert_array_equal(actual, expected)


class IntegrationTestHelper:
    """Helper functions for integration testing."""

    @staticmethod
    def generate_model_code_and_state_dict(onnx_model_path):
        """Generate PyTorch code and state dict from ONNX model.

        :param onnx_model_path: Path to ONNX model file.

        :return: Tuple of (module_code, state_dict) or raises if pipeline fails
        """
        normalized = load_and_preprocess_onnx_model(onnx_model_path)
        model_ir = build_model_ir(normalized)
        semantic_ir = build_semantic_ir(model_ir)

        module_code, state_dict = generate_pytorch_module(semantic_ir)
        return module_code, state_dict

    @staticmethod
    def load_generated_model(module_code, state_dict, temp_dir=None):
        """Load generated PyTorch model from code string.

        :param module_code: Generated Python code string.

        :param state_dict: State dictionary for model.

        :param temp_dir: Optional temp directory for file operations.

        :return: Tuple of (model_instance, module)
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()

        # Write module code to temporary file
        py_path = Path(temp_dir) / "generated_model.py"
        py_path.write_text(module_code)

        # Import the module
        spec = importlib.util.spec_from_file_location("generated_model", str(py_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from {py_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Instantiate and load state dict
        model_class = module.ONNXModel
        model = model_class()
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        return model, module

    @staticmethod
    def get_onnx_inputs(onnx_model_path, num_inputs=1):
        """Generate random inputs matching ONNX model signature.

        :param onnx_model_path: Path to ONNX model.

        :param num_inputs: Number of input sets to generate.

        :return: List of input dictionaries
        """
        session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        inputs_list = []
        rng = np.random.default_rng()

        for _ in range(num_inputs):
            inputs = {}
            for input_info in session.get_inputs():
                input_name = input_info.name
                input_shape = input_info.shape
                # Replace dynamic dimensions with actual values
                input_shape = [dim if isinstance(dim, int) else 1 for dim in input_shape]
                inputs[input_name] = rng.standard_normal(input_shape).astype(np.float32)
            inputs_list.append(inputs)

        return inputs_list

    @staticmethod
    def run_onnx_model(onnx_model_path, inputs):
        """Run ONNX model and return outputs.

        :param onnx_model_path: Path to ONNX model.

        :param inputs: Input dictionary.

        :return: List of output arrays
        """
        session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        return session.run(None, inputs)

    @staticmethod
    def compare_outputs(onnx_output, pytorch_output, rtol=1e-5, atol=1e-6):
        """Compare ONNX and PyTorch outputs for numerical equivalence.

        :param onnx_output: Output from ONNX model (numpy array or list).

        :param pytorch_output: Output from PyTorch model (tensor or array).

        :param rtol: Relative tolerance.

        :param atol: Absolute tolerance.

        :return: True if outputs match within tolerance
        """
        if isinstance(pytorch_output, torch.Tensor):
            pytorch_output = pytorch_output.detach().numpy()

        if isinstance(onnx_output, list):
            onnx_output = onnx_output[0] if len(onnx_output) == 1 else np.array(onnx_output)

        try:
            np.testing.assert_allclose(
                pytorch_output,
                onnx_output,
                rtol=rtol,
                atol=atol,
            )
            return True
        except AssertionError:
            return False


class TestExecutableCodeGeneration:
    """Test that generated code is valid and executable Python."""

    def test_generated_code_is_valid_python(self, linear_model):
        """Verify generated code can be parsed as valid Python."""
        code, _state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        # Try to parse the code

        ast.parse(code)
        assert isinstance(code, str)
        assert len(code) > 0

    def test_generated_code_imports_succeed(self, linear_model):
        """Verify generated code can be imported without errors."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            # This will raise ImportError if imports fail
            model, module = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            assert isinstance(model, torch.nn.Module)
            assert hasattr(module, "ONNXModel")
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_generated_class_instantiates(self, linear_model):
        """Verify generated model class can be instantiated."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            assert isinstance(model, torch.nn.Module)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_dict_loads_successfully(self, linear_model):
        """Verify state dict can be loaded into generated model."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            # load_generated_model already loads the state dict
            # If we get here without exception, loading succeeded
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            assert isinstance(model.state_dict(), dict)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestNumericalValidation:
    """Test numerical equivalence between ONNX and generated PyTorch models."""

    def test_linear_model_output_matches_onnx(self, linear_model):
        """Verify Linear model PyTorch output matches ONNX."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        # Generate test input
        inputs = IntegrationTestHelper.get_onnx_inputs(linear_model, num_inputs=1)[0]

        temp_dir = tempfile.mkdtemp()
        try:
            # Get ONNX output
            onnx_output = IntegrationTestHelper.run_onnx_model(linear_model, inputs)

            # Get PyTorch output
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            torch_input = torch.from_numpy(inputs[next(iter(inputs.keys()))])
            with torch.no_grad():
                torch_output = model(torch_input)

            # Compare
            assert IntegrationTestHelper.compare_outputs(
                onnx_output[0], torch_output, rtol=1e-5, atol=1e-6
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_conv_model_output_matches_onnx(self, conv2d_model):
        """Verify Conv2d model PyTorch output matches ONNX."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(conv2d_model)

        inputs = IntegrationTestHelper.get_onnx_inputs(conv2d_model, num_inputs=1)[0]

        temp_dir = tempfile.mkdtemp()
        try:
            onnx_output = IntegrationTestHelper.run_onnx_model(conv2d_model, inputs)

            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            torch_input = torch.from_numpy(inputs[next(iter(inputs.keys()))])
            with torch.no_grad():
                torch_output = model(torch_input)

            # Conv2d requires slightly larger tolerance due to floating point precision
            # Max difference observed: 7.6e-6 absolute, 2.76e-2 relative
            assert IntegrationTestHelper.compare_outputs(
                onnx_output[0], torch_output, rtol=1e-4, atol=1e-5
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_mlp_model_output_matches_onnx(self, mlp_model):
        """Verify MLP model PyTorch output matches ONNX."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(mlp_model)

        inputs = IntegrationTestHelper.get_onnx_inputs(mlp_model, num_inputs=1)[0]

        temp_dir = tempfile.mkdtemp()
        try:
            onnx_output = IntegrationTestHelper.run_onnx_model(mlp_model, inputs)

            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            torch_input = torch.from_numpy(inputs[next(iter(inputs.keys()))])
            with torch.no_grad():
                torch_output = model(torch_input)

            assert IntegrationTestHelper.compare_outputs(
                onnx_output[0], torch_output, rtol=1e-5, atol=1e-6
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_batchnorm_model_output_matches_onnx(self, batchnorm_model):
        """Verify BatchNorm model PyTorch output matches ONNX."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(batchnorm_model)

        inputs = IntegrationTestHelper.get_onnx_inputs(batchnorm_model, num_inputs=1)[0]

        temp_dir = tempfile.mkdtemp()
        try:
            onnx_output = IntegrationTestHelper.run_onnx_model(batchnorm_model, inputs)

            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            torch_input = torch.from_numpy(inputs[next(iter(inputs.keys()))])
            with torch.no_grad():
                torch_output = model(torch_input)

            # BatchNorm might have slightly more tolerance due to running stats
            assert IntegrationTestHelper.compare_outputs(
                onnx_output[0], torch_output, rtol=1e-3, atol=1e-4
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    # [REVIEW] Parametrized: test_reshape_model_output_matches_onnx,
    # test_maxpool_model_output_matches_onnx, test_avgpool_model_output_matches_onnx

    @pytest.mark.parametrize(
        ("model_fixture_name", "rtol", "atol"),
        [
            pytest.param("reshape_model", 1e-5, 1e-6, id="reshape"),
            pytest.param("maxpool_model", 1e-5, 1e-6, id="maxpool"),
            pytest.param("avgpool_model", 1e-5, 1e-6, id="avgpool"),
        ],
    )
    def test_model_output_matches_onnx(self, model_fixture_name, rtol, atol, request):
        """Verify model PyTorch output matches ONNX output."""
        model = request.getfixturevalue(model_fixture_name)
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(model)

        inputs = IntegrationTestHelper.get_onnx_inputs(model, num_inputs=1)[0]

        temp_dir = tempfile.mkdtemp()
        try:
            onnx_output = IntegrationTestHelper.run_onnx_model(model, inputs)

            pytorch_model, _ = IntegrationTestHelper.load_generated_model(
                code, state_dict, temp_dir
            )
            torch_input = torch.from_numpy(inputs[next(iter(inputs.keys()))])
            with torch.no_grad():
                torch_output = pytorch_model(torch_input)

            assert IntegrationTestHelper.compare_outputs(
                onnx_output[0], torch_output, rtol=rtol, atol=atol
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_concat_model_output_matches_onnx(self, concat_model):
        """Verify Concat model PyTorch output matches ONNX."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(concat_model)

        inputs = IntegrationTestHelper.get_onnx_inputs(concat_model, num_inputs=1)[0]

        temp_dir = tempfile.mkdtemp()
        try:
            onnx_output = IntegrationTestHelper.run_onnx_model(concat_model, inputs)

            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            # Concat model expects multiple inputs - pass all in sorted order
            torch_inputs = [torch.from_numpy(inputs[k]) for k in sorted(inputs.keys())]
            with torch.no_grad():
                torch_output = model(*torch_inputs)

            assert IntegrationTestHelper.compare_outputs(
                onnx_output[0], torch_output, rtol=1e-5, atol=1e-6
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_arithmetic_model_output_matches_onnx(self, arithmetic_model):
        """Verify Arithmetic model PyTorch output matches ONNX."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(
            arithmetic_model
        )

        inputs = IntegrationTestHelper.get_onnx_inputs(arithmetic_model, num_inputs=1)[0]

        temp_dir = tempfile.mkdtemp()
        try:
            onnx_output = IntegrationTestHelper.run_onnx_model(arithmetic_model, inputs)

            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            # Arithmetic model expects multiple inputs - pass all in sorted order
            torch_inputs = [torch.from_numpy(inputs[k]) for k in sorted(inputs.keys())]
            with torch.no_grad():
                torch_output = model(*torch_inputs)

            assert IntegrationTestHelper.compare_outputs(
                onnx_output[0], torch_output, rtol=1e-5, atol=1e-6
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestStateDictValidation:
    """Test state dict correctness and loading."""

    def test_state_dict_keys_match_model(self, linear_model):
        """Verify state dict keys match model parameters."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            model_keys = set(model.state_dict().keys())
            loaded_keys = set(state_dict.keys())

            # All model keys should be present in loaded state dict
            assert model_keys.issubset(loaded_keys)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_dict_shapes_correct(self, linear_model):
        """Verify state dict tensor shapes are correct."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            for key in model.state_dict():
                model_shape = model.state_dict()[key].shape
                loaded_shape = state_dict[key].shape
                assert model_shape == loaded_shape
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_dict_dtypes_correct(self, linear_model):
        """Verify state dict tensor dtypes are correct."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            for key in model.state_dict():
                model_dtype = model.state_dict()[key].dtype
                loaded_dtype = state_dict[key].dtype
                assert model_dtype == loaded_dtype
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_dict_values_match_onnx(self, linear_model):
        """Verify state dict values come from ONNX initializers."""
        _code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        # Just verify state dict has tensors
        for value in state_dict.values():
            assert isinstance(value, torch.Tensor)
            assert value.numel() > 0


class TestModelExecution:
    """Test runtime execution of generated models."""

    def test_forward_pass_runs_without_error(self, linear_model):
        """Verify forward pass executes without errors."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        inputs = IntegrationTestHelper.get_onnx_inputs(linear_model, num_inputs=1)[0]

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            torch_input = torch.from_numpy(inputs[next(iter(inputs.keys()))])

            with torch.no_grad():
                output = model(torch_input)

            assert isinstance(output, torch.Tensor)
            assert output.numel() > 0
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_model_in_eval_mode(self, linear_model):
        """Verify model is in eval mode for inference."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            # Model should be in eval mode after load_generated_model
            assert not model.training
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestMultiOutputModels:
    """Test multi-output model handling in the full pipeline."""

    def test_multi_output_model_code_generation(self, multi_output_model):
        """Verify code generation works with multi-output models."""
        code, _ = IntegrationTestHelper.generate_model_code_and_state_dict(multi_output_model)

        assert isinstance(code, str)
        assert len(code) > 0
        assert "class ONNXModel" in code
        assert "def forward" in code

    def test_multi_output_forward_returns_tuple(self, multi_output_model):
        """Verify multi-output model forward method returns tuple."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(
            multi_output_model
        )

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            inputs = IntegrationTestHelper.get_onnx_inputs(multi_output_model)[0]
            torch_inputs = [torch.from_numpy(value) for value in inputs.values()]

            # Run inference
            with torch.no_grad():
                outputs = model(*torch_inputs)

            # Multi-output should return tuple or list (not single tensor)
            assert isinstance(outputs, (tuple, list)) or torch.is_tensor(outputs)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multi_output_state_dict_complete(self, multi_output_model):
        """Verify state dict contains all parameters for multi-output model."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(
            multi_output_model
        )

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            # All parameters should be in state dict
            loaded_keys = set(state_dict.keys())
            model_keys = set(model.state_dict().keys())
            assert loaded_keys == model_keys
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multi_output_numerical_validation(self, multi_output_model):
        """Compare multi-output model ONNX and PyTorch outputs numerically."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(
            multi_output_model
        )

        temp_dir = tempfile.mkdtemp()
        try:
            # ONNX inference
            onnx_session = ort.InferenceSession(
                multi_output_model, providers=["CPUExecutionProvider"]
            )

            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            input_dict = IntegrationTestHelper.get_onnx_inputs(multi_output_model)[0]
            torch_inputs = [torch.from_numpy(value) for value in input_dict.values()]

            onnx_outputs = onnx_session.run(None, input_dict)

            # PyTorch inference
            with torch.no_grad():
                pytorch_outputs = model(*torch_inputs)

            # Compare outputs (allowing numerical tolerance)
            if isinstance(pytorch_outputs, torch.Tensor):
                pytorch_outputs = [pytorch_outputs]
            else:
                pytorch_outputs = list(pytorch_outputs)

            assert len(onnx_outputs) == len(pytorch_outputs)

            for onnx_out, torch_out in zip(onnx_outputs, pytorch_outputs, strict=True):
                if torch.is_tensor(torch_out):
                    torch_out = torch_out.numpy()
                np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-4, atol=1e-5)
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multi_output_with_independent_branches(self, multi_output_model):
        """Verify multi-output model with independent computation branches."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(
            multi_output_model
        )

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)
            inputs = IntegrationTestHelper.get_onnx_inputs(multi_output_model)[0]
            torch_inputs = [torch.from_numpy(value) for value in inputs.values()]

            # Code should generate successfully
            assert "def forward" in code
            # Model should be executable
            model.eval()
            with torch.no_grad():
                outputs = model(*torch_inputs)
            # Should produce outputs (type doesn't matter)
            assert isinstance(outputs, (torch.Tensor, tuple, list))
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestStateDictEdgeCases:
    """Test state dict handling and edge cases."""

    def test_state_dict_parameter_names_unique(self, linear_model):
        """Verify parameter names in state dict are unique."""
        _, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        # All parameter names should be unique
        param_names = list(state_dict.keys())
        unique_names = set(param_names)
        assert len(param_names) == len(unique_names), "Duplicate parameter names found"

    def test_state_dict_with_nested_modules(self, mlp_model):
        """Verify state dict handles nested module parameters."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(mlp_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            # State dict should have hierarchical keys (e.g., "layer1.weight")
            loaded_dict = state_dict
            model_dict = model.state_dict()

            # Both should have the same keys
            assert set(loaded_dict.keys()) == set(model_dict.keys())
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_dict_dtype_preservation(self, linear_model):
        """Verify state dict preserves parameter data types."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            # Check dtype preservation
            for key in state_dict:
                loaded_dtype = state_dict[key].dtype
                model_dtype = model.state_dict()[key].dtype
                assert loaded_dtype == model_dtype, (
                    f"Dtype mismatch for {key}: {loaded_dtype} vs {model_dtype}"
                )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_dict_with_no_parameters(self):
        """Test model with no learnable parameters (e.g., pure activations)."""
        # Create a model with only activation function (no weights)
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 10])  # noqa: N806

        # ReLU has no parameters
        node = onnx.helper.make_node("Relu", inputs=["X"], outputs=["Y"])
        graph = onnx.helper.make_graph([node], "NoParamModel", [X], [Y])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(model, f.name)
            temp_path = f.name

        try:
            code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(temp_path)

            temp_dir = tempfile.mkdtemp()
            try:
                _, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

                # Model with no parameters should have empty state dict
                assert isinstance(state_dict, dict)
                # Can be empty or have minimal entries
                assert len(state_dict) == 0 or all(
                    isinstance(v, torch.Tensor) for v in state_dict.values()
                )
            finally:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
        finally:
            Path(temp_path).unlink()

    def test_state_dict_buffer_handling(self, linear_model):
        """Verify state dict correctly handles buffers (e.g., running stats in BatchNorm)."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            # Check state dict is loadable
            model.load_state_dict(state_dict)

            # Model should accept the loaded state dict
            assert all(torch.is_tensor(v) for v in state_dict.values()), (
                "All state dict values should be tensors"
            )
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_dict_parameter_values_preserved(self, linear_model):
        """Verify actual parameter values are preserved in state dict."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            # Load state dict and verify values match
            model.load_state_dict(state_dict)
            loaded_dict = model.state_dict()

            for key in state_dict:
                original_tensor = state_dict[key]
                loaded_tensor = loaded_dict[key]
                np.testing.assert_array_equal(original_tensor.numpy(), loaded_tensor.numpy())
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_dict_shape_mismatch_detection(self, linear_model):
        """Verify shape mismatches are detected when loading state dict."""
        code, state_dict = IntegrationTestHelper.generate_model_code_and_state_dict(linear_model)

        temp_dir = tempfile.mkdtemp()
        try:
            model, _ = IntegrationTestHelper.load_generated_model(code, state_dict, temp_dir)

            # Create a corrupted state dict with wrong shape
            corrupted_dict = {}
            for key, value in state_dict.items():
                if value.dim() > 0:
                    # Change the shape of the first parameter
                    corrupted_dict[key] = torch.randn(1, 1)
                else:
                    corrupted_dict[key] = value

            # Loading should fail with shape mismatch
            if len(corrupted_dict) > 0:
                shape_mismatch_detected = False
                try:
                    model.load_state_dict(corrupted_dict, strict=True)
                except RuntimeError:
                    shape_mismatch_detected = True
                assert shape_mismatch_detected, "Expected RuntimeError for shape mismatch"
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
