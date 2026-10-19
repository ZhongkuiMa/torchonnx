from dataclasses import FrozenInstanceError

import onnx
import pytest

from torchonnx import ONNXModelInterface, ONNXTensorInterface, inspect_model_interface


def _unknown_rank_value_info(name: str) -> onnx.ValueInfoProto:
    value_info = onnx.ValueInfoProto()
    value_info.name = name
    value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    return value_info


def test_inspect_model_interface_preserves_exact_tensor_declarations() -> None:
    unknown_rank = _unknown_rank_value_info("unknown_rank")
    scalar = onnx.helper.make_tensor_value_info("scalar", onnx.TensorProto.DOUBLE, [])
    shaped = onnx.helper.make_tensor_value_info(
        "shaped", onnx.TensorProto.FLOAT16, [0, "batch", None]
    )
    output = onnx.helper.make_tensor_value_info("result", onnx.TensorProto.INT64, ["items"])
    graph = onnx.helper.make_graph([], "interface", [unknown_rank, scalar, shaped], [output])
    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 20),
            onnx.helper.make_opsetid("custom.domain", 3),
        ],
    )

    interface = inspect_model_interface(model)

    assert interface.inputs == (
        ONNXTensorInterface("unknown_rank", onnx.TensorProto.FLOAT, None),
        ONNXTensorInterface("scalar", onnx.TensorProto.DOUBLE, ()),
        ONNXTensorInterface("shaped", onnx.TensorProto.FLOAT16, (0, "batch", None)),
    )
    assert interface.outputs == (ONNXTensorInterface("result", onnx.TensorProto.INT64, ("items",)),)
    assert interface.opset_imports == (("", 20), ("custom.domain", 3))
    assert interface.ir_version == model.ir_version
    assert not interface.has_local_functions


def test_inspect_model_interface_excludes_initializer_backed_inputs() -> None:
    data = onnx.helper.make_tensor_value_info("data", onnx.TensorProto.FLOAT, [2])
    legacy_weight_input = onnx.helper.make_tensor_value_info("weight", onnx.TensorProto.FLOAT, [2])
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2])
    weight = onnx.helper.make_tensor("weight", onnx.TensorProto.FLOAT, dims=[2], vals=[1.0, 2.0])
    node = onnx.helper.make_node("Add", ["data", "weight"], ["output"])
    graph = onnx.helper.make_graph(
        [node], "legacy_initializer_input", [data, legacy_weight_input], [output], [weight]
    )
    model = onnx.helper.make_model(graph)

    interface = inspect_model_interface(model)

    assert tuple(item.name for item in interface.inputs) == ("data",)


def test_inspect_model_interface_reports_local_functions() -> None:
    input_info = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
    output_info = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
    graph = onnx.helper.make_graph([], "with_function", [input_info], [output_info])
    model = onnx.helper.make_model(graph)
    function = onnx.helper.make_function(
        "custom.domain",
        "LocalIdentity",
        ["x"],
        ["y"],
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        [onnx.helper.make_opsetid("", 20)],
    )
    model.functions.append(function)

    assert inspect_model_interface(model).has_local_functions


def test_model_interface_is_frozen() -> None:
    interface = ONNXModelInterface(
        inputs=(),
        outputs=(),
        opset_imports=(),
        ir_version=10,
        has_local_functions=False,
    )

    with pytest.raises(FrozenInstanceError):
        interface.ir_version = 11  # type: ignore[misc]


def test_inspect_model_interface_rejects_non_tensor_io() -> None:
    sequence = onnx.ValueInfoProto()
    sequence.name = "sequence"
    sequence.type.sequence_type.elem_type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    output = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])
    model = onnx.helper.make_model(onnx.helper.make_graph([], "non_tensor", [sequence], [output]))

    with pytest.raises(TypeError, match=r"sequence.*not a tensor"):
        inspect_model_interface(model)
