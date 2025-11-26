__docformat__ = "restructuredtext"
__all__ = [
    "get_onnx_input_nodes",
    "get_onnx_output_nodes",
    "get_onnx_model_input_names",
    "get_onnx_model_output_names",
    "get_onnx_nodes",
    "get_onnx_initializers",
    "get_onnx_initializer_names",
    "get_onnx_model_shapes",
    "extract_onnx_opset_version",
]

from onnx import ValueInfoProto, ModelProto, NodeProto, TensorProto


def get_onnx_input_nodes(model: ModelProto) -> list[ValueInfoProto]:
    """Get model input ValueInfoProto objects.

    :param model: ONNX model
    :return: List of input ValueInfoProto
    """
    initializers = get_onnx_initializers(model)
    # Exclude inputs that are actually initializers (weights/biases)
    model_inputs = [inp for inp in model.graph.input if inp.name not in initializers]
    return model_inputs


def get_onnx_output_nodes(model: ModelProto) -> list[ValueInfoProto]:
    """Get model output ValueInfoProto objects.

    :param model: ONNX model
    :return: List of output ValueInfoProto
    """
    return [out for out in model.graph.output]


def get_onnx_model_input_names(model: ModelProto) -> list[str]:
    """Get model input tensor names.

    :param model: ONNX model
    :return: List of input tensor names
    """
    model_inputs = get_onnx_input_nodes(model)
    return [input_info.name for input_info in model_inputs]


def get_onnx_model_output_names(model: ModelProto) -> list[str]:
    """Get model output tensor names.

    :param model: ONNX model
    :return: List of output tensor names
    """
    return [output_info.name for output_info in model.graph.output]


def get_onnx_nodes(model: ModelProto) -> list[NodeProto]:
    """Get all nodes in the model graph.

    :param model: ONNX model
    :return: List of ONNX nodes
    """
    return [node for node in model.graph.node]


def get_onnx_initializers(model: ModelProto) -> dict[str, TensorProto]:
    """Get all initializer tensors.

    :param model: ONNX model
    :return: Dictionary mapping initializer tensor names to TensorProto
    """
    return {init.name: init for init in model.graph.initializer}


def get_onnx_initializer_names(model: ModelProto) -> set[str]:
    """Get all initializer tensor names.

    :param model: ONNX model
    :return: Set of initializer tensor names
    """
    return {init.name for init in model.graph.initializer}


def get_onnx_model_shapes(
    model: ModelProto, set_batch_to_one: bool = True
) -> dict[str, tuple[int | str, ...] | None]:
    """Get shapes of all input and output tensors in the ONNX model."""
    shapes: dict[str, tuple[int | str, ...]] = {}

    for input_info in model.graph.input:
        shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        shapes[input_info.name] = tuple(shape)

    for output_info in model.graph.output:
        shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
        shapes[output_info.name] = tuple(shape)

    def _get_shape(vi) -> tuple[int | str] | None:
        dims = []
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_value > 0:  # static dimension
                dims.append(d.dim_value)
            elif d.dim_param:  # dynamic dimension
                if set_batch_to_one:
                    dims.append(1)
                else:
                    dims.append(d.dim_param)  # e.g. "batch"
            else:  # unknown dimension
                return None
        return tuple(dims)

    for value_info in model.graph.value_info:
        shape = _get_shape(value_info)
        shapes[value_info.name] = shape

    return shapes


def extract_onnx_opset_version(model: ModelProto) -> int:
    """Extract ONNX opset version from model.

    :param model: ONNX model
    :return: Opset version
    """
    if not model.opset_import:
        raise ValueError("Model has no opset_import")

    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            return opset.version

    raise ValueError("Model has no primary opset (domain='' or 'ai.onnx')")
