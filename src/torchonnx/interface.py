"""Public ONNX model-interface inspection.

The conversion pipeline normalizes models before generating PyTorch code, but
callers that bind external specifications need the declared ONNX interface
before that normalization boundary.  This module exposes that interface
without importing query- or verifier-specific concepts.
"""

__docformat__ = "restructuredtext"
__all__ = ["ONNXModelInterface", "ONNXTensorInterface", "inspect_model_interface"]

from dataclasses import dataclass

from onnx import ModelProto, ValueInfoProto

ONNXDimension = int | str | None


@dataclass(frozen=True, slots=True)
class ONNXTensorInterface:
    """One ordered tensor declaration from an ONNX graph interface.

    ``shape=None`` means that the tensor has unknown rank.  An empty tuple is
    a scalar, while ``None`` inside a shape is an unknown dimension.  Static
    zero dimensions and symbolic dimensions are preserved verbatim.

    :param name: ONNX tensor name.
    :param elem_type: ONNX ``TensorProto.DataType`` integer value.
    :param shape: Exact declared shape, or ``None`` for unknown rank.
    """

    name: str
    elem_type: int
    shape: tuple[ONNXDimension, ...] | None


@dataclass(frozen=True, slots=True)
class ONNXModelInterface:
    """Ordered public interface and model-level compatibility metadata.

    :param inputs: Runtime data inputs in graph declaration order.  Legacy
        graph inputs backed by initializers are excluded.
    :param outputs: Graph outputs in declaration order.
    :param opset_imports: ``(domain, version)`` pairs in model order.
    :param ir_version: Declared ONNX IR version.
    :param has_local_functions: Whether the model defines local functions.
    """

    inputs: tuple[ONNXTensorInterface, ...]
    outputs: tuple[ONNXTensorInterface, ...]
    opset_imports: tuple[tuple[str, int], ...]
    ir_version: int
    has_local_functions: bool


def _inspect_tensor(value_info: ValueInfoProto) -> ONNXTensorInterface:
    """Convert one tensor ``ValueInfoProto`` without normalizing its shape."""
    if not value_info.type.HasField("tensor_type"):
        raise TypeError(f"ONNX interface value {value_info.name!r} is not a tensor")

    tensor_type = value_info.type.tensor_type
    shape: tuple[ONNXDimension, ...] | None
    if not tensor_type.HasField("shape"):
        shape = None
    else:
        dimensions: list[ONNXDimension] = []
        for dimension in tensor_type.shape.dim:
            value_kind = dimension.WhichOneof("value")
            if value_kind == "dim_value":
                dimensions.append(int(dimension.dim_value))
            elif value_kind == "dim_param":
                dimensions.append(dimension.dim_param)
            else:
                dimensions.append(None)
        shape = tuple(dimensions)

    return ONNXTensorInterface(
        name=value_info.name,
        elem_type=int(tensor_type.elem_type),
        shape=shape,
    )


def inspect_model_interface(model: ModelProto) -> ONNXModelInterface:
    """Inspect the declared interface of an ONNX model without mutating it.

    The function intentionally performs no shape inference or normalization:
    callers receive the exact rank, dimensions, order, opsets, and IR version
    declared by the supplied model.

    :param model: ONNX model to inspect.
    :return: Immutable interface description.
    :raises TypeError: If a graph input or output is not a tensor value.
    """
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    initializer_names.update(
        initializer.values.name for initializer in model.graph.sparse_initializer
    )

    return ONNXModelInterface(
        inputs=tuple(
            _inspect_tensor(value_info)
            for value_info in model.graph.input
            if value_info.name not in initializer_names
        ),
        outputs=tuple(_inspect_tensor(value_info) for value_info in model.graph.output),
        opset_imports=tuple((opset.domain, int(opset.version)) for opset in model.opset_import),
        ir_version=int(model.ir_version),
        has_local_functions=bool(model.functions),
    )
