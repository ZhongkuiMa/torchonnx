"""Stage 1: ONNX to Intermediate Representation (IR) compiler.

This module implements the first stage of the 3-stage compiler architecture:
ONNX Model → Intermediate Representation (ModelIR)

The IR is a clean, typed representation that decouples ONNX specifics
from PyTorch code generation.
"""

__docformat__ = "restructuredtext"
__all__ = ["LayerIR", "ModelIR", "build_model_ir"]

from dataclasses import dataclass
from typing import Any

from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto

from ..naming import sanitize_layer_name
from ..type_inference import (
    identify_layer_parameters,
    infer_pytorch_layer_type,
    is_parametric_layer,
    map_onnx_to_pytorch_args,
)
from ..utils import extract_opset_version


@dataclass(frozen=True)
class LayerIR:
    """Intermediate representation of a PyTorch layer.

    This immutable dataclass stores all information needed to generate a PyTorch
    layer from an ONNX node, including clean layer names, constructor arguments,
    and parameter mappings.

    :ivar layer_name: Clean layer name (e.g., "conv1", "bn2", "fc3")
    :ivar layer_type: PyTorch layer type (e.g., "Conv2d", "Linear", "BatchNorm2d")
    :ivar constructor_args: PyTorch constructor arguments
    :ivar parameters: Mapping from parameter type to ONNX initializer name
    :ivar input_names: ONNX input tensor names
    :ivar output_names: ONNX output tensor names
    :ivar onnx_node: Original ONNX node for reference
    """

    layer_name: str
    layer_type: str
    constructor_args: dict[str, Any]
    parameters: dict[str, str]
    input_names: list[str]
    output_names: list[str]
    onnx_node: NodeProto


@dataclass(frozen=True)
class ModelIR:
    """Intermediate representation of entire model.

    This immutable dataclass stores all information needed to generate a complete
    PyTorch module from an ONNX model, including opset version, all layers,
    parameters, and the tensor dependency graph.

    :ivar opset_version: ONNX opset version (17-21)
    :ivar layers: All layers in topological order
    :ivar parameters: All ONNX initializers (weights, biases, etc.)
    :ivar inputs: Model input value infos
    :ivar outputs: Model output value infos
    :ivar tensor_graph: Tensor dependency graph mapping tensor names to consuming layers
    """

    opset_version: int
    layers: list[LayerIR]
    parameters: dict[str, TensorProto]
    inputs: list[ValueInfoProto]
    outputs: list[ValueInfoProto]
    tensor_graph: dict[str, list[str]]


def convert_constants_to_initializers(model: ModelProto) -> ModelProto:
    """Convert Constant nodes to initializers.

    ONNX Constant nodes contain constant tensors that don't change during inference.
    This function extracts these constants and adds them as initializers, then
    removes the Constant nodes from the graph.

    :param model: Input ONNX model
    :return: Modified ONNX model with Constant nodes converted to initializers
    """
    new_initializers = []
    nodes_to_remove = []

    for node in model.graph.node:
        if node.op_type != "Constant":
            continue

        if not node.output:
            continue

        output_name = node.output[0]

        for attr in node.attribute:
            if attr.name == "value":
                tensor = attr.t
                new_tensor = TensorProto()
                new_tensor.CopyFrom(tensor)
                new_tensor.name = output_name

                new_initializers.append(new_tensor)
                nodes_to_remove.append(node)
                break

    if not new_initializers:
        return model

    model_copy = ModelProto()
    model_copy.CopyFrom(model)

    for tensor in new_initializers:
        model_copy.graph.initializer.append(tensor)

    remaining_nodes = [n for n in model_copy.graph.node if n not in nodes_to_remove]

    del model_copy.graph.node[:]
    model_copy.graph.node.extend(remaining_nodes)

    return model_copy


def build_tensor_graph(
    model: ModelProto,
    layer_names: dict[str, str],
) -> dict[str, list[str]]:
    """Build tensor dependency graph.

    Maps each tensor name to the list of layer names that consume it.
    This is used to determine how tensors flow through the network.

    :param model: ONNX model
    :param layer_names: Dictionary mapping ONNX node names to layer names
    :return: Dictionary mapping tensor names to consuming layer names
    """
    tensor_graph: dict[str, list[str]] = {}

    for node in model.graph.node:
        node_name = node.name if node.name else node.output[0]
        layer_name = layer_names.get(node_name, node_name)

        for input_tensor in node.input:
            if input_tensor not in tensor_graph:
                tensor_graph[input_tensor] = []
            tensor_graph[input_tensor].append(layer_name)

    return tensor_graph


def build_layer_name_mapping(
    nodes: list[NodeProto],
    layer_names: list[str],
) -> dict[str, str]:
    """Build mapping from ONNX node names to layer names.

    :param nodes: List of ONNX nodes
    :param layer_names: List of generated layer names in same order
    :return: Dictionary mapping ONNX node names to layer names
    """
    mapping: dict[str, str] = {}

    for node, layer_name in zip(nodes, layer_names):
        node_name = node.name if node.name else node.output[0]
        mapping[node_name] = layer_name

    return mapping


def get_model_inputs(model: ModelProto) -> list[str]:
    """Get model input tensor names.

    :param model: ONNX model
    :return: List of input tensor names
    """
    return [input_info.name for input_info in model.graph.input]


def get_model_outputs(model: ModelProto) -> list[str]:
    """Get model output tensor names.

    :param model: ONNX model
    :return: List of output tensor names
    """
    return [output_info.name for output_info in model.graph.output]


def get_initializer_names(model: ModelProto) -> set[str]:
    """Get all initializer tensor names.

    :param model: ONNX model
    :return: Set of initializer tensor names
    """
    return {init.name for init in model.graph.initializer}


def build_layer_ir(
    node: NodeProto,
    initializers: dict[str, TensorProto],
    layer_name: str,
) -> LayerIR:
    """Build LayerIR for a single ONNX node.

    :param node: ONNX node
    :param initializers: All ONNX initializers
    :param layer_name: Generated layer name
    :return: LayerIR for this node
    """
    layer_type = infer_pytorch_layer_type(node)

    constructor_args: dict[str, Any] = {}
    if is_parametric_layer(layer_type) or layer_type in {"ReLU"}:
        try:
            constructor_args = map_onnx_to_pytorch_args(node, initializers)
        except ValueError:
            pass

    if layer_type == "Linear" and not constructor_args:
        layer_type = "Gemm"
    if layer_type == "Conv2d" and not constructor_args:
        layer_type = "Conv"
    if layer_type == "ConvTranspose2d" and not constructor_args:
        layer_type = "ConvTranspose"

    parameters: dict[str, str] = {}
    if is_parametric_layer(layer_type):
        parameters = identify_layer_parameters(node, initializers, layer_type)

    input_names = list(node.input)
    output_names = list(node.output)

    return LayerIR(
        layer_name=layer_name,
        layer_type=layer_type,
        constructor_args=constructor_args,
        parameters=parameters,
        input_names=input_names,
        output_names=output_names,
        onnx_node=node,
    )


def build_model_ir(model: ModelProto) -> ModelIR:
    """Build complete intermediate representation from ONNX model.

    This is the main entry point for Stage 1 compilation.

    :param model: ONNX model
    :return: ModelIR containing all layers and metadata
    """
    opset_version = extract_opset_version(model)

    model = convert_constants_to_initializers(model)

    initializers = {init.name: init for init in model.graph.initializer}

    initializer_names = get_initializer_names(model)

    nodes = [node for node in model.graph.node]

    counter: dict[str, int] = {}
    layer_names: list[str] = []
    for node in nodes:
        layer_type = infer_pytorch_layer_type(node)
        layer_name = sanitize_layer_name(node, layer_type, counter)
        layer_names.append(layer_name)

    layers: list[LayerIR] = []
    for node, layer_name in zip(nodes, layer_names):
        layer_ir = build_layer_ir(node, initializers, layer_name)
        layers.append(layer_ir)

    layer_name_mapping = build_layer_name_mapping(nodes, layer_names)
    tensor_graph = build_tensor_graph(model, layer_name_mapping)

    inputs = list(model.graph.input)
    outputs = list(model.graph.output)

    inputs_filtered = [inp for inp in inputs if inp.name not in initializer_names]

    return ModelIR(
        opset_version=opset_version,
        layers=layers,
        parameters=initializers,
        inputs=inputs_filtered,
        outputs=outputs,
        tensor_graph=tensor_graph,
    )
