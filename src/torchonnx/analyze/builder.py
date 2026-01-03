"""Stage 3: Semantic IR Builder.

Builds semantic intermediate representation by converting Stage 2's structural IR
to PyTorch-style typed containers with resolved tensor data.
"""

__docformat__ = "restructuredtext"
__all__ = ["build_semantic_ir"]

from torchonnx.analyze.tensor_classifier import classify_inputs, classify_outputs
from torchonnx.analyze.type_mapping import (
    convert_to_pytorch_type,
    extract_layer_args,
    extract_operation_args,
    is_layer_with_args,
    is_operation,
    is_operator,
)
from torchonnx.analyze.types import (
    ArgumentInfo,
    ConstantInfo,
    OperatorClass,
    ParameterInfo,
    SemanticLayerIR,
    SemanticModelIR,
    VariableInfo,
)
from torchonnx.build import ModelIR, NodeIR


def _classify_operator_class(pytorch_type: str) -> OperatorClass:
    """Determine operator class from PyTorch type.

    :param pytorch_type: PyTorch type string
    :return: Operator classification
    """
    if is_operator(pytorch_type):
        return OperatorClass.OPERATOR
    if is_operation(pytorch_type):
        return OperatorClass.OPERATION
    # If it has args, it's a layer
    if is_layer_with_args(pytorch_type):
        return OperatorClass.LAYER
    # Fallback to operation for stateless types
    return OperatorClass.OPERATION


def _build_semantic_layer_ir(
    layer_ir: NodeIR,
    model_ir: ModelIR,
    code_name_counters: dict[str, int],
    variable_mapping: dict[str, str],
    constant_mapping: dict[str, ConstantInfo],
) -> SemanticLayerIR:
    """Convert raw NodeIR to semantic LayerIR with typed containers.

    :param layer_ir: Raw node IR from Stage 2
    :param model_ir: Complete model IR
    :param code_name_counters: Shared counters for code name generation
    :param variable_mapping: Shared mapping from onnx_name to code_name for variables
    :return: Semantic layer IR with typed inputs/outputs
    """
    # Get PyTorch type from ONNX node
    pytorch_type = convert_to_pytorch_type(layer_ir.node, model_ir.initializers)

    # Classify operator type
    operator_class = _classify_operator_class(pytorch_type)

    # Extract arguments for constructor/function call
    arguments = []
    if operator_class == OperatorClass.LAYER and is_layer_with_args(pytorch_type):
        raw_args = extract_layer_args(layer_ir.node, model_ir.initializers)
        # Convert dict to list[ArgumentInfo]
        # Note: extract_layer_args should be updated to return list[ArgumentInfo] directly
        # For now, convert the dict format
        for arg_name, arg_value in raw_args.items():
            arguments.append(
                ArgumentInfo(
                    onnx_name=None,  # Not directly from ONNX attribute
                    pytorch_name=arg_name,
                    value=arg_value,
                    default_value=None,
                )
            )
    elif operator_class == OperatorClass.OPERATION:
        # Extract arguments for operations (Cast, Gather, Slice, etc.)
        raw_args = extract_operation_args(
            layer_ir.node, model_ir.initializers, layer_ir.onnx_op_type
        )
        for arg_name, arg_value in raw_args.items():
            arguments.append(
                ArgumentInfo(
                    onnx_name=None,
                    pytorch_name=arg_name,
                    value=arg_value,
                    default_value=None,
                )
            )

    # Classify inputs into typed containers (single ordered list)
    inputs = classify_inputs(
        input_names=layer_ir.input_names,
        initializers=model_ir.initializers,
        pytorch_type=pytorch_type,
        shapes=model_ir.shapes,
        code_name_counters=code_name_counters,
        variable_mapping=variable_mapping,
        constant_mapping=constant_mapping,
        node=layer_ir.node,
    )

    # Classify outputs (all are variables)
    outputs = classify_outputs(
        output_names=layer_ir.output_names,
        shapes=model_ir.shapes,
        code_name_counters=code_name_counters,
        variable_mapping=variable_mapping,
    )

    return SemanticLayerIR(
        # Structural (from Stage 2)
        name=layer_ir.name,
        onnx_op_type=layer_ir.onnx_op_type,
        # Semantic (added by Stage 3)
        pytorch_type=pytorch_type,
        operator_class=operator_class,
        # Typed inputs/outputs (single ordered lists)
        inputs=inputs,
        outputs=outputs,
        # Arguments
        arguments=arguments,
    )


def build_semantic_ir(model_ir: ModelIR) -> SemanticModelIR:
    """Build semantic IR from raw IR.

    Converts Stage 2's structural IR to Stage 3's semantic IR:
    - Maps ONNX operators to PyTorch types
    - Classifies operators (Layer/Operation/Operator)
    - Converts string lists to typed containers (VariableInfo, ParameterInfo, etc.)
    - Generates code names (x0, x1, p0, p1, c0, c1)
    - Resolves ONNX tensors to torch.Tensor
    - Removes ONNX dependencies (no ModelProto, no initializers)

    :param model_ir: Raw model IR from Stage 2
    :return: Semantic model IR with fully resolved PyTorch data
    """
    # Initialize global code name counters
    code_name_counters = {
        "var": 0,
        "param": 0,
        "const": 0,
    }

    # Initialize variable mapping (onnx_name -> code_name)
    variable_mapping: dict[str, str] = {}

    # Initialize constant mapping (onnx_name -> ConstantInfo) for reuse
    constant_mapping: dict[str, ConstantInfo] = {}

    # Build semantic layers
    semantic_layers: list[SemanticLayerIR] = []
    all_parameters: list[ParameterInfo] = []
    all_constants: list[ConstantInfo] = []
    all_variables: list[VariableInfo] = []

    # Process model inputs first (they are variables)
    for input_name in model_ir.input_names:
        code_name = f"x{code_name_counters['var']}"
        code_name_counters["var"] += 1
        variable_mapping[input_name] = code_name
        all_variables.append(
            VariableInfo(
                onnx_name=input_name,
                code_name=code_name,
                shape=model_ir.shapes.get(input_name),
            )
        )

    # Process each layer
    for layer_ir in model_ir.layers:
        semantic_layer = _build_semantic_layer_ir(
            layer_ir, model_ir, code_name_counters, variable_mapping, constant_mapping
        )
        semantic_layers.append(semantic_layer)

        # Collect parameters and constants from inputs
        for typed_input in semantic_layer.inputs:
            if isinstance(typed_input, ParameterInfo) and not any(
                p.onnx_name == typed_input.onnx_name for p in all_parameters
            ):
                all_parameters.append(typed_input)
            elif isinstance(typed_input, ConstantInfo) and not any(
                c.onnx_name == typed_input.onnx_name for c in all_constants
            ):
                all_constants.append(typed_input)

        # Collect all variables (outputs and intermediate variables)
        for var in semantic_layer.outputs:
            if not any(v.onnx_name == var.onnx_name for v in all_variables):
                all_variables.append(var)

    return SemanticModelIR(
        layers=semantic_layers,
        parameters=all_parameters,
        constants=all_constants,
        variables=all_variables,
        input_names=model_ir.input_names,
        output_names=model_ir.output_names,
        shapes=model_ir.shapes,
        # REMOVED: initializers, model (no ONNX dependencies in Stage 3)
    )
