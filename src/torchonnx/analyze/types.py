"""Stage 3: Semantic Type Definitions.

Defines semantic IR types with typed input/output containers.
All ONNX data is resolved to PyTorch tensors.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "ArgumentInfo",
    "ConstantInfo",
    "OperatorClass",
    "ParameterInfo",
    "SemanticLayerIR",
    "SemanticModelIR",
    "VariableInfo",
]

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


@dataclass(frozen=True)
class VariableInfo:
    """Runtime variable tensor (depends on inputs or previous operations).

    Appears in forward() as a runtime tensor variable.

    :param onnx_name: Original ONNX tensor name (for reference/tracing)
    :param code_name: Generated variable name in code (e.g., "x1", "x2")
    :param shape: Inferred shape (None if unknown/dynamic, may contain symbolic str dims)
    """

    onnx_name: str
    code_name: str
    shape: tuple[int | str, ...] | None


@dataclass(frozen=True)
class ParameterInfo:
    """Trainable parameter tensor -> nn.Parameter in __init__.

    Loaded from state_dict (large tensor data).

    :param onnx_name: Original ONNX initializer name (for reference/tracing)
    :param pytorch_name: PyTorch parameter name ("weight", "bias", "running_mean", etc.)
    :param code_name: Generated parameter name in code (e.g., "p1", "p2")
    :param shape: Parameter shape
    :param dtype: PyTorch dtype
    :param data: Resolved tensor data as PyTorch tensor
    """

    onnx_name: str
    pytorch_name: str
    code_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    data: torch.Tensor


@dataclass(frozen=True)
class ConstantInfo:
    """Static constant tensor -> register_buffer() in __init__.

    Loaded from state_dict (large tensor data).

    :param onnx_name: Original ONNX initializer name (for reference/tracing)
    :param code_name: Generated constant name in code (e.g., "c1", "c2")
    :param shape: Constant shape
    :param dtype: PyTorch dtype
    :param data: Resolved tensor data as PyTorch tensor
    """

    onnx_name: str
    code_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    data: torch.Tensor


@dataclass(frozen=True)
class ArgumentInfo:
    """Literal argument value (for __init__ or forward()).

    Appears as Python literal in generated code.
    NOT loaded from state_dict - embedded as literal.

    :param onnx_name: Original ONNX attribute name (None if synthetic/derived)
    :param pytorch_name: PyTorch argument name (e.g., "padding", "stride", "axes")
    :param value: Actual value (int, float, tuple, list, bool, str, etc.)
    :param default_value: PyTorch default value (None if no default exists)
    """

    onnx_name: str | None
    pytorch_name: str
    value: Any
    default_value: Any | None = None

    def is_default(self) -> bool:
        """Check if value matches PyTorch default.

        :return: True if argument can be omitted in code generation
        """
        return self.default_value is not None and self.value == self.default_value


class OperatorClass(Enum):
    """Classification of ONNX operators for PyTorch code generation.

    :cvar LAYER: Trainable module (Conv2d, Linear, BatchNorm2d)
    :cvar OPERATION: Stateless function (reshape, transpose, pad)
    :cvar OPERATOR: Stateless math operator (+, -, *, @)
    """

    LAYER = "layer"
    OPERATION = "operation"
    OPERATOR = "operator"


@dataclass(frozen=True)
class SemanticLayerIR:
    """Semantic IR for a single layer/operation.

    Combines structural info from Stage 2 with semantic classifications from Stage 3.

    :param name: Node name (for layer instance or variable naming)
    :param onnx_op_type: ONNX operator type (e.g., "Conv", "Add", "Reshape")
    :param pytorch_type: PyTorch type (e.g., "Conv2d", "Linear", "+", "reshape")
    :param operator_class: Layer, Operation, or Operator
    :param inputs: Ordered list of typed inputs (VariableInfo | ParameterInfo | ConstantInfo)
    :param outputs: Ordered list of typed outputs (all VariableInfo)
    :param arguments: Literal arguments (for __init__ or forward())
    """

    # Structural (from Stage 2)
    name: str
    onnx_op_type: str

    # Semantic (added by Stage 3)
    pytorch_type: str
    operator_class: OperatorClass

    # Classified Inputs/Outputs (single ordered list - preserves input order)
    inputs: list[VariableInfo | ParameterInfo | ConstantInfo]
    outputs: list[VariableInfo]

    # Arguments (unified - for both __init__ and forward())
    arguments: list[ArgumentInfo]


@dataclass(frozen=True)
class SemanticModelIR:
    """Semantic IR for complete ONNX model.

    Fully resolved - no ONNX dependencies (TensorProto, ModelProto).
    All tensor data converted to PyTorch tensors.

    :param layers: Semantically annotated layers
    :param parameters: All trainable parameters (with resolved PyTorch tensors)
    :param constants: All constant tensors (with resolved PyTorch tensors)
    :param variables: All runtime variables
    :param input_names: Model input tensor names (ONNX names)
    :param output_names: Model output tensor names (ONNX names)
    :param shapes: All tensor shapes (ONNX names as keys)
    """

    layers: list[SemanticLayerIR]
    parameters: list[ParameterInfo]
    constants: list[ConstantInfo]
    variables: list[VariableInfo]
    input_names: list[str]
    output_names: list[str]
    shapes: dict[str, tuple[int | str, ...] | None]
