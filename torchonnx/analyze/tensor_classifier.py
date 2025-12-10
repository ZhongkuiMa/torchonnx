"""Stage 3: Tensor Role Classification.

Classifies input/output tensors into typed containers (VariableInfo, ParameterInfo, ConstantInfo).
Converts ONNX tensor names to typed semantic information with generated code names.
"""

__docformat__ = "restructuredtext"
__all__ = ["classify_inputs", "classify_outputs"]

import torch
from onnx import TensorProto, numpy_helper

from .types import VariableInfo, ParameterInfo, ConstantInfo

# ONNX dtype to PyTorch dtype mapping
_ONNX_TO_TORCH_DTYPE = {
    1: torch.float32,  # FLOAT
    2: torch.uint8,  # UINT8
    3: torch.int8,  # INT8
    # 4: torch.uint16,  # UINT16  (not directly supported in PyTorch)
    5: torch.int16,  # INT16
    6: torch.int32,  # INT32
    7: torch.int64,  # INT64
    9: torch.bool,  # BOOL
    10: torch.float16,  # FLOAT16
    11: torch.float64,  # DOUBLE
    # 12: torch.uint32,  # UINT32 (not directly supported in PyTorch)
    # 13: torch.uint64,  # UINT64 (not directly supported in PyTorch)
}


def _tensor_proto_to_torch(tensor: TensorProto) -> torch.Tensor:
    """Convert ONNX TensorProto to PyTorch tensor.

    :param tensor: ONNX TensorProto
    :return: PyTorch tensor
    """
    # Use onnx.numpy_helper to convert to numpy first
    np_array = numpy_helper.to_array(tensor)

    # Convert to PyTorch tensor
    return torch.from_numpy(np_array.copy())


def _onnx_dtype_to_torch(onnx_dtype: int) -> torch.dtype:
    """Convert ONNX dtype code to PyTorch dtype.

    :param onnx_dtype: ONNX data type code
    :return: PyTorch dtype
    """
    return _ONNX_TO_TORCH_DTYPE.get(onnx_dtype, torch.float32)


def _get_parameter_role(
    onnx_name: str,
    input_names: list[str],
    pytorch_type: str,
) -> str | None:
    """Determine PyTorch parameter role based on input position and layer type.

    :param onnx_name: ONNX tensor name
    :param input_names: All input names for this node
    :param pytorch_type: PyTorch layer type (e.g., "nn.Conv2d" or "Conv2d")
    :return: Parameter role ("weight", "bias", etc.) or None if not a parameter
    """
    # Find position in input list
    try:
        idx = input_names.index(onnx_name)
    except ValueError:
        return None

    # Strip nn. prefix if present for comparison
    layer_type = pytorch_type
    if layer_type.startswith("nn."):
        layer_type = layer_type[3:]

    # Determine role based on layer type and position
    # First input (idx=0) is always the data input, not a parameter

    if layer_type in ("Conv2d", "Conv1d"):
        if idx == 1:
            return "weight"
        if idx == 2:
            return "bias"

    elif layer_type in ("ConvTranspose2d", "ConvTranspose1d"):
        if idx == 1:
            return "weight"
        if idx == 2:
            return "bias"

    elif layer_type == "Linear":
        if idx == 1:
            return "weight"
        if idx == 2:
            return "bias"

    elif layer_type == "BatchNorm2d":
        if idx == 1:
            return "weight"  # scale (gamma)
        if idx == 2:
            return "bias"  # bias (beta)
        if idx == 3:
            return "running_mean"
        if idx == 4:
            return "running_var"

    elif layer_type == "Upsample":
        # Note: Upsample scales/sizes are constructor args, not parameters
        # They should become constants, not parameters
        pass

    # Not a parameter for this layer
    return None


def classify_inputs(
    input_names: list[str],
    initializers: dict[str, TensorProto],
    pytorch_type: str,
    shapes: dict[str, tuple[int, ...] | None],
    code_name_counters: dict[str, int],
    variable_mapping: dict[str, str] | None = None,
    constant_mapping: dict[str, ConstantInfo] | None = None,
    node: "NodeProto | None" = None,
) -> list[VariableInfo | ParameterInfo | ConstantInfo]:
    """Classify each input name into Variable, Parameter, or Constant.

    Converts Stage 2's input_names (list of strings) into Stage 3's typed containers.
    Preserves input order.

    :param input_names: List of ONNX input tensor names (from Stage 2 NodeIR)
    :param initializers: All ONNX initializers (from Stage 2 ModelIR)
    :param pytorch_type: PyTorch layer type (to determine parameter roles)
    :param shapes: Tensor shapes (from Stage 2 ModelIR)
    :param code_name_counters: Mutable dict with keys 'var', 'param', 'const'
    :param variable_mapping: Optional mapping from onnx_name to code_name for variables
    :param constant_mapping: Optional mapping from onnx_name to ConstantInfo for constants
    :param node: Optional ONNX NodeProto for extracting attributes (e.g., transB for Gemm)
    :return: Ordered list of typed input info objects
    """
    results = []

    for onnx_name in input_names:
        # Skip empty strings (optional inputs not provided in ONNX)
        if not onnx_name:
            continue
        if onnx_name in initializers:
            # Static tensor - is it Parameter or Constant?
            tensor = initializers[onnx_name]

            # Determine if this is a parameter for this layer
            param_role = _get_parameter_role(onnx_name, input_names, pytorch_type)

            if param_role:
                # It's a trainable parameter
                code_name = f"p{code_name_counters['param']}"
                code_name_counters["param"] += 1

                torch_tensor = _tensor_proto_to_torch(tensor)

                # Transpose Linear layer weights from ONNX format to PyTorch format
                # ONNX Gemm behavior depends on transB attribute:
                #   - transB=0 (default): weight shape = (in_features, out_features) → needs transpose
                #   - transB=1: weight shape = (out_features, in_features) → already correct
                # PyTorch nn.Linear expects: weight shape = (out_features, in_features)
                if (pytorch_type in ["Linear", "nn.Linear"] and
                    param_role == "weight" and
                    torch_tensor.ndim == 2):
                    # Check transB attribute from ONNX node
                    trans_b = 0  # Default value
                    if node is not None:
                        from .attr_extractor import extract_onnx_attrs
                        attrs = extract_onnx_attrs(node, initializers)
                        trans_b = attrs.get("transB", 0)

                    # Only transpose if transB=0 (ONNX weight is in_features × out_features)
                    if trans_b == 0:
                        torch_tensor = torch_tensor.T

                results.append(
                    ParameterInfo(
                        onnx_name=onnx_name,
                        pytorch_name=param_role,
                        code_name=code_name,
                        shape=tuple(tensor.dims),
                        dtype=_onnx_dtype_to_torch(tensor.data_type),
                        data=torch_tensor,
                    )
                )
            else:
                # It's a constant - check if already exists
                if constant_mapping is not None and onnx_name in constant_mapping:
                    # Reuse existing ConstantInfo
                    results.append(constant_mapping[onnx_name])
                else:
                    # Create new constant
                    code_name = f"c{code_name_counters['const']}"
                    code_name_counters["const"] += 1

                    torch_tensor = _tensor_proto_to_torch(tensor)

                    const_info = ConstantInfo(
                        onnx_name=onnx_name,
                        code_name=code_name,
                        shape=tuple(tensor.dims),
                        dtype=_onnx_dtype_to_torch(tensor.data_type),
                        data=torch_tensor,
                    )
                    results.append(const_info)

                    # Cache for reuse
                    if constant_mapping is not None:
                        constant_mapping[onnx_name] = const_info
        else:
            # Runtime variable - check if it already exists
            if variable_mapping is not None and onnx_name in variable_mapping:
                # Reuse existing code_name
                code_name = variable_mapping[onnx_name]
            else:
                # Create new code_name
                code_name = f"x{code_name_counters['var']}"
                code_name_counters["var"] += 1
                if variable_mapping is not None:
                    variable_mapping[onnx_name] = code_name

            results.append(
                VariableInfo(
                    onnx_name=onnx_name,
                    code_name=code_name,
                    shape=shapes.get(onnx_name),
                )
            )

    return results


def classify_outputs(
    output_names: list[str],
    shapes: dict[str, tuple[int, ...] | None],
    code_name_counters: dict[str, int],
    variable_mapping: dict[str, str] | None = None,
) -> list[VariableInfo]:
    """Classify each output name into VariableInfo.

    All outputs are runtime variables.
    Converts Stage 2's output_names (list of strings) into Stage 3's VariableInfo list.

    :param output_names: List of ONNX output tensor names (from Stage 2 NodeIR)
    :param shapes: Tensor shapes (from Stage 2 ModelIR)
    :param code_name_counters: Mutable dict with key 'var'
    :param variable_mapping: Optional mapping from onnx_name to code_name for variables
    :return: Ordered list of VariableInfo objects
    """
    results = []

    for onnx_name in output_names:
        # Check if variable already exists (shouldn't for outputs, but keep consistent)
        if variable_mapping is not None and onnx_name in variable_mapping:
            code_name = variable_mapping[onnx_name]
        else:
            code_name = f"x{code_name_counters['var']}"
            code_name_counters["var"] += 1
            if variable_mapping is not None:
                variable_mapping[onnx_name] = code_name

        results.append(
            VariableInfo(
                onnx_name=onnx_name,
                code_name=code_name,
                shape=shapes.get(onnx_name),
            )
        )

    return results
