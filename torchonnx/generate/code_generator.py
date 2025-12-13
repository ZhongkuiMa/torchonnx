"""Main PyTorch code generation orchestrator.

Assembles complete PyTorch module from semantic IR.
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_pytorch_module"]

import torch

from ._forward_gen import generate_forward_method, get_forward_gen_context, ForwardGenContext
from ._init_gen import generate_init_method, build_layer_name_mapping
from ._state_dict_gen import build_state_dict
from ._templates import MODULE_TEMPLATE
from ._utils import sanitize_identifier
from ..analyze import SemanticModelIR


def generate_pytorch_module(
    semantic_ir: SemanticModelIR,
    module_name: str = "ONNXModel",
) -> tuple[str, dict[str, torch.Tensor]]:
    """Generate complete PyTorch module from semantic IR.

    Creates a complete PyTorch nn.Module class with:
    - Imports
    - Class definition
    - __init__ method
    - forward() method
    - state_dict

    :param semantic_ir: Semantic IR from Stage 3/4
    :param module_name: Name for the generated class
    :return: Tuple of (module_code_string, state_dict)
    """
    # Sanitize class name
    class_name = sanitize_identifier(module_name)

    # Build layer name mapping (shared between init, forward, and state_dict)
    layer_name_mapping = build_layer_name_mapping(semantic_ir)

    # Generate imports
    imports = _generate_imports(semantic_ir)

    # Generate forward() method FIRST - this populates the context with helper usage info
    forward_method, forward_context = _generate_forward_with_context(
        semantic_ir, layer_name_mapping
    )

    # Generate helper functions based on actual usage (not just op type existence)
    helpers = _generate_helpers_from_context(forward_context)

    # Generate __init__ method with all constants (Stage 6 will remove unused ones)
    init_method = generate_init_method(semantic_ir, layer_name_mapping)

    # Assemble complete module
    module_code = MODULE_TEMPLATE.format(
        imports=imports,
        helpers=helpers,
        class_name=class_name,
        init_method=init_method,
        forward_method=forward_method,
    )

    # Build state_dict (with layer name mapping for correct keys, all constants included)
    state_dict = build_state_dict(semantic_ir, layer_name_mapping)

    return module_code, state_dict


def _generate_forward_with_context(
    semantic_ir: SemanticModelIR,
    layer_name_mapping: dict[str, str],
) -> tuple[str, ForwardGenContext]:
    """Generate forward method and return the context with helper usage info.

    :param semantic_ir: Semantic IR
    :param layer_name_mapping: Layer name mapping
    :return: Tuple of (forward_method_code, context)
    """
    # Generate forward method
    forward_method = generate_forward_method(semantic_ir, layer_name_mapping)

    # Analyze the IR to determine which helpers are actually needed
    # (this mirrors the logic in the handlers but is more reliable since
    # the context is cleaned up after forward generation)
    helper_context = _get_helper_needs_from_ir(semantic_ir)

    return forward_method, helper_context


def _get_helper_needs_from_ir(semantic_ir: SemanticModelIR) -> ForwardGenContext:
    """Determine helper needs by analyzing the IR.

    This is a fallback that analyzes the IR directly to determine which
    helpers are actually needed after code generation optimizations.

    :param semantic_ir: Semantic IR
    :return: Context with helper needs flags
    """
    from ._forward_gen import ForwardGenContext
    from ..analyze import ConstantInfo

    ctx = ForwardGenContext()

    for layer in semantic_ir.layers:
        if layer.onnx_op_type == "Slice":
            # Check if this slice can be handled without helper
            if len(layer.inputs) >= 3:
                starts_input = layer.inputs[1]
                ends_input = layer.inputs[2]
                axes_input = layer.inputs[3] if len(layer.inputs) > 3 else None
                steps_input = layer.inputs[4] if len(layer.inputs) > 4 else None

                # All constants -> native slicing, no helper needed
                all_constants = (
                    isinstance(starts_input, ConstantInfo)
                    and isinstance(ends_input, ConstantInfo)
                    and (axes_input is None or isinstance(axes_input, ConstantInfo))
                    and (steps_input is None or isinstance(steps_input, ConstantInfo))
                )
                if all_constants:
                    continue

                # Check if single-axis with step=1 and constant bounds -> torch.narrow, no helper needed
                # We only use narrow when ALL parameters are constant because ONNX Slice
                # has bounds clipping that torch.narrow doesn't handle.
                axes_constant = axes_input is None or isinstance(axes_input, ConstantInfo)
                steps_constant = steps_input is None or isinstance(steps_input, ConstantInfo)
                starts_constant = isinstance(starts_input, ConstantInfo)
                ends_constant = isinstance(ends_input, ConstantInfo)

                if axes_constant and steps_constant and starts_constant and ends_constant:
                    if axes_input is None:
                        axes_list = [0]
                    else:
                        axes_list = axes_input.data.tolist()
                        if not isinstance(axes_list, list):
                            axes_list = [axes_list]

                    if steps_input is None:
                        steps_list = [1] * len(axes_list)
                    else:
                        steps_list = steps_input.data.tolist()
                        if not isinstance(steps_list, list):
                            steps_list = [steps_list]

                    # Single axis with step=1 and constant bounds uses torch.narrow
                    if len(axes_list) == 1 and steps_list[0] == 1:
                        start_val = int(starts_input.data.item())
                        end_val = int(ends_input.data.item())
                        if end_val - start_val > 0:
                            continue

                # Otherwise needs helper
                ctx.needs_dynamic_slice = True

        elif layer.onnx_op_type == "ScatterND":
            ctx.needs_scatter_nd = True

        elif layer.onnx_op_type == "Expand":
            # Check if this expand can be handled without helper
            if len(layer.inputs) >= 2:
                shape_input = layer.inputs[1]

                # Check output shape from inference
                output_info = layer.outputs[0]
                output_shape = output_info.shape if output_info else None

                # Known output shape with all integers -> reshape, no helper
                if output_shape and all(isinstance(dim, int) for dim in output_shape):
                    continue

                # Constant shape with known data shape -> inline expand
                if isinstance(shape_input, ConstantInfo):
                    data_input = layer.inputs[0]
                    if hasattr(data_input, 'shape') and data_input.shape:
                        data_shape = data_input.shape
                        if all(isinstance(d, int) for d in data_shape):
                            continue

                # Otherwise needs helper
                ctx.needs_dynamic_expand = True

    return ctx


def _generate_imports(semantic_ir: SemanticModelIR) -> str:
    """Generate import statements based on operations used.

    :param semantic_ir: Semantic IR
    :return: Import statements string
    """
    imports = [
        "import torch",
        "import torch.nn as nn",
    ]

    # Check if F.* operations are used
    needs_functional = any(
        layer.pytorch_type.startswith("F.") for layer in semantic_ir.layers
    )

    if needs_functional:
        imports.append("import torch.nn.functional as F")

    return "\n".join(imports)


def _generate_helpers_from_context(ctx: ForwardGenContext) -> str:
    """Generate helper functions based on actual usage tracked in context.

    Only emits helpers that are actually called in the generated code,
    not just based on ONNX op type existence.

    :param ctx: Forward generation context with helper usage flags
    :return: Helper function definitions string
    """
    helpers = []

    if ctx.needs_dynamic_slice:
        helpers.append(DYNAMIC_SLICE_HELPER)

    if ctx.needs_scatter_nd:
        helpers.append(SCATTER_ND_HELPER)

    if ctx.needs_dynamic_expand:
        helpers.append(EXPAND_HELPER)

    return "\n\n".join(helpers)


def _generate_helpers(semantic_ir: SemanticModelIR) -> str:
    """Generate helper functions for operations without direct PyTorch equivalents.

    DEPRECATED: Use _generate_helpers_from_context instead for more precise helper emission.

    :param semantic_ir: Semantic IR
    :return: Helper function definitions string
    """
    helpers = []

    # Check which helpers are needed
    onnx_types = {layer.onnx_op_type for layer in semantic_ir.layers}

    if "Slice" in onnx_types:
        helpers.append(DYNAMIC_SLICE_HELPER)

    if "ScatterND" in onnx_types:
        helpers.append(SCATTER_ND_HELPER)

    if "Expand" in onnx_types:
        helpers.append(EXPAND_HELPER)

    return "\n\n".join(helpers)


# Helper function templates
DYNAMIC_SLICE_HELPER = '''\
def dynamic_slice(data, starts, ends, axes=None, steps=None):
    """Dynamic slice helper for ONNX Slice operation."""
    # Ensure tensor
    starts = torch.as_tensor(starts, device=data.device)
    ends   = torch.as_tensor(ends,   device=data.device)
    if axes is None:
        axes = torch.arange(starts.numel(), device=data.device)
    else:
        axes = torch.as_tensor(axes, device=data.device)
    if steps is None:
        steps = torch.ones_like(starts, device=starts.device)
    else:
        steps = torch.as_tensor(steps, device=data.device)

    # Normalize negative starts/ends
    dims = torch.as_tensor(data.shape, device=data.device)
    # axes tells where to read dim size
    dim_sizes = dims[axes]

    starts = torch.where(starts < 0, dim_sizes + starts, starts)
    ends   = torch.where(ends   < 0, dim_sizes + ends,   ends)

    # Clip to bounds (ONNX semantics)
    # Use tensors for both min and max to avoid type mismatch
    zero = torch.zeros_like(dim_sizes, device=dim_sizes.device)
    starts = torch.clamp(starts, min=zero, max=dim_sizes)
    ends   = torch.clamp(ends,   min=zero, max=dim_sizes)

    # Build index tuple dynamically
    index = [slice(None)] * data.ndim
    for i in range(axes.shape[0]):
        ax = axes[i].item()
        idx = torch.arange(starts[i], ends[i], steps[i], device=data.device)
        index[ax] = idx

    return data[tuple(index)]'''


SCATTER_ND_HELPER = '''\
def scatter_nd(data, indices, updates, reduction="none"):
    """PyTorch equivalent of ONNX ScatterND using tensor-based indexing.

    indices: (..., K)
    updates: broadcastable to indices[..., :-1] + data.shape[K:]
    reduction: "none", "add" (ONNX opset >= 11)
    """
    result = data.clone()
    indices = indices.to(torch.long)

    # Ensure updates has the same dtype as result (important for float64 testing)
    updates = updates.to(result.dtype)

    # Convert indices (..., K) -> tuple of K tensors
    idx_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))

    if reduction == "none":
        result.index_put_(idx_tuple, updates)
    elif reduction == "add":
        result.index_put_(idx_tuple, updates, accumulate=True)
    else:
        raise NotImplementedError(f"Unsupported reduction: {reduction}")

    return result'''


EXPAND_HELPER = '''\
def dynamic_expand(data, target_shape):
    """Dynamic expand helper for ONNX Expand operation.

    Handles dimension mismatches and ONNX semantics conversion:
    - ONNX allows expanding from higher dims to lower dims (e.g., 4D->3D)
    - ONNX uses 1 to mean "keep dimension", PyTorch uses -1
    """
    # Ensure target_shape is a list of integers
    if isinstance(target_shape, torch.Tensor):
        target_shape = target_shape.to(torch.int64).tolist()
    # Convert to integers (handle cases where list contains floats or numpy ints)
    target_shape = [int(x) for x in target_shape]

    # If data has more dimensions than target, squeeze leading dimensions
    if data.ndim > len(target_shape):
        # Remove leading dimensions by reshaping to match target length
        new_shape = tuple(int(s) for s in data.shape[data.ndim - len(target_shape):])
        data = data.reshape(new_shape)

    # Convert ONNX semantics to PyTorch
    # For each dimension in target_shape:
    # - If target[i] == 1 and data has a non-1 dimension at that position, keep data's dim (-1)
    # - Otherwise, use target[i]
    converted_shape = []
    for i in range(len(target_shape)):
        if i < len(target_shape) - data.ndim:
            # Dimension doesn't exist in data, use target value
            converted_shape.append(int(target_shape[i]))
        else:
            # Dimension exists in data
            data_idx = i - (len(target_shape) - data.ndim)
            if target_shape[i] == 1 and data.shape[data_idx] != 1:
                # Keep data's dimension
                converted_shape.append(-1)
            else:
                # Use target dimension
                converted_shape.append(int(target_shape[i]))

    return data.expand(converted_shape)'''
