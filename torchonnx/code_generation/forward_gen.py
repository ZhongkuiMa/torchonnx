"""forward() method code generation for PyTorch modules.

This module generates the forward() method for converted PyTorch modules,
implementing the tensor flow graph from the ONNX model.
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_forward_method"]

from ..ir import LayerIR
from ..type_inference import (
    extract_functional_args,
    get_functional_operation_template,
    get_functional_operator,
    is_functional_operation,
    is_functional_operation_with_args,
)


def generate_operation_code(
    layer: LayerIR,
    output_var: str,
    input_vars: list[str],
    name_mapping: dict[str, str] | None = None,
    initializers: dict | None = None,
    var_source_layers: dict[str, str] | None = None,
) -> str:
    """Generate code for a single operation.

    Handles layers, functional operations, and functional operations with arguments.

    :param layer: Layer IR
    :param output_var: Output variable name
    :param input_vars: List of input variable names
    :param name_mapping: Mapping from ONNX names to simplified parameter names
    :param initializers: ONNX initializers (needed for functional args extraction)
    :param var_source_layers: Mapping from variable names to their source layer types
    :return: Python code string for this operation
    """
    if is_functional_operation(layer.layer_type):
        operator = get_functional_operator(layer.layer_type)

        if layer.layer_type == "MatMul":
            return f"    {output_var} = {input_vars[0]} @ {input_vars[1]}"

        if len(input_vars) == 2:
            return f"    {output_var} = {input_vars[0]} {operator} {input_vars[1]}"
        elif len(input_vars) == 1:
            return f"    {output_var} = {operator}{input_vars[0]}"
        else:
            raise ValueError(
                f"Unexpected number of inputs for {layer.layer_type}: {len(input_vars)}"
            )

    elif is_functional_operation_with_args(layer.layer_type):
        template = get_functional_operation_template(layer.layer_type)

        if layer.layer_type == "Pad":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            pad = func_args.get("pad", [])
            mode = func_args.get("mode", "constant")
            return f'    {output_var} = F.pad({input_vars[0]}, {pad}, mode="{mode}")'

        elif layer.layer_type == "Flatten":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            start_dim = func_args.get("start_dim", 1)
            return f"    {output_var} = torch.flatten({input_vars[0]}, start_dim={start_dim})"

        elif layer.layer_type == "Reshape":
            shape_var = input_vars[1]
            if shape_var.startswith("self."):
                shape_var = f"tuple({shape_var}.data.flatten().int().tolist())"
            else:
                shape_var = f"tuple({shape_var}.flatten().int().tolist())"
            return f"    {output_var} = {input_vars[0]}.reshape({shape_var})"

        elif layer.layer_type == "Transpose":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            perm = func_args.get("perm", [])
            return f"    {output_var} = {input_vars[0]}.permute({perm})"

        elif layer.layer_type == "Squeeze":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            dim = func_args.get("dim")

            # For ONNX opset >= 13, axes come as an input (not attribute)
            if dim is None and len(input_vars) > 1:
                axes_var = input_vars[1]
                # If axes is a parameter, extract the value at runtime
                if axes_var.startswith("self."):
                    # For single axis, use .item() to get scalar
                    # For multiple axes, use tuple comprehension
                    return f"    {output_var} = {input_vars[0]}.squeeze(dim=int({axes_var}.item()) if {axes_var}.numel() == 1 else None)"
                else:
                    return f"    {output_var} = {input_vars[0]}.squeeze(dim=int({axes_var}.item()) if {axes_var}.numel() == 1 else None)"
            elif dim is not None:
                return f"    {output_var} = {input_vars[0]}.squeeze(dim={dim})"
            else:
                return f"    {output_var} = {input_vars[0]}.squeeze()"

        elif layer.layer_type == "Unsqueeze":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            dim = func_args.get("dim")

            # For ONNX opset >= 13, axes come as an input (not attribute)
            if dim is None and len(input_vars) > 1:
                axes_var = input_vars[1]
                # If axes is a parameter, extract the value at runtime
                if axes_var.startswith("self."):
                    # Generate code to extract axes value and use it
                    return f"    {output_var} = {input_vars[0]}.unsqueeze(int({axes_var}.item()))"
                else:
                    # Axes is a variable, use it directly
                    return f"    {output_var} = {input_vars[0]}.unsqueeze(int({axes_var}.item()))"
            elif dim is not None:
                return f"    {output_var} = {input_vars[0]}.unsqueeze(dim={dim})"
            else:
                # Fallback to dimension 0 (should rarely happen)
                return f"    {output_var} = {input_vars[0]}.unsqueeze(0)"

        elif layer.layer_type == "Shape":
            return f"    {output_var} = torch.tensor(list({input_vars[0]}.shape), dtype=torch.int64)"

        elif layer.layer_type == "Gather":
            axis = 0
            for attr in layer.onnx_node.attribute:
                if attr.name == "axis":
                    axis = attr.i
            indices_var = input_vars[1]

            # Try to determine index shape from initializers (for static constant indices)
            index_shape = None
            if indices_var.startswith("self.") and initializers:
                # Extract the param name without "self."
                var_param_name = indices_var[5:]  # Remove "self."

                for init_name, init_tensor in initializers.items():
                    # Try exact match first (normalized)
                    param_name = (
                        init_name.split("/")[-1]
                        .replace(".", "_")
                        .replace("-", "_")
                        .lower()
                    )
                    if var_param_name == param_name:
                        index_shape = (
                            init_tensor.shape
                            if hasattr(init_tensor, "shape")
                            else init_tensor.dims
                        )
                        break

                # If not found, try fuzzy match (param name appears in init name)
                if index_shape is None:
                    for init_name, init_tensor in initializers.items():
                        # Normalize full init name and check if var_param_name is in it
                        normalized_full = (
                            init_name.replace("/", "_")
                            .replace(".", "_")
                            .replace("-", "_")
                            .lower()
                        )
                        if var_param_name in normalized_full:
                            index_shape = (
                                init_tensor.shape
                                if hasattr(init_tensor, "shape")
                                else init_tensor.dims
                            )
                            break

            # Generate code based on what we know
            if index_shape is not None:
                # STATIC INDEX: Shape determined from initializers
                if len(index_shape) == 0:
                    # Scalar (0-D) - use .select() for dimension reduction
                    return f"    {output_var} = {input_vars[0]}.select({axis}, int({indices_var}.item()))"
                elif len(index_shape) == 1:
                    # 1-D index (including (1,)) - use index_select (preserves dimension)
                    # ONNX Gather with (1,) index returns (1,) output, not scalar
                    return f"    {output_var} = torch.index_select({input_vars[0]}, {axis}, {indices_var}.long())"
                else:
                    # Higher dimensional - use torch.gather
                    return f"    {output_var} = torch.gather({input_vars[0]}, {axis}, {indices_var}.long())"
            else:
                # DYNAMIC INDEX: Shape unknown - use safe fallback with runtime check
                if indices_var.startswith("self."):
                    # Unknown static parameter - check dimensionality at runtime
                    # For scalar (0-D) tensors, use .select() which handles negative indices
                    # For 1-D tensors, use index_select
                    # Also check if input is 0-dim to avoid "select() cannot be applied to a 0-dim tensor"
                    return f"    {output_var} = ({input_vars[0]}.select({axis}, int({indices_var}.item())) if {indices_var}.ndim == 0 else torch.index_select({input_vars[0]}, {axis}, {indices_var}.long())) if {input_vars[0]}.ndim > 0 else {input_vars[0]}"
                else:
                    # Dynamic intermediate variable
                    if axis == 0:
                        return (
                            f"    {output_var} = {input_vars[0]}[{indices_var}.long()]"
                        )
                    else:
                        return f"    {output_var} = torch.gather({input_vars[0]}, {axis}, {indices_var}.long())"

        elif layer.layer_type == "Cast":
            onnx_to_torch_dtype = {
                1: "torch.float32",
                2: "torch.uint8",
                3: "torch.int8",
                6: "torch.int32",
                7: "torch.int64",
                10: "torch.float16",
                11: "torch.float64",
            }
            to_type = 1
            for attr in layer.onnx_node.attribute:
                if attr.name == "to":
                    to_type = attr.i
            dtype = onnx_to_torch_dtype.get(to_type, "torch.float32")
            return f"    {output_var} = {input_vars[0]}.to({dtype})"

        elif layer.layer_type == "Concat":
            axis = 0
            for attr in layer.onnx_node.attribute:
                if attr.name == "axis":
                    axis = attr.i

            # Pattern from _forward_part.py: Handle mixed parameter/variable inputs
            # Parameters (start with "self.") may need unsqueeze depending on their shape
            all_params = all(v.startswith("self.") for v in input_vars)
            all_vars = all(not v.startswith("self.") for v in input_vars)
            mixed = not all_params and not all_vars

            if mixed:
                # Check if first variable comes from Shape or Slice
                first_var = next(
                    (v for v in input_vars if not v.startswith("self.")), None
                )
                first_var_layer = (
                    var_source_layers.get(first_var)
                    if var_source_layers and first_var
                    else None
                )

                # For Shape/Slice outputs (1D tensors):
                # - Only unsqueeze scalar parameters (shape [])
                # - Don't unsqueeze 1D parameters (shape [1])
                if first_var_layer in ("Shape", "Slice"):
                    # Map input variables to ONNX names to check shapes
                    adjusted_inputs = []
                    for i, var in enumerate(input_vars):
                        if var.startswith("self."):
                            # Get ONNX input name for this parameter
                            if i < len(layer.input_names):
                                onnx_name = layer.input_names[i]
                                # Check if parameter is scalar (0D) by looking at initializer
                                if initializers and onnx_name in initializers:
                                    from onnx import numpy_helper

                                    param_array = numpy_helper.to_array(
                                        initializers[onnx_name]
                                    )
                                    # Only unsqueeze if parameter is truly scalar (0D)
                                    if param_array.ndim == 0:
                                        adjusted_inputs.append(f"{var}.unsqueeze(0)")
                                    else:
                                        # 1D or higher, don't unsqueeze
                                        adjusted_inputs.append(var)
                                else:
                                    # Can't determine shape, don't unsqueeze
                                    adjusted_inputs.append(var)
                            else:
                                adjusted_inputs.append(var)
                        else:
                            adjusted_inputs.append(var)
                    inputs_str = ", ".join(adjusted_inputs)
                else:
                    # For other mixed cases:
                    # Check parameter shapes and only unsqueeze scalars (0-D)
                    # This applies regardless of what operation produced the first variable
                    # For other cases, check parameter shapes
                    # Only unsqueeze scalar (0D) parameters, not 1D parameters
                    adjusted_inputs = []
                    for i, var in enumerate(input_vars):
                        if var.startswith("self."):
                            # Get ONNX input name for this parameter
                            if i < len(layer.input_names):
                                onnx_name = layer.input_names[i]
                                # Check if parameter is scalar (0D) by looking at initializer
                                if initializers and onnx_name in initializers:
                                    from onnx import numpy_helper

                                    param_array = numpy_helper.to_array(
                                        initializers[onnx_name]
                                    )
                                    # Only unsqueeze if parameter is truly scalar (0D)
                                    if param_array.ndim == 0:
                                        adjusted_inputs.append(f"{var}.unsqueeze(0)")
                                    else:
                                        # 1D or higher, don't unsqueeze
                                        adjusted_inputs.append(var)
                                else:
                                    # Can't determine shape, don't unsqueeze
                                    adjusted_inputs.append(var)
                            else:
                                adjusted_inputs.append(var)
                        else:
                            adjusted_inputs.append(var)
                    inputs_str = ", ".join(adjusted_inputs)
            else:
                inputs_str = ", ".join(input_vars)

            return f"    {output_var} = torch.cat([{inputs_str}], dim={axis})"

        elif layer.layer_type == "Gemm":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            alpha = func_args.get("alpha", 1.0)
            beta = func_args.get("beta", 1.0)
            trans_a = func_args.get("transA", 0)
            trans_b = func_args.get("transB", 0)

            a_var = f"{input_vars[0]}.T" if trans_a else input_vars[0]
            b_var = f"{input_vars[1]}.T" if trans_b else input_vars[1]

            matmul_expr = f"{a_var} @ {b_var}"
            if alpha != 1.0:
                matmul_expr = f"{alpha} * {matmul_expr}"

            if len(input_vars) >= 3:
                bias_expr = input_vars[2]
                if beta != 1.0:
                    bias_expr = f"{beta} * {bias_expr}"
                return f"    {output_var} = {matmul_expr} + {bias_expr}"
            else:
                return f"    {output_var} = {matmul_expr}"

        elif layer.layer_type == "Conv":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            stride_list = func_args.get("stride", [1, 1])
            pads = func_args.get("padding", [0, 0, 0, 0])
            dilation_list = func_args.get("dilation", [1, 1])
            groups = func_args.get("groups", 1)

            bias_arg = f", {input_vars[2]}" if len(input_vars) >= 3 else ", None"

            if len(stride_list) == 1:
                stride = stride_list[0]
                padding = pads[0] if pads else 0
                dilation = dilation_list[0] if dilation_list else 1
                return f"    {output_var} = F.conv1d({input_vars[0]}, {input_vars[1]}{bias_arg}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups})"
            else:
                stride = tuple(stride_list)
                padding = tuple(pads[:2])
                dilation = tuple(dilation_list)
                return f"    {output_var} = F.conv2d({input_vars[0]}, {input_vars[1]}{bias_arg}, stride={stride}, padding={padding}, dilation={dilation}, groups={groups})"

        elif layer.layer_type == "ScatterND":
            # ONNX ScatterND: scatter(data, indices, updates) -> output
            # Inputs: data (tensor), indices (tensor of indices), updates (tensor of values)
            # PyTorch implementation using advanced indexing
            if len(input_vars) < 3:
                raise ValueError(f"ScatterND requires 3 inputs, got {len(input_vars)}")

            data_var = input_vars[0]
            indices_var = input_vars[1]
            updates_var = input_vars[2]

            # Generate code to perform scatter_nd operation
            # Clone data to avoid modifying the original
            # Split indices along the last dimension to get individual index tensors
            # Use advanced indexing to scatter updates into the cloned data
            code = f"    {output_var} = {data_var}.clone()\n"
            code += f"    _scatter_indices = {indices_var}.long()\n"
            code += f"    _idx_list = [_scatter_indices[..., i] for i in range(_scatter_indices.shape[-1])]\n"
            code += f"    {output_var}[tuple(_idx_list)] = {updates_var}"
            return code

        elif layer.layer_type == "Upsample":
            # ONNX Upsample/Resize can have dynamic scales or sizes
            # Inputs: [X, (roi), scales/sizes]
            # For Resize: inputs[0]=data, inputs[1]=roi, inputs[2]=scales, inputs[3]=sizes
            # For Upsample: inputs[0]=data, inputs[1]=scales
            from ..type_inference.functional import _extract_onnx_attributes

            attrs = _extract_onnx_attributes(layer.onnx_node)
            mode = attrs.get("mode", "nearest")

            # Map ONNX mode to PyTorch mode
            mode_map = {"nearest": "nearest", "linear": "bilinear", "cubic": "bicubic"}
            torch_mode = mode_map.get(mode, "nearest")

            input_var = input_vars[0]

            # Check if we have scales or sizes
            # Resize node: roi in input[1], scales in input[2], sizes in input[3]
            # Upsample node: scales in input[1]
            if layer.onnx_node.op_type == "Resize" and len(input_vars) >= 3:
                # Resize node with scales in input[2] or sizes in input[3]
                if len(input_vars) >= 4 and input_vars[3]:
                    # Use sizes if available
                    sizes_var = input_vars[3]
                    code = f"    _upsample_size = {sizes_var}.int().tolist()[2:] if {sizes_var}.ndim > 0 and len({sizes_var}) > 2 else {sizes_var}.int().tolist()\n"
                    code += f"    {output_var} = F.interpolate({input_var}, size=_upsample_size, mode='{torch_mode}')"
                    return code
                elif input_vars[2]:
                    # Use scales
                    scales_var = input_vars[2]
                    code = f"    _upsample_scales = {scales_var}.float().tolist()[2:] if {scales_var}.ndim > 0 and len({scales_var}) > 2 else {scales_var}.float().tolist()\n"
                    code += f"    {output_var} = F.interpolate({input_var}, scale_factor=_upsample_scales, mode='{torch_mode}')"
                    return code
            elif len(input_vars) >= 2:
                # Upsample node with scales in input[1]
                scales_var = input_vars[1]
                code = f"    _upsample_scales = {scales_var}.float().tolist()[2:] if {scales_var}.ndim > 0 and len({scales_var}) > 2 else {scales_var}.float().tolist()\n"
                code += f"    {output_var} = F.interpolate({input_var}, scale_factor=_upsample_scales, mode='{torch_mode}')"
                return code

            # Fallback: use default scale_factor of 2
            return f"    {output_var} = F.interpolate({input_var}, scale_factor=2.0, mode='{torch_mode}')"

        elif layer.layer_type == "ConvTranspose":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            stride_list = func_args.get("stride", [1, 1])
            pads = func_args.get("padding", [0, 0, 0, 0])
            output_padding_list = func_args.get("output_padding", [0, 0])
            dilation_list = func_args.get("dilation", [1, 1])
            groups = func_args.get("groups", 1)

            bias_arg = f", {input_vars[2]}" if len(input_vars) >= 3 else ", None"

            if len(stride_list) == 1:
                stride = stride_list[0]
                padding = pads[0] if pads else 0
                output_padding = output_padding_list[0] if output_padding_list else 0
                dilation = dilation_list[0] if dilation_list else 1
                return f"    {output_var} = F.conv_transpose1d({input_vars[0]}, {input_vars[1]}{bias_arg}, stride={stride}, padding={padding}, output_padding={output_padding}, dilation={dilation}, groups={groups})"
            else:
                stride = tuple(stride_list)
                padding = tuple(pads[:2])
                output_padding = tuple(output_padding_list)
                dilation = tuple(dilation_list)
                return f"    {output_var} = F.conv_transpose2d({input_vars[0]}, {input_vars[1]}{bias_arg}, stride={stride}, padding={padding}, output_padding={output_padding}, dilation={dilation}, groups={groups})"

        elif layer.layer_type == "Slice":
            data_var = input_vars[0]

            # Check if starts/ends/axes/steps are inputs (new opset) or attributes (old opset)
            if len(input_vars) > 1:
                # New opset: inputs
                starts_var = input_vars[1]
                ends_var = input_vars[2] if len(input_vars) > 2 else "None"
                axes_var = input_vars[3] if len(input_vars) > 3 else "None"
                steps_var = input_vars[4] if len(input_vars) > 4 else "None"

                if starts_var != "None":
                    starts_var = f"{starts_var}.flatten().int().tolist()"
                if ends_var != "None":
                    ends_var = f"{ends_var}.flatten().int().tolist()"
                if axes_var != "None":
                    axes_var = f"{axes_var}.flatten().int().tolist()"
                if steps_var != "None":
                    steps_var = f"{steps_var}.flatten().int().tolist()"
            else:
                # Old opset: attributes
                func_args = extract_functional_args(
                    layer.onnx_node, initializers or {}, layer.layer_type
                )
                starts_var = str(func_args.get("starts", []))
                ends_var = str(func_args.get("ends", []))
                axes_var = (
                    str(func_args.get("axes", None)) if "axes" in func_args else "None"
                )
                steps_var = "None"  # steps not in old opset

            return f"    starts, ends, axes, steps = {starts_var}, {ends_var}, {axes_var}, {steps_var}\n    slices = [slice(None)] * {data_var}.ndim\n    if axes is None:\n        axes = list(range(len(starts)))\n    if steps is None:\n        steps = [1] * len(starts)\n    for axis, start, end, step in zip(axes, starts, ends, steps):\n        slices[axis] = slice(start, end, step)\n    {output_var} = {data_var}[tuple(slices)]"

        elif layer.layer_type == "Sign":
            return f"    {output_var} = torch.sign({input_vars[0]})"

        elif layer.layer_type == "Split":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            axis = func_args.get("axis", 0)

            split_sizes = None
            for attr in layer.onnx_node.attribute:
                if attr.name == "split":
                    split_sizes = list(attr.ints) if attr.ints else None
                    break

            data_var = input_vars[0]
            split_var = input_vars[1] if len(input_vars) > 1 else None

            if split_var is None and split_sizes is None:
                num_outputs = len(layer.output_names)
                return f"    {output_var} = torch.chunk({data_var}, {num_outputs}, dim={axis})"
            elif split_var is not None:
                if split_var.startswith("self."):
                    split_var = f"{split_var}.int().tolist()"
                else:
                    split_var = f"{split_var}.int().tolist()"
                return f"    {output_var} = torch.split({data_var}, {split_var}, dim={axis})"
            else:
                return f"    {output_var} = torch.split({data_var}, {list(split_sizes)}, dim={axis})"

        elif layer.layer_type == "ConstantOfShape":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            value = func_args.get("value", 0.0)
            shape_var = input_vars[0]

            if shape_var.startswith("self."):
                shape_var = f"tuple({shape_var}.flatten().int().tolist())"
            else:
                shape_var = f"tuple({shape_var}.flatten().int().tolist())"

            return f"    {output_var} = torch.full({shape_var}, {value})"

        elif layer.layer_type == "ReduceMean":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            axes = func_args.get("axes")
            keepdims = func_args.get("keepdims", True)

            if len(input_vars) > 1:
                axes_var = input_vars[1]
                if axes_var.startswith("self."):
                    axes_var = f"tuple({axes_var}.int().tolist())"
                else:
                    axes_var = f"tuple({axes_var}.int().tolist())"
                return f"    {output_var} = torch.mean({input_vars[0]}, dim={axes_var}, keepdim={keepdims})"
            elif axes is not None:
                if isinstance(axes, (list, tuple)):
                    axes_str = str(tuple(axes))
                else:
                    axes_str = str(axes)
                return f"    {output_var} = torch.mean({input_vars[0]}, dim={axes_str}, keepdim={keepdims})"
            else:
                return f"    {output_var} = torch.mean({input_vars[0]}, keepdim={keepdims})"

        elif layer.layer_type == "ReduceSum":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            axes = func_args.get("axes")
            keepdims = func_args.get("keepdims", True)

            if len(input_vars) > 1:
                axes_var = input_vars[1]
                if axes_var.startswith("self."):
                    axes_var = f"tuple({axes_var}.int().tolist())"
                else:
                    axes_var = f"tuple({axes_var}.int().tolist())"
                return f"    {output_var} = torch.sum({input_vars[0]}, dim={axes_var}, keepdim={keepdims})"
            elif axes is not None:
                if isinstance(axes, (list, tuple)):
                    axes_str = str(tuple(axes))
                else:
                    axes_str = str(axes)
                return f"    {output_var} = torch.sum({input_vars[0]}, dim={axes_str}, keepdim={keepdims})"
            else:
                return (
                    f"    {output_var} = torch.sum({input_vars[0]}, keepdim={keepdims})"
                )

        elif layer.layer_type == "ArgMax":
            func_args = extract_functional_args(
                layer.onnx_node, initializers or {}, layer.layer_type
            )
            axis = func_args.get("axis", 0)
            keepdims = func_args.get("keepdims", True)
            return f"    {output_var} = torch.argmax({input_vars[0]}, dim=min({axis}, {input_vars[0]}.ndim - 1) if {input_vars[0]}.ndim > 0 else 0, keepdim={keepdims})"

        elif layer.layer_type == "Min":
            if len(input_vars) >= 2:
                return f"    {output_var} = torch.minimum({input_vars[0]}, {input_vars[1]})"
            else:
                return f"    {output_var} = torch.min({input_vars[0]})"

        elif layer.layer_type == "Max":
            if len(input_vars) >= 2:
                return f"    {output_var} = torch.maximum({input_vars[0]}, {input_vars[1]})"
            else:
                return f"    {output_var} = torch.max({input_vars[0]})"

        elif layer.layer_type == "Cos":
            return f"    {output_var} = torch.cos({input_vars[0]})"

        elif layer.layer_type == "Sin":
            return f"    {output_var} = torch.sin({input_vars[0]})"

        elif layer.layer_type == "Floor":
            return f"    {output_var} = torch.floor({input_vars[0]})"

        elif layer.layer_type == "Neg":
            return f"    {output_var} = torch.neg({input_vars[0]})"

        elif layer.layer_type == "Expand":
            data_var = input_vars[0]
            shape_var = input_vars[1] if len(input_vars) > 1 else "None"

            if shape_var != "None":
                # ONNX Expand uses -1 to mean "keep the existing dimension"
                # Dimensions are aligned from the right (broadcast rules)
                # Note: Sometimes ONNX Expand is used as a broadcasting hint, and actual
                # broadcasting happens in subsequent operations (e.g., Add, Sub, Mul).
                # If expand fails due to incompatible shapes, fall back to returning data as-is
                # and let PyTorch's automatic broadcasting handle it in the next operation.
                code = f"    _expand_shape = {shape_var}.int().tolist() if isinstance({shape_var}, torch.Tensor) else {shape_var}\n"
                code += f"    _data_shape = list({data_var}.shape)\n"
                code += f"    _expand_shape = [s if s != -1 else _data_shape[i - (len(_expand_shape) - len(_data_shape))] if i - (len(_expand_shape) - len(_data_shape)) >= 0 else 1 for i, s in enumerate(_expand_shape)]\n"
                code += f"    try:\n"
                code += (
                    f"        {output_var} = {data_var}.expand(tuple(_expand_shape))\n"
                )
                code += f"    except RuntimeError:\n"
                code += f"        {output_var} = {data_var}  # Fall back to original if expand fails"
                return code
            else:
                return f"    {output_var} = {data_var}"

        elif layer.layer_type == "Range":
            start_var = input_vars[0]
            limit_var = input_vars[1] if len(input_vars) > 1 else "None"
            delta_var = input_vars[2] if len(input_vars) > 2 else "1"

            if start_var.startswith("self."):
                start_var = f"{start_var}.item()"
            if limit_var != "None" and limit_var.startswith("self."):
                limit_var = f"{limit_var}.item()"
            if delta_var != "1" and delta_var.startswith("self."):
                delta_var = f"{delta_var}.item()"

            return f"    {output_var} = torch.arange({start_var}, {limit_var}, {delta_var})"

        elif layer.layer_type == "Where":
            condition_var = input_vars[0]
            x_var = input_vars[1] if len(input_vars) > 1 else "0"
            y_var = input_vars[2] if len(input_vars) > 2 else "0"
            return f"    {output_var} = torch.where({condition_var}, {x_var}, {y_var})"

        elif layer.layer_type == "Clip":
            # ONNX Clip: optional min/max can be attributes (opset < 11) or inputs (opset >= 11)
            # Inputs: [input, min (optional), max (optional)]
            from ..type_inference.functional import _extract_onnx_attributes

            input_var = input_vars[0]
            min_var = None
            max_var = None

            # Helper function to check if a variable name is valid
            def is_valid_var(v):
                if not v or not v.strip():
                    return False
                # Filter out malformed "self." without anything after
                if v == "self." or v.strip() == "self.":
                    return False
                return True

            # Check for min/max in input tensors (opset >= 11)
            if len(input_vars) >= 2 and is_valid_var(input_vars[1]):
                min_var = input_vars[1]
            if len(input_vars) >= 3 and is_valid_var(input_vars[2]):
                max_var = input_vars[2]

            # If not in inputs, try to get from attributes (opset < 11)
            if min_var is None or max_var is None:
                attrs = _extract_onnx_attributes(layer.onnx_node)
                if min_var is None and "min" in attrs and attrs["min"] is not None:
                    min_var = str(attrs["min"])
                if max_var is None and "max" in attrs and attrs["max"] is not None:
                    max_var = str(attrs["max"])

            # Build torch.clamp call
            clamp_args = [input_var]
            if min_var is not None and is_valid_var(min_var):
                if min_var.startswith("self."):
                    clamp_args.append(f"min={min_var}.item()")
                else:
                    clamp_args.append(f"min={min_var}")
            if max_var is not None and is_valid_var(max_var):
                if max_var.startswith("self."):
                    clamp_args.append(f"max={max_var}.item()")
                else:
                    clamp_args.append(f"max={max_var}")

            # If neither min nor max is specified, just pass through
            if len(clamp_args) == 1:
                return f"    {output_var} = {input_var}"

            return f"    {output_var} = torch.clamp({', '.join(clamp_args)})"

        else:
            return f"    {output_var} = {template}({input_vars[0]})"

    else:
        # Handle BatchNorm2d with non-4D input
        if (
            layer.layer_type == "BatchNorm2d"
            and len(input_vars) > 0
            and var_source_layers
        ):
            first_input = input_vars[0]
            source_layer = var_source_layers.get(first_input)
            if source_layer == "Transpose":
                # Add unsqueeze(-1) to convert 3D → 4D for BatchNorm2d, then squeeze(-1) back to 3D
                input_str = f"{first_input}.unsqueeze(-1)"
                return f"    {output_var} = self.{layer.layer_name}({input_str}).squeeze(-1)"
            elif source_layer == "ReduceMean":
                # ReduceMean with keepdim=False produces 2D output (N, C)
                # Add unsqueeze(-1) twice to convert 2D → 4D: (N, C) → (N, C, 1, 1)
                # This keeps C in the correct position for BatchNorm2d
                input_str = f"{first_input}.unsqueeze(-1).unsqueeze(-1)"
                return f"    {output_var} = self.{layer.layer_name}({input_str}).squeeze(-1).squeeze(-1)"

        input_str = ", ".join(input_vars)
        return f"    {output_var} = self.{layer.layer_name}({input_str})"


def generate_forward_method(
    layers: list[LayerIR],
    inputs: list[str],
    outputs: list[str],
    name_mapping: dict[str, str] | None = None,
    initializers: dict | None = None,
) -> str:
    """Generate forward() method with tensor flow.

    Creates forward pass code that:
    - Takes input tensor(s)
    - Passes data through layers in sequence
    - Handles functional operations inline
    - Uses @ operator for matrix multiplication
    - References parameters with simplified names
    - Returns output tensor(s)

    Example output:
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x1 = self.flatten1(x)
            x2 = x1 @ self.weight1
            x3 = x2 + self.bias1
            x4 = self.relu1(x3)
            return x4

    :param layers: List of LayerIR in execution order
    :param inputs: List of input tensor names
    :param outputs: List of output tensor names
    :param name_mapping: Mapping from ONNX names to simplified parameter names
    :param initializers: ONNX initializers (needed for functional args extraction)
    :return: Python code string for forward() method
    """
    lines: list[str] = [
        "def forward(self, x: torch.Tensor) -> torch.Tensor:",
        '    """Forward pass."""',
    ]

    # Build layer parameter mapping: ONNX name -> (layer_name, param_name)
    layer_param_mapping: dict[str, tuple[str, str]] = {}
    for layer in layers:
        if layer.parameters:
            for param_name, onnx_name in layer.parameters.items():
                layer_param_mapping[onnx_name] = (layer.layer_name, param_name)

    tensor_names: dict[str, str] = {}
    var_source_layers: dict[str, str] = {}

    for input_name in inputs:
        tensor_names[input_name] = "x"

    var_counter = 1

    for layer in layers:
        input_vars = get_input_variables(
            layer, tensor_names, name_mapping, layer_param_mapping
        )

        output_var_name = f"x{var_counter}"
        var_counter += 1

        operation_code = generate_operation_code(
            layer,
            output_var_name,
            input_vars,
            name_mapping,
            initializers,
            var_source_layers,
        )
        lines.append(operation_code)

        if len(layer.output_names) > 1:
            output_var_names = []
            for i in range(len(layer.output_names)):
                individual_var = f"x{var_counter}"
                var_counter += 1
                output_var_names.append(individual_var)

            unpack_line = f"    {', '.join(output_var_names)} = {output_var_name}"
            lines.append(unpack_line)

            for output_name, individual_var in zip(
                layer.output_names, output_var_names
            ):
                tensor_names[output_name] = individual_var
                var_source_layers[individual_var] = layer.layer_type
        else:
            for output_name in layer.output_names:
                tensor_names[output_name] = output_var_name
                var_source_layers[output_var_name] = layer.layer_type

    final_output_var = tensor_names[outputs[0]]
    lines.append(f"    return {final_output_var}")

    return "\n".join(lines)


def get_input_variables(
    layer: LayerIR,
    tensor_names: dict[str, str],
    name_mapping: dict[str, str] | None = None,
    layer_param_mapping: dict[str, tuple[str, str]] | None = None,
) -> list[str]:
    """Get input variable names for a layer.

    For parametric layers (Conv2d, Linear, BatchNorm2d), only includes actual data inputs,
    not the parameter inputs (weight, bias, etc.) which are internal to the layer.

    For functional operations (MatMul, Add, etc.), includes all inputs including
    parameter references. Uses hierarchical names for layer parameters (self.fc3.bias)
    and simplified names for standalone parameters (self.param40).

    :param layer: Layer IR
    :param tensor_names: Mapping from tensor names to Python variables
    :param name_mapping: Mapping from ONNX names to simplified parameter names
    :param layer_param_mapping: Mapping from ONNX names to (layer_name, param_name) tuples
    :return: List of input variable names
    """
    input_vars: list[str] = []

    parameter_input_names = (
        set(layer.parameters.values()) if layer.parameters else set()
    )

    for input_name in layer.input_names:
        if is_functional_operation(
            layer.layer_type
        ) or is_functional_operation_with_args(layer.layer_type):
            if input_name in tensor_names:
                input_vars.append(tensor_names[input_name])
            elif layer_param_mapping and input_name in layer_param_mapping:
                # Layer parameter: use hierarchical name
                layer_name, param_name = layer_param_mapping[input_name]
                input_vars.append(f"self.{layer_name}.{param_name}")
            elif name_mapping and input_name in name_mapping:
                # Standalone parameter: use simplified name
                simplified_name = name_mapping[input_name]
                input_vars.append(f"self.{simplified_name}")
            else:
                input_vars.append(f"self.{input_name}")
        else:
            if input_name in parameter_input_names:
                continue
            if layer.layer_type == "Dropout" and len(input_vars) >= 1:
                continue
            if input_name in tensor_names:
                input_vars.append(tensor_names[input_name])
            else:
                input_vars.append(f"self.{input_name}")

    if not input_vars:
        return ["x"]

    return input_vars
