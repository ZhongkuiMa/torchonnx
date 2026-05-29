"""Vmap-compatible runtime helpers.

Inlined by the code generator into modules emitted with ``vmap_mode=True``
(the default). Uses functional ``torch.scatter`` / ``torch.gather`` so the
helpers compose with ``torch.vmap`` and ``torch.compile``.
``dynamic_slice`` returns ``(result, valid_flag)`` so a downstream
``scatter_nd`` can suppress itself without Python branching.
"""

__docformat__ = "restructuredtext"
__all__ = ["dynamic_expand", "dynamic_slice", "scatter_nd"]

import torch


def dynamic_slice(data, starts, ends, axes=None, steps=None, slice_lengths=None):
    """Vmap-compatible dynamic slice helper for ONNX Slice operation.

    Returns a tuple (result, valid_flag) where:
    - result: The sliced data (zeros if slice was empty/out-of-bounds)
    - valid_flag: 1.0 if slice was non-empty, 0.0 if empty

    The valid_flag should be accumulated across multiple slices and passed
    to scatter_nd to determine whether to actually perform the scatter.

    For vmap compatibility:
    - axes/steps MUST be constant (Python ints or lists)
    - starts/ends can be tensors (input-dependent)
    - slice_lengths MUST be provided (list of ints, one per axis)

    Args:
        data: Input tensor to slice
        starts: Start indices (tensor or list)
        ends: End indices (tensor or list)
        axes: Axes to slice along (constant list or None)
        steps: Step sizes (constant list or None, must be 1 for now)
        slice_lengths: Static slice lengths for each axis (REQUIRED for vmap mode)
    """
    # Convert to tensors if needed; force int64 to avoid dtype mismatch with
    # dim_size arithmetic below (older ONNX exporters emit int32 slice params).
    starts = torch.as_tensor(starts, device=data.device).to(torch.long)
    ends = torch.as_tensor(ends, device=data.device).to(torch.long)

    # Handle axes - MUST be constant for vmap compatibility
    if axes is None:
        axes_list = list(range(starts.numel()))
    elif isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    elif isinstance(axes, torch.Tensor):
        axes_list = axes.tolist()
        if not isinstance(axes_list, list):
            axes_list = [axes_list]
    else:
        axes_list = [int(axes)]

    # Handle steps - MUST be constant for vmap compatibility
    if steps is None:
        steps_list = [1] * len(axes_list)
    elif isinstance(steps, (list, tuple)):
        steps_list = list(steps)
    elif isinstance(steps, torch.Tensor):
        steps_list = steps.tolist()
        if not isinstance(steps_list, list):
            steps_list = [steps_list]
    else:
        steps_list = [int(steps)]

    # Handle slice_lengths - default to 1 if not provided
    if slice_lengths is None:
        lengths_list = [1] * len(axes_list)
    else:
        lengths_list = list(slice_lengths)

    result = data
    # Track validity: 1.0 if all slices non-empty, 0.0 if any slice empty
    cumulative_valid = torch.ones((), dtype=data.dtype, device=data.device)

    for i, (axis, step, slice_len) in enumerate(
        zip(axes_list, steps_list, lengths_list, strict=False)
    ):
        axis = int(axis)
        step = int(step)
        slice_len = int(slice_len)
        dim_size = result.shape[axis]

        # Get start for this axis (handle scalar or 1D tensor)
        if starts.numel() == 1:
            start = starts.reshape(())
        else:
            start = starts[i]

        # Normalize negative start index
        start = torch.where(start < 0, dim_size + start, start)

        # Clamp start to valid range
        start = torch.clamp(start.long(), 0, dim_size)

        # Check if slice would be out of bounds (start + slice_len > dim_size)
        # is_valid = 1.0 if in bounds, 0.0 if out of bounds
        is_valid = (start + slice_len <= dim_size).to(data.dtype)
        cumulative_valid = cumulative_valid * is_valid

        # Generate indices for gather: start, start+step, start+2*step, ...
        # These are relative offsets that we add to start
        offsets = torch.arange(slice_len, device=data.device, dtype=torch.long) * step

        # Compute actual indices: start + offsets
        # Clamp to valid range (even if out of bounds, we need valid indices for gather)
        indices = (start + offsets).clamp(0, dim_size - 1)

        # Reshape indices for gather: [slice_len] -> proper broadcast shape
        # indices needs shape where axis dimension is slice_len, others are 1
        shape = [1] * result.ndim
        shape[axis] = slice_len
        indices = indices.view(*shape)

        # Expand indices to match result shape except for the slice axis
        expand_shape = list(result.shape)
        expand_shape[axis] = slice_len
        indices = indices.expand(*expand_shape)

        # Gather along axis
        result = torch.gather(result, axis, indices)

    # Return both the result and validity flag
    # Result is multiplied by validity so out-of-bounds slices return zeros
    return result * cumulative_valid, cumulative_valid


def scatter_nd(data, indices, updates, reduction="none", valid=None):
    """Vmap-compatible PyTorch equivalent of ONNX ScatterND.

    Uses functional torch.scatter instead of in-place index_put_ so the helper
    composes with vmap and functorch transforms.

    Shapes (ONNX semantics):
        data:    (D0, D1, ..., D_{n-1})
        indices: (*outer, K) with K <= n
        updates: (*outer, D_K, D_{K+1}, ..., D_{n-1})  -- one slice per outer position

    The previous implementation computed strides only over the first K
    dimensions and then flattened ``updates`` directly, which silently
    scattered into the wrong cells whenever K < data.ndim (the documented
    ScatterND case). We compute strides over the full data.shape and expand
    each per-outer base offset across the tail "slice size" so every scalar
    in ``updates`` lands in the right linear position.

    Args:
        data: Target tensor to scatter into
        indices: (..., K) where K is the number of leading dimensions to index
        updates: Values to scatter (see shape rule above)
        reduction: "none" or "add"
        valid: Optional validity flag (scalar tensor). If provided and < 0.5,
               returns data unchanged (simulates empty scatter from empty slices).
    """
    indices = indices.to(torch.long)
    updates = updates.to(data.dtype)

    data_shape = data.shape
    n = data.ndim
    k = indices.shape[-1]

    # Stride for dim i over the FULL shape: prod(data.shape[i+1:])
    full_strides = [1] * n
    for i in range(n - 2, -1, -1):
        full_strides[i] = full_strides[i + 1] * int(data_shape[i + 1])
    leading_strides = torch.tensor(full_strides[:k], device=data.device, dtype=torch.long)

    # Slice size = product of trailing dimensions not indexed by indices.
    slice_size = 1
    for i in range(k, n):
        slice_size *= int(data_shape[i])

    # Base linear offset to the start of each slice, one per outer position.
    base_linear = (indices * leading_strides).sum(dim=-1)

    # Expand base_linear over the trailing slice_size dimension and add the
    # tail offset so we cover every scalar of the slice.
    tail_offsets = torch.arange(slice_size, device=data.device, dtype=torch.long)
    full_linear = base_linear.unsqueeze(-1) + tail_offsets

    flat_data = data.reshape(-1)
    flat_updates = updates.reshape(-1)
    flat_linear = full_linear.reshape(-1)

    if reduction == "none":
        scattered = flat_data.scatter(0, flat_linear, flat_updates)
    elif reduction == "add":
        scattered = flat_data.scatter_add(0, flat_linear, flat_updates)
    else:
        raise NotImplementedError(f"Unsupported reduction: {reduction}")

    result = scattered.reshape(data_shape)

    # torch.where keeps the helper vmap-safe (no Python branching).
    if valid is not None:
        result = torch.where(valid > 0.5, result, data)

    return result


def dynamic_expand(data, target_shape):
    """Vmap-compatible dynamic expand helper for ONNX Expand operation.

    This version handles the conversion from ONNX semantics to PyTorch
    while remaining compatible with vmap.

    Note: For full vmap compatibility, target_shape should be constant
    (known at code generation time). Dynamic target_shape from tensors
    may still work but with limitations.
    """
    # Convert target_shape to list of ints
    if isinstance(target_shape, torch.Tensor):
        target_shape = target_shape.to(torch.int64).tolist()
    target_shape = [int(x) for x in target_shape]

    # If data has more dimensions than target, squeeze leading dimensions
    if data.ndim > len(target_shape):
        new_shape = tuple(int(s) for s in data.shape[data.ndim - len(target_shape) :])
        data = data.reshape(new_shape)

    # Convert ONNX semantics to PyTorch
    # ONNX: 1 means "keep dimension", PyTorch: -1 means "keep dimension"
    converted_shape = []
    offset = len(target_shape) - data.ndim

    for i in range(len(target_shape)):
        if i < offset:
            # Dimension doesn't exist in data, use target value
            converted_shape.append(target_shape[i])
        else:
            # Dimension exists in data
            data_idx = i - offset
            if target_shape[i] == 1 and data.shape[data_idx] != 1:
                # Keep data's dimension
                converted_shape.append(-1)
            else:
                # Use target dimension
                converted_shape.append(target_shape[i])

    return data.expand(*converted_shape)
