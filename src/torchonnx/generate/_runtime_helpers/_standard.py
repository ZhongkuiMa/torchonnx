"""Standard (non-vmap) runtime helpers.

Inlined by the code generator into modules emitted with ``vmap_mode=False``.
These are the simplest correct implementations; they may use ``.item()``
or ``index_put_`` (legal in eager mode, forbidden under ``torch.vmap``).
"""

__docformat__ = "restructuredtext"
__all__ = ["dynamic_expand", "dynamic_slice", "scatter_nd"]

import torch


def dynamic_slice(data, starts, ends, axes=None, steps=None):
    """Dynamic slice helper for ONNX Slice operation."""
    # Force int64 (torch.long). ONNX exporters emit a mix of int32 / int64;
    # downstream torch.clamp and torch.arange below require dtype parity with
    # dim_sizes (int64), so mismatched int32 starts/ends would raise at runtime.
    starts = torch.as_tensor(starts, device=data.device).to(torch.long)
    ends = torch.as_tensor(ends, device=data.device).to(torch.long)
    if axes is None:
        axes = torch.arange(starts.numel(), device=data.device)
    else:
        axes = torch.as_tensor(axes, device=data.device).to(torch.long)
    if steps is None:
        steps = torch.ones_like(starts, device=starts.device)
    else:
        steps = torch.as_tensor(steps, device=data.device).to(torch.long)

    # Normalize negative starts/ends
    dims = torch.as_tensor(data.shape, device=data.device)
    # axes tells where to read dim size
    dim_sizes = dims[axes]

    starts = torch.where(starts < 0, dim_sizes + starts, starts)
    ends = torch.where(ends < 0, dim_sizes + ends, ends)

    # Clip to bounds (ONNX semantics)
    # Use tensors for both min and max to avoid type mismatch
    zero = torch.zeros_like(dim_sizes, device=dim_sizes.device)
    starts = torch.clamp(starts, min=zero, max=dim_sizes)
    ends = torch.clamp(ends, min=zero, max=dim_sizes)

    # Build index tuple dynamically
    index = [slice(None)] * data.ndim
    for i in range(axes.shape[0]):
        ax = axes[i].item()
        idx = torch.arange(starts[i], ends[i], steps[i], device=data.device)
        index[ax] = idx

    return data[tuple(index)]


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

    return result


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
        new_shape = tuple(int(s) for s in data.shape[data.ndim - len(target_shape) :])
        data = data.reshape(new_shape)

    # Convert ONNX semantics to PyTorch
    # For each dimension in target_shape:
    # - If target[i] == 1 and data has a non-1 dim at that position,
    #   keep data's dim (-1)
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

    return data.expand(converted_shape)
