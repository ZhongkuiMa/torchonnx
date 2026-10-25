"""Standard (non-vmap) runtime helpers.

Inlined by the code generator into modules emitted with ``vmap_mode=False``.
These are the simplest correct implementations; they may use ``.item()``
or ``index_put_`` (legal in eager mode, forbidden under ``torch.vmap``).
"""

__docformat__ = "restructuredtext"
__all__ = [
    "bias_gelu",
    "dequantize_linear",
    "dynamic_expand",
    "dynamic_slice",
    "layer_normalization",
    "quantize_linear",
    "scatter_nd",
]

import torch


def bias_gelu(data, bias):
    """Apply the Microsoft BiasGelu fusion with exact GELU semantics."""
    return torch.nn.functional.gelu(data + bias)


def layer_normalization(data, weight, bias=None, axis=-1, epsilon=1e-5, simplified=False):
    """Apply LayerNormalization or its RMS-only simplified form."""
    normalized_axis = axis if axis >= 0 else axis + data.ndim
    if normalized_axis < 0 or normalized_axis >= data.ndim:
        raise ValueError(f"LayerNormalization axis {axis} is invalid for rank {data.ndim}")
    normalized_shape = tuple(data.shape[normalized_axis:])
    if tuple(weight.shape) != normalized_shape:
        raise ValueError(
            "LayerNormalization weight shape must match the normalized data dimensions"
        )
    if bias is not None and tuple(bias.shape) != normalized_shape:
        raise ValueError("LayerNormalization bias shape must match the normalized data dimensions")

    dimensions = tuple(range(normalized_axis, data.ndim))
    centered = data if simplified else data - data.mean(dim=dimensions, keepdim=True)
    variance = (centered * centered).mean(dim=dimensions, keepdim=True)
    result = centered * torch.rsqrt(variance + epsilon) * weight
    return result if bias is None else result + bias


def quantize_linear(data, scale, zero_point=None, axis=1):
    """Apply ONNX QuantizeLinear with scalar or per-axis parameters."""

    def reshape_parameter(parameter):
        if parameter.numel() == 1:
            return parameter.reshape(())
        if parameter.ndim != 1:
            raise ValueError("QuantizeLinear scale and zero_point must be scalar or 1-D")
        normalized_axis = axis if axis >= 0 else axis + data.ndim
        if normalized_axis < 0 or normalized_axis >= data.ndim:
            raise ValueError(f"QuantizeLinear axis {axis} is invalid for rank {data.ndim}")
        if parameter.shape[0] != data.shape[normalized_axis]:
            raise ValueError(
                "QuantizeLinear per-axis parameter length does not match the data dimension"
            )
        shape = [1] * data.ndim
        shape[normalized_axis] = parameter.shape[0]
        return parameter.reshape(shape)

    scale = reshape_parameter(scale).to(device=data.device)
    if zero_point is None:
        output_dtype = torch.uint8
        zero = torch.zeros((), device=data.device, dtype=scale.dtype)
    else:
        if zero_point.dtype.is_floating_point:
            raise ValueError("QuantizeLinear zero_point must have an integer dtype")
        output_dtype = zero_point.dtype
        zero = reshape_parameter(zero_point).to(device=data.device, dtype=scale.dtype)
    qinfo = torch.iinfo(output_dtype)
    quantized = torch.clamp(torch.round(data / scale) + zero, min=qinfo.min, max=qinfo.max)
    return quantized.to(dtype=output_dtype)


def dequantize_linear(data, scale, zero_point=None, axis=1):
    """Apply ONNX DequantizeLinear with scalar or per-axis parameters."""

    def reshape_parameter(parameter):
        if parameter.numel() == 1:
            return parameter.reshape(())
        if parameter.ndim != 1:
            raise ValueError("DequantizeLinear scale and zero_point must be scalar or 1-D")
        normalized_axis = axis if axis >= 0 else axis + data.ndim
        if normalized_axis < 0 or normalized_axis >= data.ndim:
            raise ValueError(f"DequantizeLinear axis {axis} is invalid for rank {data.ndim}")
        if parameter.shape[0] != data.shape[normalized_axis]:
            raise ValueError(
                "DequantizeLinear per-axis parameter length does not match the data dimension"
            )
        shape = [1] * data.ndim
        shape[normalized_axis] = parameter.shape[0]
        return parameter.reshape(shape)

    scale = reshape_parameter(scale).to(device=data.device)
    data = data.to(dtype=scale.dtype)
    if zero_point is None:
        zero = torch.zeros((), device=data.device, dtype=scale.dtype)
    else:
        zero = reshape_parameter(zero_point).to(device=data.device, dtype=scale.dtype)
    return (data - zero) * scale


def dynamic_slice(data, starts, ends, axes=None, steps=None):
    """Dynamic slice helper for ONNX Slice operation."""
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

    dims = torch.as_tensor(data.shape, device=data.device)
    dim_sizes = dims[axes]
    starts = torch.where(starts < 0, dim_sizes + starts, starts)
    ends = torch.where(ends < 0, dim_sizes + ends, ends)
    zero = torch.zeros_like(dim_sizes, device=dim_sizes.device)
    starts = torch.clamp(starts, min=zero, max=dim_sizes)
    ends = torch.clamp(ends, min=zero, max=dim_sizes)

    index = [slice(None)] * data.ndim
    for i in range(axes.shape[0]):
        ax = axes[i].item()
        idx = torch.arange(starts[i], ends[i], steps[i], device=data.device)
        index[ax] = idx

    return data[tuple(index)]


def scatter_nd(data, indices, updates, reduction="none"):
    """Apply ONNX ScatterND using tensor-based indexing."""
    result = data.clone()
    indices = indices.to(torch.long)
    dim_sizes = torch.as_tensor(data.shape[: indices.shape[-1]], device=data.device)
    indices = torch.where(indices < 0, indices + dim_sizes, indices)
    updates = updates.to(result.dtype)
    idx_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))

    if reduction == "none":
        result.index_put_(idx_tuple, updates)
    elif reduction == "add":
        result.index_put_(idx_tuple, updates, accumulate=True)
    else:
        raise NotImplementedError(f"Unsupported reduction: {reduction}")

    return result


def dynamic_expand(data, target_shape):
    """Apply ONNX Expand while accepting a runtime target shape."""
    if isinstance(target_shape, torch.Tensor):
        target_shape = target_shape.to(torch.int64).tolist()
    target_shape = [int(x) for x in target_shape]

    if data.ndim > len(target_shape):
        new_shape = tuple(int(s) for s in data.shape[data.ndim - len(target_shape) :])
        data = data.reshape(new_shape)

    converted_shape = []
    for i in range(len(target_shape)):
        if i < len(target_shape) - data.ndim:
            converted_shape.append(int(target_shape[i]))
        else:
            data_idx = i - (len(target_shape) - data.ndim)
            if target_shape[i] == 1 and data.shape[data_idx] != 1:
                converted_shape.append(-1)
            else:
                converted_shape.append(int(target_shape[i]))

    return data.expand(converted_shape)
