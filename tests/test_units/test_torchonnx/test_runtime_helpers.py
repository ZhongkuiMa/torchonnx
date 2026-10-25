"""Unit tests for runtime helpers inlined into generated modules."""

__docformat__ = "restructuredtext"

import pytest

torch = pytest.importorskip("torch")

from torchonnx.generate._runtime_helpers import _standard  # noqa: E402


def test_quantize_linear_preserves_rounding_saturation_and_dtype() -> None:
    """QuantizeLinear uses ties-to-even, saturation, and the zero-point dtype."""
    data = torch.tensor([-200.0, -1.5, -0.5, 0.5, 1.5, 200.0])
    scale = torch.tensor(0.5)
    zero_point = torch.tensor(128, dtype=torch.uint8)

    result = _standard.quantize_linear(data, scale, zero_point)

    assert result.dtype == torch.uint8
    assert torch.equal(result, torch.tensor([0, 125, 127, 129, 131, 255], dtype=torch.uint8))


def test_quantize_linear_defaults_to_uint8() -> None:
    """The omitted zero-point form has the ONNX default uint8 output type."""
    result = _standard.quantize_linear(torch.tensor([-1.0, 1.0]), torch.tensor(0.5))
    assert result.dtype == torch.uint8
    assert torch.equal(result, torch.tensor([0, 2], dtype=torch.uint8))


def test_dequantize_linear_broadcasts_per_axis_parameters() -> None:
    """Per-axis parameters follow the ONNX axis coordinate."""
    data = torch.tensor([[12, 25, 0], [8, 15, 34]], dtype=torch.uint8)
    scale = torch.tensor([0.1, 0.2, 0.5])
    zero_point = torch.tensor([10, 20, 30], dtype=torch.uint8)

    result = _standard.dequantize_linear(data, scale, zero_point, axis=1)

    expected = torch.tensor([[0.2, 1.0, -15.0], [-0.2, -1.0, 2.0]])
    assert torch.allclose(result, expected)


def test_linear_quantization_rejects_mismatched_per_axis_parameters() -> None:
    """Invalid parameter shapes fail closed instead of broadcasting by accident."""
    with pytest.raises(ValueError, match="does not match"):
        _standard.quantize_linear(
            torch.zeros(2, 3),
            torch.ones(2),
            torch.zeros(2, dtype=torch.uint8),
            axis=1,
        )


def test_bias_gelu_matches_native_torch() -> None:
    """BiasGelu is the native exact GELU of the broadcast sum."""
    data = torch.tensor([[1.0, -2.0, 0.5]])
    bias = torch.tensor([0.2, -0.1, 0.4])
    assert torch.equal(
        _standard.bias_gelu(data, bias),
        torch.nn.functional.gelu(data + bias),
    )


def test_layer_normalization_matches_native_torch() -> None:
    """Standard LayerNormalization follows the normalized suffix dimensions."""
    data = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
    weight = torch.tensor([1.0, 1.5, 0.5])
    bias = torch.tensor([0.1, -0.2, 0.3])

    result = _standard.layer_normalization(data, weight, bias, epsilon=1e-5)
    expected = torch.nn.functional.layer_norm(data, (3,), weight, bias, 1e-5)

    assert torch.allclose(result, expected)
    result.sum().backward()
    assert data.grad is not None
    assert torch.isfinite(data.grad).all()


def test_simplified_layer_normalization_uses_rms_without_centering() -> None:
    """The simplified fusion implements RMS normalization, not standard LN."""
    data = torch.tensor([[1.0, -2.0, 0.5]])
    weight = torch.tensor([1.0, 1.5, 0.5])
    expected = data * torch.rsqrt(data.square().mean(dim=-1, keepdim=True) + 1e-5) * weight
    result = _standard.layer_normalization(data, weight, simplified=True)
    assert torch.allclose(result, expected)
