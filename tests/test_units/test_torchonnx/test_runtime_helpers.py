"""Unit tests for the inlined runtime helpers.

These helpers are extracted via inspect.getsource() and pasted into every
generated PyTorch module. Testing them as real functions catches behavior
regressions at the helper level instead of through the much slower
ONNX-to-module-to-onnxruntime round trip.
"""

__docformat__ = "restructuredtext"

import pytest

torch = pytest.importorskip("torch")

from torchonnx.generate._runtime_helpers import _standard, _vmap  # noqa: E402


class TestStandardDynamicSlice:
    """``_standard.dynamic_slice`` -- the simple eager-mode variant."""

    def test_basic_slice_along_one_axis(self):
        data = torch.arange(12).reshape(3, 4)
        result = _standard.dynamic_slice(data, starts=[1], ends=[3], axes=[0])
        expected = data[1:3, :]
        assert torch.equal(result, expected)

    def test_int32_starts_ends_promote_to_long(self):
        data = torch.arange(8).reshape(2, 4)
        starts = torch.tensor([0], dtype=torch.int32)
        ends = torch.tensor([3], dtype=torch.int32)
        result = _standard.dynamic_slice(data, starts=starts, ends=ends, axes=[1])
        expected = data[:, 0:3]
        assert torch.equal(result, expected)

    def test_negative_starts_normalize(self):
        data = torch.arange(10)
        result = _standard.dynamic_slice(data, starts=[-3], ends=[10], axes=[0])
        expected = data[-3:]
        assert torch.equal(result, expected)


class TestStandardScatterND:
    """``_standard.scatter_nd`` -- correctness under K < ndim."""

    def test_k_equal_ndim(self):
        data = torch.zeros(4, 4)
        indices = torch.tensor([[0, 1], [2, 3]])
        updates = torch.tensor([10.0, 20.0])
        result = _standard.scatter_nd(data, indices, updates)
        assert result[0, 1].item() == 10.0
        assert result[2, 3].item() == 20.0

    def test_k_less_than_ndim_writes_full_slice(self):
        data = torch.zeros(3, 4)
        indices = torch.tensor([[1]])
        updates = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = _standard.scatter_nd(data, indices, updates)
        assert torch.equal(result[1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert result[0].sum().item() == 0.0
        assert result[2].sum().item() == 0.0

    def test_reduction_add(self):
        data = torch.zeros(3)
        indices = torch.tensor([[0], [0], [1]])
        updates = torch.tensor([1.0, 2.0, 5.0])
        result = _standard.scatter_nd(data, indices, updates, reduction="add")
        assert result[0].item() == 3.0
        assert result[1].item() == 5.0

    def test_unsupported_reduction_raises(self):
        data = torch.zeros(3)
        indices = torch.tensor([[0]])
        updates = torch.tensor([1.0])
        with pytest.raises(NotImplementedError, match="Unsupported reduction"):
            _standard.scatter_nd(data, indices, updates, reduction="mul")


class TestStandardDynamicExpand:
    """``_standard.dynamic_expand`` -- ONNX semantics conversion."""

    def test_expand_with_keep_dim_signal(self):
        data = torch.tensor([[1.0], [2.0], [3.0]])
        result = _standard.dynamic_expand(data, target_shape=[3, 4])
        assert result.shape == (3, 4)
        assert torch.equal(result[:, 0], torch.tensor([1.0, 2.0, 3.0]))

    def test_higher_dim_data_gets_squeezed_leading(self):
        data = torch.zeros(1, 1, 3, 4)
        result = _standard.dynamic_expand(data, target_shape=[3, 4])
        assert result.shape == (3, 4)


class TestVmapDynamicSlice:
    """``_vmap.dynamic_slice`` -- returns (result, valid_flag)."""

    def test_returns_result_and_valid_flag(self):
        data = torch.arange(20.0).reshape(4, 5)
        result, valid = _vmap.dynamic_slice(data, starts=[1], ends=[3], axes=[0], slice_lengths=[2])
        assert result.shape == (2, 5)
        assert valid.item() == 1.0

    def test_out_of_bounds_marks_invalid_and_zeros_result(self):
        data = torch.arange(6.0).reshape(2, 3)
        result, valid = _vmap.dynamic_slice(data, starts=[5], ends=[8], axes=[0], slice_lengths=[3])
        assert valid.item() == 0.0
        assert torch.equal(result, torch.zeros_like(result))


class TestVmapScatterND:
    """``_vmap.scatter_nd`` -- full-stride correctness and valid suppression."""

    def test_k_less_than_ndim_full_stride(self):
        data = torch.zeros(3, 4)
        indices = torch.tensor([[1]])
        updates = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = _vmap.scatter_nd(data, indices, updates)
        assert torch.equal(result[1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
        assert result[0].sum().item() == 0.0

    def test_valid_zero_returns_data_unchanged(self):
        data = torch.ones(2, 3)
        indices = torch.tensor([[0]])
        updates = torch.tensor([[9.0, 9.0, 9.0]])
        valid = torch.zeros(())
        result = _vmap.scatter_nd(data, indices, updates, valid=valid)
        assert torch.equal(result, data)

    def test_valid_one_applies_scatter(self):
        data = torch.zeros(2, 3)
        indices = torch.tensor([[1]])
        updates = torch.tensor([[7.0, 7.0, 7.0]])
        valid = torch.ones(())
        result = _vmap.scatter_nd(data, indices, updates, valid=valid)
        assert torch.equal(result[1], torch.tensor([7.0, 7.0, 7.0]))


class TestVmapDynamicExpand:
    """``_vmap.dynamic_expand`` -- behaves like the standard variant."""

    def test_basic_expand(self):
        data = torch.tensor([[1.0], [2.0]])
        result = _vmap.dynamic_expand(data, target_shape=[2, 5])
        assert result.shape == (2, 5)
        assert torch.equal(result[:, 0], torch.tensor([1.0, 2.0]))
