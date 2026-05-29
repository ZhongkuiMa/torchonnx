"""Unit tests for ``torchonnx.presets.has_batch_dim``.

The function used to substring-match the benchmark names against the
full path string, which silently flipped its answer for any user
directory whose name happened to contain a benchmark name as a
substring. R12 switched to a path-component match; these tests pin
the new contract so the regression cannot return.
"""

__docformat__ = "restructuredtext"

import pytest

from torchonnx.presets import BENCHMARKS_WITHOUT_BATCH_DIM, has_batch_dim


class TestHasBatchDim:
    """``has_batch_dim`` matches per path component, not substring."""

    @pytest.mark.parametrize(
        "path",
        [
            "/home/u/models/resnet50.onnx",
            "/data/imagenet/vgg16-7.onnx",
            "models/acasxu_2023/onnx/ACASXU_run2a_1_2_batch_2000.onnx",
        ],
    )
    def test_unknown_paths_return_true(self, path):
        """Paths whose components are not in the no-batch list have a batch dim."""
        assert has_batch_dim(path) is True

    @pytest.mark.parametrize(
        ("benchmark", "path_template"),
        [
            ("cctsdb_yolo", "/data/{}/onnx/patch-1.onnx"),
            ("test_nano.onnx", "/tmp/{}"),
            ("pensieve_big_parallel.onnx", "/data/nn4sys/{}"),
        ],
    )
    def test_known_no_batch_matches_path_component(self, benchmark, path_template):
        """Exact component / basename match flips the answer to False."""
        path = path_template.format(benchmark)
        assert has_batch_dim(path) is False

    def test_user_folder_containing_benchmark_substring_is_not_false_match(self):
        """The pre-R12 substring footgun: ``cctsdb_yolo_experiments`` no longer matches.

        Before R12, ``has_batch_dim("/me/cctsdb_yolo_experiments/foo.onnx")``
        returned False because ``"cctsdb_yolo" in onnx_path`` was True. The
        user folder name is not a benchmark path component, so the answer
        must be True.
        """
        path = "/home/me/cctsdb_yolo_experiments/foo.onnx"
        assert has_batch_dim(path) is True

    def test_benchmark_subdir_below_known_benchmark_still_matches(self):
        """A genuine cctsdb_yolo/<file>.onnx still gets recognised."""
        path = "/data/cctsdb_yolo/onnx/patch-1.onnx"
        assert has_batch_dim(path) is False

    def test_benchmarks_constant_is_a_frozenset_or_tuple(self):
        """The constant must be immutable so callers can not mutate it."""
        assert isinstance(BENCHMARKS_WITHOUT_BATCH_DIM, (tuple, frozenset))
