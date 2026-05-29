"""Preset configurations for common benchmarks and models.

Provides benchmark-specific settings for model conversion and analysis,
including batch dimension detection and other benchmark properties.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "BENCHMARKS_WITHOUT_BATCH_DIM",
    "has_batch_dim",
]

from pathlib import Path

# Benchmarks and models without batch dimensions. Each entry is either a
# benchmark *directory name* (``"cctsdb_yolo"``) or a model *filename*
# (``"pensieve_big_parallel.onnx"``). ``has_batch_dim`` matches each entry
# against full path components, not arbitrary substrings, so a user folder
# named ``cctsdb_yolo_experiments`` no longer silently flips the answer.
BENCHMARKS_WITHOUT_BATCH_DIM = (
    "cctsdb_yolo",
    "pensieve_big_parallel.onnx",
    "pensieve_mid_parallel.onnx",
    "pensieve_small_parallel.onnx",
    "test_nano.onnx",
    "test_small.onnx",
    "test_tiny.onnx",
)


def has_batch_dim(onnx_path: str) -> bool:
    """Determine if a model has a batch dimension.

    Matches each ``BENCHMARKS_WITHOUT_BATCH_DIM`` entry against the path's
    components (``Path.parts``) and against the basename, NOT against an
    arbitrary substring of the full path. The earlier substring check
    silently flipped the answer for any user directory whose name happened
    to contain a benchmark name as a substring (for example
    ``/home/me/cctsdb_yolo_experiments/foo.onnx``), so the function would
    declare the model un-batched even though the actual benchmark folder
    was nowhere on the path.

    :param onnx_path: Path to ONNX model file.

    :return: True if the model has a batch dimension, False otherwise.
    """
    path = Path(onnx_path)
    components = set(path.parts)
    components.add(path.name)
    return not any(bname in components for bname in BENCHMARKS_WITHOUT_BATCH_DIM)
