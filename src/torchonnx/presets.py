"""Preset configurations for common benchmarks and models.

Provides benchmark-specific settings for model conversion and analysis,
including batch dimension detection and other benchmark properties.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "BENCHMARKS_WITHOUT_BATCH_DIM",
    "if_has_batch_dim",
]


# Benchmarks and models without batch dimensions
BENCHMARKS_WITHOUT_BATCH_DIM = (
    "cctsdb_yolo",
    "pensieve_big_parallel.onnx",
    "pensieve_mid_parallel.onnx",
    "pensieve_small_parallel.onnx",
    "test_nano.onnx",
    "test_small.onnx",
    "test_tiny.onnx",
)


def if_has_batch_dim(onnx_path: str) -> bool:
    """Determine if model has batch dimension by checking full path.

    Checks both benchmark name and model filename in the path
    against known models without batch dimensions.

    :param onnx_path: Path to ONNX model file
    :return: True if model has batch dimension, False otherwise
    """
    return all(bname not in onnx_path for bname in BENCHMARKS_WITHOUT_BATCH_DIM)
