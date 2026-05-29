"""High-level ONNX-to-PyTorch conversion pipeline."""

__docformat__ = "restructuredtext"
__all__ = ["TorchONNX"]

import logging
import time
from pathlib import Path

import onnx
import torch

from torchonnx.analyze import build_semantic_ir
from torchonnx.build import build_model_ir
from torchonnx.generate import generate_pytorch_module, to_camel_case
from torchonnx.normalize import load_and_preprocess_onnx_model
from torchonnx.simplify import add_file_header, format_code, optimize_generated_code

_logger = logging.getLogger(__name__)


def _enable_verbose() -> None:
    """Configure package-level logger for console output."""
    pkg_logger = logging.getLogger("torchonnx")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG)


class TorchONNX:
    """ONNX-to-PyTorch converter using a 5-stage pipeline.

    Converts ONNX models to standalone PyTorch nn.Module code by:
    normalize -> build IR -> analyze semantics -> generate code -> simplify + export.

    :param verbose: Print stage-by-stage progress.
    :param use_shapeonnx: Use shapeonnx for shape inference instead of onnxruntime.
    """

    def __init__(self, verbose: bool = False, use_shapeonnx: bool = False):
        self.verbose = verbose
        if verbose:
            _enable_verbose()
        self.use_shapeonnx = use_shapeonnx

    def convert(
        self,
        onnx_path: str,
        benchmark_name: str | None = None,
        target_py_path: str | None = None,
        target_pth_path: str | None = None,
        vmap_mode: bool = True,
    ) -> None:
        """Convert ONNX model to PyTorch module.

        :param onnx_path: Path to input ONNX model.
        :param benchmark_name: Name of benchmark (for module naming).
        :param target_py_path: Path to save generated Python module.
        :param target_pth_path: Path to save state dict.
        :param vmap_mode: If True (default), generate vmap-compatible helper
            functions that avoid ``.item()`` calls and in-place operations,
            enabling compatibility with ``torch.vmap`` and functorch transforms.
        """
        _logger.info(f"TorchONNX: converting {onnx_path}")
        t_total = time.perf_counter()

        # Stage 1: Normalize ONNX model
        t = time.perf_counter()
        model = load_and_preprocess_onnx_model(
            onnx_path,
            target_opset=20,
            infer_shapes=True,
            check_model=True,
            use_shapeonnx=self.use_shapeonnx,
        )
        _logger.info(f"  Normalize: opset 20, shape inference ({time.perf_counter() - t:.4f}s)")

        # Stage 2: Build structural IR
        t = time.perf_counter()
        raw_ir = build_model_ir(model)
        _logger.info(f"  Build IR ({time.perf_counter() - t:.4f}s)")

        # Stage 3: Build semantic IR
        t = time.perf_counter()
        semantic_ir = build_semantic_ir(raw_ir)
        _logger.info(f"  Analyze semantics ({time.perf_counter() - t:.4f}s)")

        # Stage 4: Generate PyTorch code
        t = time.perf_counter()
        model_name = Path(onnx_path).stem
        if benchmark_name:
            module_class_name = f"{benchmark_name}_{model_name}"
        else:
            module_class_name = model_name

        # Convert to CamelCase
        camel_class_name = to_camel_case(module_class_name)

        code, state_dict = generate_pytorch_module(
            semantic_ir, camel_class_name, vmap_mode=vmap_mode
        )
        _logger.info(f"  Generate code: {camel_class_name} ({time.perf_counter() - t:.4f}s)")

        # Stage 5: Simplify generated code and filter state_dict.
        # The overloaded signature of optimize_generated_code lets the type
        # checker see the tuple return when state_dict is non-None, so the
        # destructure is direct.
        t = time.perf_counter()
        optimized_code, state_dict = optimize_generated_code(code, state_dict, enable=True)

        # Stage 5b: Apply Black-compatible formatting
        formatted_code = format_code(optimized_code)

        # Stage 5c: Add file header with metadata
        final_code = add_file_header(formatted_code, camel_class_name, onnx_path)

        # Save outputs
        if target_py_path is None:
            target_py_path = onnx_path.replace(".onnx", ".py")
        if target_pth_path is None:
            target_pth_path = onnx_path.replace(".onnx", ".pth")

        Path(target_py_path).write_text(final_code)
        torch.save(state_dict, target_pth_path)
        _logger.info(f"  Simplify + export ({time.perf_counter() - t:.4f}s)")
        _logger.info(f"  Output: {target_py_path}")
        _logger.info(f"  State dict: {target_pth_path}")
        _logger.info(f"  Total: {time.perf_counter() - t_total:.4f}s")

    @staticmethod
    def preprocess(
        onnx_path: str,
        target_opset: int | None = None,
        infer_shapes: bool = True,
        clear_docstrings: bool = True,
    ) -> onnx.ModelProto:
        """Load and preprocess ONNX model.

        Preprocessing steps:
        1. Load model from file
        2. Validate with ONNX checker
        3. Convert to target opset version (default: 21 for shapeonnx compatibility)
        4. Run shape inference (if enabled)
        5. Clear node docstrings (if enabled)
        6. Mark as processed by SlimONNX (if enabled)

        Recommended opset: 17-21 (tested with shapeonnx)
        Default: 21 (for shapeonnx compatibility)

        :param onnx_path: Path to ONNX model.
        :param target_opset: Target opset version (None = keep original, default = 21).
        :param infer_shapes: Run ONNX shape inference (default: True).
        :param clear_docstrings: Clear node docstrings (default: True).

        :return: Preprocessed model.
        """
        return load_and_preprocess_onnx_model(
            onnx_path,
            target_opset=target_opset,
            infer_shapes=infer_shapes,
            check_model=True,
            clear_docstrings=clear_docstrings,
        )
