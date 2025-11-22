from pathlib import Path

import onnx
import torch

from .code_generation import generate_pytorch_module_with_state_dict
from .ir import build_model_ir


class TorchONNX:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert(
        self,
        onnx_path: str,
        benchmark_name: str | None = None,
        target_py_path: str = None,
        target_pth_path: str = None,
    ):
        # Preprocess model (load, convert to opset 17, clear docs, mark SlimONNX)
        model = self.preprocess(
            onnx_path,
            target_opset=17,
            infer_shapes=False,
            clear_docstrings=True,
            mark_slimonnx=True,
        )

        # Build intermediate representation
        model_ir = build_model_ir(model)

        model_name = Path(onnx_path).stem
        code, state_dict = generate_pytorch_module_with_state_dict(
            model_ir, model_name, benchmark_name
        )

        # Save outputs
        if target_py_path is None:
            target_py_path = onnx_path.replace(".onnx", ".py")
        if target_pth_path is None:
            target_pth_path = onnx_path.replace(".onnx", ".pth")
        Path(target_py_path).write_text(code)
        torch.save(state_dict, target_pth_path)

    def preprocess(
        self,
        onnx_path: str,
        target_opset: int | None = None,
        infer_shapes: bool = True,
        clear_docstrings: bool = True,
        mark_slimonnx: bool = True,
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

        :param onnx_path: Path to ONNX model
        :param target_opset: Target opset version (None = keep original, default = 21)
        :param infer_shapes: Run ONNX shape inference (default: True)
        :param clear_docstrings: Clear node docstrings (default: True)
        :param mark_slimonnx: Mark model as processed by SlimONNX (default: True)
        :return: Preprocessed model
        """
        from .preprocess import load_and_preprocess

        return load_and_preprocess(
            onnx_path,
            target_opset=target_opset,
            infer_shapes=infer_shapes,
            check_model=True,
            clear_docstrings=clear_docstrings,
            mark_slimonnx=mark_slimonnx,
        )
