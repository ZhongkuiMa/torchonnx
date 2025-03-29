__docformat__ = "restructuredtext"
__all__ = ["TorchONNX"]

import time

import onnx
import torch
from torch import Tensor

from .forward_part import gen_forward_code
from .header_part import gen_header_code
from .init_part import gen_init_code


def _gen_module_name(file_path: str) -> str:
    file_name = file_path.split("/")[-1].split(".")[0]
    # Remove all dots and dashes
    file_name = file_name.replace(".", "").replace("-", "")
    # Change to title case
    file_name = file_name.title().replace("_", "")
    return file_name


def _convert_initializers(model: onnx.ModelProto) -> dict[str, Tensor]:
    initializers = {}
    for initializer in model.graph.initializer:
        tensor = torch.tensor(onnx.numpy_helper.to_array(initializer))
        initializers[initializer.name] = tensor

    return initializers


class TorchONNX:
    """Generate a torch model file from an onnx file."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def convert(
        self,
        onnx_path: str,
        module_class_name: str = None,
        target_py_path: str = None,
        target_pth_path: str = None,
    ):
        if module_class_name is None:
            module_class_name = _gen_module_name(onnx_path)
        if target_py_path is None:
            target_py_path = onnx_path.replace(".onnx", ".py")
        if target_pth_path is None:
            target_pth_path = onnx_path.replace(".onnx", ".pth")

        if self.verbose:
            print(f"Converting {onnx_path}...")

        model = onnx.load(onnx_path)

        if self.verbose:
            print(f"Extracting initializers...")
            t = time.perf_counter()

        initializers = _convert_initializers(model)
        torch.save(initializers, target_pth_path)

        if self.verbose:
            t = time.perf_counter() - t
            print(f"Saved initializers to {target_pth_path} ({t:.4f}s)")

        if self.verbose:
            print(f"Generating pytorch code...")
            t = time.perf_counter()

        content = gen_header_code(model, module_class_name)
        with open(target_py_path, "w") as f:
            f.write(content)

        content = gen_init_code(model, target_pth_path)
        with open(target_py_path, "a") as f:
            f.write(content)

        content = gen_forward_code(model)
        with open(target_py_path, "a") as f:
            f.write(content)

        if self.verbose:
            t = time.perf_counter() - t
            print(f"Saved pytorch code to {target_py_path} ({t:.4f}s)")
