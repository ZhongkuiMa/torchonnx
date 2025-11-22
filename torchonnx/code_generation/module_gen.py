"""Complete PyTorch module code generation.

This module generates complete, executable PyTorch module code from IR,
including imports, class definition, __init__, and forward() methods.
"""

__docformat__ = "restructuredtext"
__all__ = ["generate_pytorch_module_with_state_dict"]

import torch
from onnx import TensorProto, numpy_helper

from .forward_gen import generate_forward_method
from .init_gen import generate_init_method
from ..ir import ModelIR, LayerIR
from ..naming import sanitize_parameter_names, sanitize_module_name


def generate_imports(model_ir: ModelIR) -> str:
    """Generate import statements based on model requirements.

    :param model_ir: Model intermediate representation
    :return: Python import statements
    """
    imports: list[str] = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
    ]

    return "\n".join(imports)


def indent(text: str, prefix: str) -> str:
    """Indent each line of text with the given prefix.

    :param text: Text to indent
    :param prefix: Prefix to add to each line
    :return: Indented text
    """
    lines = text.split("\n")
    return "\n".join(prefix + line if line.strip() else line for line in lines)


def generate_module_code(model_ir: ModelIR, class_name: str) -> str:
    """Generate complete PyTorch module code with simplified parameter names.

    Combines all components into a complete, executable Python module:
    - Module docstring
    - Imports
    - Class definition with __init__ and forward() methods
    - Parameter registrations with simplified names (weight1, bias1, etc.)
    - Forward pass using @ operator and simplified parameter references

    Example output:
        \"\"\"Generated PyTorch module from ONNX model.\"\"\"

        __docformat__ = "restructuredtext"
        __all__ = ["ConvertedModel"]

        import torch
        import torch.nn as nn


        class ConvertedModel(nn.Module):
            \"\"\"Converted PyTorch module.\"\"\"

            def __init__(self):
                super().__init__()

                # Register parameters
                self.weight1 = nn.Parameter(torch.empty(64, 3, 3, 3))
                self.bias1 = nn.Parameter(torch.empty(64))

                self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
                self.relu1 = nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x1 = self.conv1(x)
                x2 = self.relu1(x1)
                return x2

    :param model_ir: Model intermediate representation
    :param class_name: Name for the generated PyTorch module class
    :return: Complete Python module code
    """
    imports = generate_imports(model_ir)

    input_names = [inp.name for inp in model_ir.inputs]
    output_names = [out.name for out in model_ir.outputs]

    initializer_names = list(model_ir.parameters.keys())
    name_mapping = sanitize_parameter_names(initializer_names)

    init_method = generate_init_method(
        model_ir.layers,
        model_ir.parameters,
        name_mapping,
    )
    forward_method = generate_forward_method(
        model_ir.layers,
        input_names,
        output_names,
        name_mapping,
        model_ir.parameters,
    )

    code = f'''"""Generated PyTorch module from ONNX model."""

__docformat__ = "restructuredtext"
__all__ = ["{class_name}"]

{imports}


class {class_name}(nn.Module):
    """Converted PyTorch module."""

{indent(init_method, "    ")}

{indent(forward_method, "    ")}
'''

    return code


def build_state_dict(
    layers: list[LayerIR],
    initializers: dict[str, TensorProto],
    name_mapping: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    """Build PyTorch state_dict from ONNX initializers.

    Converts ALL ONNX initializers to PyTorch tensors with both hierarchical
    and simplified names as needed:
    - Layer parameters use hierarchical names (e.g., conv1.weight, bn1.bias)
    - Standalone parameters use simplified names (e.g., weight1, param1)

    Example:
        ONNX initializers:
            'conv_weight' -> [64, 3, 3, 3]
            'bn_weight' -> [64]
            'input_AvgImg' -> [5]

        PyTorch state_dict:
            'conv1.weight' -> torch.Tensor([64, 3, 3, 3])
            'bn1.weight' -> torch.Tensor([64])
            'param1' -> torch.Tensor([5])

    :param layers: List of LayerIR with parameter mappings
    :param initializers: ONNX initializers (TensorProto)
    :param name_mapping: Mapping from ONNX names to simplified names
    :return: PyTorch state_dict
    """
    from ..type_inference import is_parametric_layer

    state_dict: dict[str, torch.Tensor] = {}
    covered_initializers: set[str] = set()

    for layer in layers:
        if not is_parametric_layer(layer.layer_type):
            continue

        for param_name, tensor_name in layer.parameters.items():
            if tensor_name not in initializers:
                continue

            if layer.layer_type == "Upsample" and param_name in {"scales", "sizes"}:
                covered_initializers.add(tensor_name)
                continue

            onnx_tensor = initializers[tensor_name]
            numpy_array = numpy_helper.to_array(onnx_tensor)
            pytorch_tensor = torch.from_numpy(numpy_array.copy())

            state_dict_key = f"{layer.layer_name}.{param_name}"
            state_dict[state_dict_key] = pytorch_tensor
            covered_initializers.add(tensor_name)

    if name_mapping:
        for onnx_name, simplified_name in name_mapping.items():
            if onnx_name not in initializers:
                continue

            if onnx_name in covered_initializers:
                continue

            onnx_tensor = initializers[onnx_name]
            numpy_array = numpy_helper.to_array(onnx_tensor)
            pytorch_tensor = torch.from_numpy(numpy_array.copy())

            state_dict[simplified_name] = pytorch_tensor

    return state_dict


def generate_pytorch_module_with_state_dict(
    model_ir: ModelIR,
    model_name: str,
    benchmark_name: str | None = None,
) -> tuple[str, dict[str, torch.Tensor]]:
    """Generate PyTorch module code and state_dict from IR.

    This is the main entry point for Stage 3 compilation, which produces:
    1. Python code for the PyTorch module with simplified parameter names
    2. state_dict with ALL parameters (including functional operation parameters)

    The class name is automatically sanitized from model_name and benchmark_name
    to ensure valid Python identifiers (e.g., "vgg16-7" + "vggnet16_2023" → "Vggnet162023Vgg167Model").

    Example:
        model_ir = build_intermediate_representation(onnx_model)
        code, state_dict = generate_pytorch_module(
            model_ir,
            model_name="resnet50",
            benchmark_name="cifar100"
        )

        # Save outputs
        Path("model.py").write_text(code)
        torch.save(state_dict, "model.pth")

        # Use module
        model = Cifar100Resnet50Model()
        model.load_state_dict(state_dict)

    :param model_ir: Model intermediate representation from Stage 1
    :param model_name: Original model name from ONNX file
    :param benchmark_name: Optional benchmark name (prepended to avoid naming conflicts)
    :return: Tuple of (python_code, state_dict)
    """
    class_name = sanitize_module_name(model_name, benchmark_name)

    initializer_names = list(model_ir.parameters.keys())
    name_mapping = sanitize_parameter_names(initializer_names)

    code = generate_module_code(model_ir, class_name)

    state_dict = build_state_dict(model_ir.layers, model_ir.parameters, name_mapping)

    return code, state_dict
