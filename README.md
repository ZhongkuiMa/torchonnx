# TorchONNX: Convert ONNX Model to PyTorch Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![ONNX 1.17](https://img.shields.io/badge/ONNX-1.17-brightgreen.svg)](https://onnx.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

**torchonnx** is a compiler-based tool that converts ONNX models (.onnx files) into native PyTorch models (.pth files for parameters and .py files for model structure).

Extensively tested on [VNNCOMP 2024](https://github.com/ChristopherBrix/vnncomp2024_benchmarks) benchmarks including Vision Transformers, CNNs, and complex neural network architectures.


## Motivation

While PyTorch provides the `torch.onnx` module to convert PyTorch models to ONNX, the reverse process—converting ONNX models back to PyTorch—is not officially supported. This tool addresses this gap for several key reasons:

- **Version Fragmentation**: ONNX model format evolves across versions, with different versions supporting different operations. This creates significant compatibility challenges when working with models from various sources.

- **Framework Inconsistencies**: There are numerous inconsistencies between ONNX and PyTorch models in terms of naming conventions, parameter handling, and operational semantics. PyTorch does not officially support reverse conversion, likely considering it unnecessary for their ecosystem.

- **Neural Network Verification Requirements**: For the Neural Network Verification (NNV) community, ONNX has become the unified model format. Being able to work with these models natively in PyTorch is essential for research and verification tasks.

- **Code Quality and Maintainability**: ONNX's computational graph representation does not always align with logical groupings that make sense in PyTorch. We need a tool that generates clean, maintainable PyTorch code.

## Why TorchONNX?

While other tools exist for ONNX-to-PyTorch conversion, most fall short in performance and code quality. The most well-known tool, [onnx2pytorch](https://github.com/Talmaj/onnx2pytorch), serves as a runtime wrapper rather than a true compiler. Its forward method iterates over ONNX nodes at runtime instead of generating static PyTorch code, and parameter conversion is inefficient.

**torchonnx** takes a different approach: it is a true compiler that generates clean, efficient PyTorch code. The tool converts ONNX models into two separate files:

1. A .py file defining the neural network structure as native PyTorch code
2. A .pth file containing the model parameters as a state dictionary

This design eliminates runtime overhead and produces code that is readable, maintainable, and performs identically to hand-written PyTorch models.

## Key Advantages

### True Compiler Architecture
- **Zero Runtime Overhead**: Generated PyTorch code has no ONNX dependencies and runs at native PyTorch speed
- **Static Code Generation**: All operations are compiled to clean Python code, not interpreted at runtime
- **Optimized Parameter Handling**: Intelligent tracking eliminates unused parameters, reducing model size by up to 50%
- **Idiomatic PyTorch**: Uses `.expand()`, native type conversions, and PyTorch best practices throughout

### Production-Ready Code Quality
- **Complete Type Hints**: All generated code includes full type annotations for Python 3.10+
- **Clean Structure**: Human-readable modules with proper naming, documentation, and organization
- **No Dead Code**: Automatic elimination of unused operations and parameters
- **Formatted Output**: All code formatted with `black` for consistency

### Extensible and Maintainable
- **Pure Python Implementation**: No compiled dependencies, easy to inspect and modify
- **Modular Architecture**: Clean separation between IR construction, code generation, and optimization
- **Easy to Extend**: Add new operations or modify existing ones without breaking the codebase
- **Well-Documented**: reStructuredText docstrings with `:param:` and `:return:` annotations

### Comprehensive Testing
- **VNNCOMP 2024 Benchmarks**: Extensively tested on official neural network verification competition benchmarks
- **Diverse Model Coverage**: Successfully converts Vision Transformers, CNNs, MLPs, and complex architectures
- **Validated Output**: Generated models produce numerically identical results to original ONNX models

## Module Structure

```
torchonnx/
├── torchonnx/
│   ├── __init__.py              # Main converter interface
│   ├── converter.py             # TorchONNX class implementation
│   ├── ir/                      # Intermediate Representation
│   │   ├── __init__.py
│   │   ├── model_ir.py          # Model IR data structures
│   │   └── layer_ir.py          # Layer IR data structures
│   ├── code_generation/         # Code generation modules
│   │   ├── __init__.py
│   │   ├── module_gen.py        # Complete module generation
│   │   ├── forward_gen.py       # Forward method generation
│   │   ├── init_gen.py          # __init__ method generation
│   │   └── import_gen.py        # Import statement generation
│   ├── type_inference/          # Type inference and operation mapping
│   │   ├── __init__.py
│   │   └── type_mapping.py      # ONNX to PyTorch type mapping
│   └── optimization/            # Code optimization passes
│       ├── __init__.py
│       └── parameter_cleanup.py # Unused parameter elimination
├── test/                        # Testing infrastructure
│   ├── benchmarks/              # Original ONNX benchmark files
│   ├── baselines/               # Expected PyTorch outputs
│   ├── results/                 # Generated test outputs
│   └── test_benchmarks.py       # VNNCOMP 2024 test suite
└── README.md
```

## Installation

There are no complex installation steps. The tool requires:

- Python 3.10 or higher (tested with Python 3.12)
- `onnx==1.17.0`
- `numpy==2.2.4`
- `torch` (any recent version compatible with your system)

Please refer to the official [PyTorch](https://pytorch.org/) and [ONNX](https://github.com/onnx/onnx) installation guides for platform-specific instructions.

## Usage

### Example: ViT Model Conversion

The following example demonstrates conversion of a Vision Transformer (ViT) model from [VNNCOMP 2023](https://sites.google.com/view/vnn2023/home). Note that you should use [slimonnx](https://github.com/ZhongkuiMa/slimonnx) to simplify the model first, as the original may contain unsupported operations.

You can visualize the ONNX computational graph using [netron.app](https://netron.app).

```python
from torchonnx import TorchONNX

if __name__ == "__main__":
    file_path = "../nets/ibp_3_3_8_v22_simplified.onnx"
    converter = TorchONNX(verbose=True)
    converter.convert(file_path)
```

### Generated Code Example

The following shows generated PyTorch code for the ViT model. Note the clean structure, proper parameter registration, and readable forward pass:

```python
"""Generated PyTorch module from ONNX model."""

__docformat__ = "restructuredtext"
__all__ = ["Vit2023Pgd2316Model"]

import torch
import torch.nn as nn
import torch.nn.functional as F


class Vit2023Pgd2316Model(nn.Module):
    """Converted PyTorch module."""

    def __init__(self):
        """Initialize module."""
        super().__init__()

        # Register parameters
        self.param1 = nn.Parameter(torch.empty(48,))
        self.param2 = nn.Parameter(torch.empty(5, 48))
        self.weight1 = nn.Parameter(torch.empty(48, 3, 16, 16))
        self.bias1 = nn.Parameter(torch.empty(48,))
        self.bias3 = nn.Parameter(torch.empty(48,))
        self.bias4 = nn.Parameter(torch.empty(48,))
        self.bias5 = nn.Parameter(torch.empty(48,))
        self.bias6 = nn.Parameter(torch.empty(48,))
        self.bias8 = nn.Parameter(torch.empty(96,))
        self.bias9 = nn.Parameter(torch.empty(48,))
        self.bias11 = nn.Parameter(torch.empty(48,))
        self.bias12 = nn.Parameter(torch.empty(48,))
        self.bias13 = nn.Parameter(torch.empty(48,))
        self.bias14 = nn.Parameter(torch.empty(48,))
        self.bias16 = nn.Parameter(torch.empty(96,))
        self.bias17 = nn.Parameter(torch.empty(48,))
        self.weight8 = nn.Parameter(torch.empty(48, 48))
        self.weight9 = nn.Parameter(torch.empty(48, 48))
        self.weight10 = nn.Parameter(torch.empty(48, 48))
        self.weight11 = nn.Parameter(torch.empty(48, 48))
        self.weight12 = nn.Parameter(torch.empty(48, 96))
        self.weight13 = nn.Parameter(torch.empty(96, 48))
        self.weight14 = nn.Parameter(torch.empty(48, 48))
        self.weight15 = nn.Parameter(torch.empty(48, 48))
        self.weight16 = nn.Parameter(torch.empty(48, 48))
        self.weight17 = nn.Parameter(torch.empty(48, 48))
        self.weight18 = nn.Parameter(torch.empty(48, 96))
        self.weight19 = nn.Parameter(torch.empty(96, 48))
        self.param17 = nn.Parameter(torch.empty(1,))
        self.param19 = nn.Parameter(torch.empty(1,))
        self.param20 = nn.Parameter(torch.empty(1,))
        self.param23 = nn.Parameter(torch.empty(1,))
        self.param24 = nn.Parameter(torch.empty(1,))
        self.param25 = nn.Parameter(torch.empty(1,))
        self.param27 = nn.Parameter(torch.empty(1,))
        self.param28 = nn.Parameter(torch.empty(1,))
        self.param29 = nn.Parameter(torch.empty(1,))
        self.param31 = nn.Parameter(torch.empty(1,))
        self.param32 = nn.Parameter(torch.empty(1,))
        self.param33 = nn.Parameter(torch.empty(1,))
        self.param34 = nn.Parameter(torch.empty(()))
        self.param36 = nn.Parameter(torch.empty(1,))
        self.param37 = nn.Parameter(torch.empty(1,))
        self.param40 = nn.Parameter(torch.empty(1,))
        self.param41 = nn.Parameter(torch.empty(1,))
        self.param42 = nn.Parameter(torch.empty(1,))
        self.param44 = nn.Parameter(torch.empty(1,))
        self.param45 = nn.Parameter(torch.empty(1,))
        self.param46 = nn.Parameter(torch.empty(1,))
        self.param48 = nn.Parameter(torch.empty(1,))
        self.param49 = nn.Parameter(torch.empty(1,))
        self.param50 = nn.Parameter(torch.empty(1,))
        self.param51 = nn.Parameter(torch.empty(()))
        self.param53 = nn.Parameter(torch.empty(1,))
        self.param54 = nn.Parameter(torch.empty(1,))

        self.bn1 = nn.BatchNorm2d(num_features=48, eps=9.999999747378752e-06, momentum=0.10000002384185791, track_running_stats=True)
        self.softmax1 = nn.Softmax(dim=-1)
        self.bn2 = nn.BatchNorm2d(num_features=48, eps=9.999999747378752e-06, momentum=0.10000002384185791, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(num_features=48, eps=9.999999747378752e-06, momentum=0.10000002384185791, track_running_stats=True)
        self.softmax2 = nn.Softmax(dim=-1)
        self.bn4 = nn.BatchNorm2d(num_features=48, eps=9.999999747378752e-06, momentum=0.10000002384185791, track_running_stats=True)
        self.relu2 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(num_features=48, eps=9.999999747378752e-06, momentum=0.10000002384185791, track_running_stats=True)
        self.fc1 = nn.Linear(in_features=48, out_features=10, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x1 = torch.tensor(x.shape, dtype=torch.int64)
        x2 = x1[0]
        x3 = F.conv2d(x, self.weight1, self.bias1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), groups=1)
        x4 = torch.tensor(x3.shape, dtype=torch.int64)
        x5 = x4[:2]
        x6 = torch.cat([x5, self.param17], dim=0)
        x7 = x3.reshape(tuple(x6.flatten().int().tolist()))
        x8 = x7.permute([0, 2, 1])
        x9 = x2.unsqueeze(0)
        x10 = torch.cat([x9, self.param19, self.param20], dim=0)
        x11 = torch.full(tuple(x10.int().tolist()), 0.0, dtype=x.dtype)
        x12 = x11 + self.param1
        x13 = torch.cat([x12, x8], dim=1)
        x14 = x13 + self.param2
        x15 = x14.permute([0, 2, 1])
        x16 = self.bn1(x15.unsqueeze(-1)).squeeze(-1)
        x17 = x16.permute([0, 2, 1])
        x18 = torch.tensor(x17.shape, dtype=torch.int64)
        x19 = x18[0]
        x20 = x17 @ self.weight8
        x21 = self.bias3 + x20
        x22 = x17 @ self.weight9
        x23 = self.bias4 + x22
        x24 = x17 @ self.weight10
        x25 = self.bias5 + x24
        x26 = x19.unsqueeze(0)
        x27 = torch.cat([x26, self.param23, self.param24, self.param25], dim=0)
        x28 = x19.unsqueeze(0)
        x29 = torch.cat([x28, self.param27, self.param28, self.param29], dim=0)
        x30 = x19.unsqueeze(0)
        x31 = torch.cat([x30, self.param31, self.param32, self.param33], dim=0)
        x32 = x21.reshape(tuple(x27.flatten().int().tolist()))
        x33 = x32.permute([0, 2, 1, 3])
        x34 = x23.reshape(tuple(x29.flatten().int().tolist()))
        x35 = x25.reshape(tuple(x31.flatten().int().tolist()))
        x36 = x35.permute([0, 2, 1, 3])
        x37 = x34.permute([0, 2, 3, 1])
        x38 = x33 @ x37
        x39 = x38 * self.param34
        x40 = torch.tensor(x39.shape, dtype=torch.int64)
        x41 = torch.flatten(x39, start_dim=3)
        x42 = self.softmax1(x41)
        x43 = x42.reshape(tuple(x40.flatten().int().tolist()))
        x44 = x43 @ x36
        x45 = x44.permute([0, 2, 1, 3])
        x46 = x19.unsqueeze(0)
        x47 = torch.cat([x46, self.param36, self.param37], dim=0)
        x48 = x45.reshape(tuple(x47.flatten().int().tolist()))
        x49 = x48 @ self.weight11
        x50 = self.bias6 + x49
        x51 = x50 + x14
        x52 = x51.permute([0, 2, 1])
        x53 = self.bn2(x52.unsqueeze(-1)).squeeze(-1)
        x54 = x53.permute([0, 2, 1])
        x55 = x54 @ self.weight12
        x56 = self.bias8 + x55
        x57 = self.relu1(x56)
        x58 = x57 @ self.weight13
        x59 = self.bias9 + x58
        x60 = x59 + x51
        x61 = x60.permute([0, 2, 1])
        x62 = self.bn3(x61.unsqueeze(-1)).squeeze(-1)
        x63 = x62.permute([0, 2, 1])
        x64 = torch.tensor(x63.shape, dtype=torch.int64)
        x65 = x64[0]
        x66 = x63 @ self.weight14
        x67 = self.bias11 + x66
        x68 = x63 @ self.weight15
        x69 = self.bias12 + x68
        x70 = x63 @ self.weight16
        x71 = self.bias13 + x70
        x72 = x65.unsqueeze(0)
        x73 = torch.cat([x72, self.param40, self.param41, self.param42], dim=0)
        x74 = x65.unsqueeze(0)
        x75 = torch.cat([x74, self.param44, self.param45, self.param46], dim=0)
        x76 = x65.unsqueeze(0)
        x77 = torch.cat([x76, self.param48, self.param49, self.param50], dim=0)
        x78 = x67.reshape(tuple(x73.flatten().int().tolist()))
        x79 = x78.permute([0, 2, 1, 3])
        x80 = x69.reshape(tuple(x75.flatten().int().tolist()))
        x81 = x71.reshape(tuple(x77.flatten().int().tolist()))
        x82 = x81.permute([0, 2, 1, 3])
        x83 = x80.permute([0, 2, 3, 1])
        x84 = x79 @ x83
        x85 = x84 * self.param51
        x86 = torch.tensor(x85.shape, dtype=torch.int64)
        x87 = torch.flatten(x85, start_dim=3)
        x88 = self.softmax2(x87)
        x89 = x88.reshape(tuple(x86.flatten().int().tolist()))
        x90 = x89 @ x82
        x91 = x90.permute([0, 2, 1, 3])
        x92 = x65.unsqueeze(0)
        x93 = torch.cat([x92, self.param53, self.param54], dim=0)
        x94 = x91.reshape(tuple(x93.flatten().int().tolist()))
        x95 = x94 @ self.weight17
        x96 = self.bias14 + x95
        x97 = x96 + x60
        x98 = x97.permute([0, 2, 1])
        x99 = self.bn4(x98.unsqueeze(-1)).squeeze(-1)
        x100 = x99.permute([0, 2, 1])
        x101 = x100 @ self.weight18
        x102 = self.bias16 + x101
        x103 = self.relu2(x102)
        x104 = x103 @ self.weight19
        x105 = self.bias17 + x104
        x106 = x105 + x97
        x107 = torch.mean(x106, dim=(1,), keepdim=False)
        x108 = self.bn5(x107.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        x109 = self.fc1(x108)
        return x109

```

## Testing & Validation

TorchONNX is extensively tested on the [VNNCOMP 2024 benchmarks](https://github.com/ChristopherBrix/vnncomp2024_benchmarks), the official benchmark suite for neural network verification competitions. The test suite includes:

- **Vision Transformers (ViT)**: Complex transformer architectures with attention mechanisms
- **Convolutional Neural Networks**: Various CNN architectures from traffic sign detection to autonomous control
- **Feedforward Networks**: MLPs with various activation functions and normalizations
- **Hybrid Architectures**: Models combining multiple architectural patterns

All converted models are validated to produce numerically identical outputs to their original ONNX counterparts, ensuring correctness across diverse model types and operations.

### Running Tests

To test with VNNCOMP 2024 benchmarks, clone the [vnncomp2024](https://github.com/ChristopherBrix/vnncomp2024_benchmarks) repository and ensure the following structure:

```
torchonnx/
│   ├── torchonnx/
│   ├── README.md
│   └── test/
└── ...
vnncomp2024/
│   ├── benchmarks/
└── ...
```

## Supported Operations

The tool implements most commonly used operations in feedforward neural networks, including:

- **Convolution**: Conv1d, Conv2d, ConvTranspose
- **Pooling**: MaxPool, AveragePool, GlobalAveragePool
- **Normalization**: BatchNorm, LayerNorm, InstanceNorm
- **Activation**: ReLU, Sigmoid, Tanh, Softmax, GELU
- **Linear**: MatMul, Gemm, Linear layers
- **Shape Operations**: Reshape, Transpose, Flatten, Expand, Squeeze, Unsqueeze
- **Tensor Operations**: Concat, Slice, Gather, ScatterND
- **Arithmetic**: Add, Sub, Mul, Div, Pow
- **Reduction**: ReduceMean, ReduceSum, ReduceMax

Transformer-based architectures are decomposed into basic operations and handled correctly.

## Contributing

Contributions are welcome from the community. Whether fixing bugs, adding features, improving documentation, or sharing ideas, all contributions are appreciated.

Note: Direct pushes to the `main` branch are restricted. Please fork the repository and submit a Pull Request for any changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
