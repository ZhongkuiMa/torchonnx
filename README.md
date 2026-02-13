# TorchONNX: Convert ONNX Model to PyTorch Model

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/torchonnx/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/torchonnx/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/torchonnx/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/torchonnx)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![ONNX 1.16](https://img.shields.io/badge/ONNX-1.16-brightgreen.svg)](https://onnx.ai)
[![Version](https://img.shields.io/github/v/tag/ZhongkuiMa/torchonnx?sort=semver)](https://github.com/ZhongkuiMa/torchonnx/releases)

**torchonnx** is a compiler-based tool that converts ONNX models (.onnx files) into native PyTorch models (.pth files for parameters and .py files for model structure).

**Extensively tested on [VNNCOMP 2024](https://github.com/ChristopherBrix/vnncomp2024_benchmarks) benchmarks including Vision Transformers, CNNs, and complex neural network architectures.**

## Motivation

While PyTorch provides the `torch.onnx` module to convert PyTorch models to ONNX, the reverse process—converting ONNX models back to PyTorch—is not officially supported. This tool addresses this gap for several key reasons:

- **Version Fragmentation**: ONNX model format evolves across versions, with different versions supporting different operations. This creates significant compatibility challenges when working with models from various sources.

- **Framework Inconsistencies**: There are numerous inconsistencies between ONNX and PyTorch models in terms of naming conventions, parameter handling, and operational semantics. PyTorch does not officially support reverse conversion, likely considering it unnecessary for their ecosystem.

- **Neural Network Verification Requirements**: For the Neural Network Verification (NNV) community, ONNX has become the unified model format. Being able to work with these models natively in PyTorch is essential for research and verification tasks.

- **Code Quality and Maintainability**: ONNX's computational graph representation does not always align with logical groupings that make sense in PyTorch. We need a tool that generates clean, maintainable PyTorch code.

- **Dynamic Batch Dimension**: Many ONNX models are exported with hardcoded batch size (typically `batch_size=1`), using operations like `reshape(1, ...)`. TorchONNX generates code that supports dynamic batch dimensions, making models compatible with arbitrary batch sizes.

- **Vectorization Support**: For neural network verification and adversarial robustness testing, efficient batch processing via `torch.vmap` is essential. TorchONNX provides a `vmap_mode` that generates vectorization-compatible code, enabling efficient parallel evaluation across multiple inputs.

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
- **Optimized Parameter Handling**: Intelligent tracking eliminates unused parameters, reducing model size
- **Cached Constants**: Constant tensors are registered as buffers for efficient device management
- **Idiomatic PyTorch**: Uses native PyTorch operations, type conversions, and best practices throughout

### Production-Ready Code Quality

- **Complete Type Hints**: All generated code includes full type annotations for Python 3.10+
- **Clean Structure**: Human-readable modules with proper naming, documentation, and organization
- **No Dead Code**: Automatic elimination of unused operations, parameters, and buffers
- **Code Optimization**: Post-processing removes default arguments and converts to positional arguments
- **Formatted Output**: All code formatted with `black` for consistency

### Extensible and Maintainable

- **Pure Python Implementation**: No compiled dependencies, easy to inspect and modify
- **Modular Architecture**: Clean 6-stage compiler pipeline with separation of concerns
- **Easy to Extend**: Add new operations or modify existing ones without breaking the codebase
- **Well-Documented**: reStructuredText docstrings with `:param:` and `:return:` annotations

### Comprehensive Testing

- **VNNCOMP 2024 Benchmarks**: Extensively tested on official neural network verification competition benchmarks
- **Diverse Model Coverage**: Successfully converts Vision Transformers, CNNs, MLPs, and complex architectures
- **Validated Output**: Generated models produce numerically identical results to original ONNX models

## Vectorization (vmap) Mode

TorchONNX supports generating `torch.vmap`-compatible code for efficient batched evaluation. This is particularly important for neural network verification and adversarial robustness testing where many inputs need to be evaluated in parallel.

### The Challenge

Standard ONNX models often contain operations that are incompatible with `torch.vmap`:

1. **In-place operations**: Operations like `index_put_` break vmap's functional requirements
2. **Dynamic `.item()` calls**: Converting tensor values to Python scalars is not vmap-compatible
3. **Input-dependent control flow**: Conditional behavior based on input values

### vmap Mode Features

When `vmap_mode=True` is specified during conversion:

- **Functional helpers**: Uses `torch.scatter` instead of in-place `index_put_`
- **Tensor-based indexing**: Uses `torch.gather` with pre-computed slice lengths instead of `.item()` calls
- **Validity flag propagation**: Tracks empty/out-of-bounds slices and propagates validity to downstream operations

### cctsdb_yolo_2023 Benchmark

The `cctsdb_yolo_2023` benchmark from VNNCOMP presents a particularly challenging case for vectorization:

- **Input-dependent dynamic slicing**: Slice indices are computed from input values
- **Out-of-bounds handling**: When slice indices exceed array bounds, standard mode returns empty tensors while vmap mode must return fixed-shape tensors

TorchONNX solves this with **validity flag propagation**:

1. `dynamic_slice` returns `(result, valid_flag)` where `valid_flag=0` indicates out-of-bounds
2. Validity flags are accumulated across multiple slice operations
3. `scatter_nd` receives the validity flag and returns original data unchanged when `valid=0`

This ensures **identical outputs** between standard and vmap modes for all inputs, including edge cases.

### Usage

```python
from torchonnx import TorchONNX

converter = TorchONNX(verbose=True)

# Default conversion (vmap_mode=True by default)
converter.convert("model.onnx", target_py_path="model.py")

# Explicitly disable vmap mode if needed (for legacy compatibility)
converter.convert("model.onnx", target_py_path="model_legacy.py", vmap_mode=False)
```

### Applying vmap

```python
import torch

# Load vmap-compatible model
from model_vmap import Model
model = Model()
model.load_state_dict(torch.load("model.pth"))

# Batch of inputs
batch_inputs = torch.randn(100, 3, 64, 64)

# Vectorized evaluation
vmapped_model = torch.vmap(model)
batch_outputs = vmapped_model(batch_inputs)
```

## Compiler Architecture

TorchONNX implements a 6-stage compiler pipeline that transforms ONNX models into optimized PyTorch code:

### Stage 1: Normalization

Loads and normalizes ONNX models to a consistent format:

- Model validation using ONNX checker
- Opset version conversion (target: opset 20)
- Shape inference using ONNX shape inference or [shapeonnx](https://github.com/ZhongkuiMa/shapeonnx)
- Metadata cleanup

**Key Files**: `normalize/normalize.py`, `normalize/utils.py`

### Stage 2: Structural IR Building

Extracts pure structural information from ONNX graph:

- Builds `ModelIR` containing list of `NodeIR` instances
- Captures graph topology, tensor shapes, and initializers
- No semantic interpretation at this stage (pure structural representation)

**Key Files**: `build/builder.py`, `build/types.py`

### Stage 3: Semantic IR Building

Transforms structural IR into semantic IR with PyTorch types:

- Classifies initializers into parameters (trainable), constants (buffers), and arguments (literals)
- Maps ONNX operations to PyTorch types (layers, functions, operators)
- Resolves tensor data types and shapes
- Builds `SemanticModelIR` with typed inputs (`VariableInfo`, `ParameterInfo`, `ConstantInfo`, `ArgumentInfo`)

**Key Files**:

- `analyze/builder.py` - Main semantic IR builder
- `analyze/types.py` - Semantic type definitions
- `analyze/tensor_classifier.py` - Tensor classification logic
- `analyze/type_mapping/` - ONNX to PyTorch type mappings
- `analyze/attr_extractor.py` - ONNX attribute extraction

### Stage 4: IR Optimization

Optimizes semantic IR before code generation:

- Constant folding (future)
- Dead code elimination (future)
- Operation fusion (future)

**Key Files**: `optimize/optimizer.py`

### Stage 5: Code Generation

Generates PyTorch module code from semantic IR:

- `__init__` method: Parameter/constant registration and layer construction
- `forward` method: Operation-by-operation code generation using handlers
- State dict: Parameter and constant tensors
- Import statements and module structure

**Key Files**:

- `generate/code_generator.py` - Main orchestrator
- `generate/_init_gen.py` - `__init__` method generation
- `generate/_forward_gen.py` - `forward` method generation
- `generate/_state_dict_gen.py` - State dict building
- `generate/_templates.py` - Code templates
- `generate/_handlers/` - Operation-specific code generators

**Operation Handlers**:

- `_layers.py` - Layer handlers (nn.Conv2d, nn.Linear, etc.)
- `_operators.py` - Operator handlers (torch.add, torch.matmul, etc.)
- `_operations.py` - Function handlers (reshape, concat, slice, etc.)
- `_registry.py` - Handler registration system

### Stage 6: Code Optimization

Post-processes generated code for cleanliness:

- Removes unused buffer registrations using regex parsing
- Removes default arguments from layer constructors (e.g., `bias=True` → removed)
- Removes default arguments from functions (e.g., `F.relu(x, inplace=False)` → `F.relu(x)`)
- Converts named arguments to positional where appropriate (e.g., `nn.Conv2d(in_channels=3, out_channels=64)` → `nn.Conv2d(3, 64)`)
- Filters state dict to exclude removed buffers

**Key Files**:

- `simplify/_optimizer.py` - Main optimizer orchestrator
- `simplify/_line_optimizer.py` - Line-by-line optimization
- `simplify/_rules.py` - Optimization rules and patterns

## Module Structure

```
torchonnx/
├── torchonnx/
│   ├── __init__.py                   # Exports TorchONNX class
│   ├── _torchonnx.py                 # TorchONNX class (main API)
│   ├── normalize/                    # Stage 1: ONNX normalization
│   │   ├── __init__.py
│   │   ├── normalize.py              # Model preprocessing
│   │   └── utils.py                  # ONNX utilities
│   ├── build/                        # Stage 2: Structural IR
│   │   ├── __init__.py
│   │   ├── builder.py                # IR builder
│   │   └── types.py                  # NodeIR, ModelIR types
│   ├── analyze/                      # Stage 3: Semantic IR
│   │   ├── __init__.py
│   │   ├── builder.py                # Semantic IR builder
│   │   ├── types.py                  # Semantic type definitions
│   │   ├── tensor_classifier.py      # Tensor classification
│   │   ├── attr_extractor.py         # Attribute extraction
│   │   └── type_mapping/             # ONNX → PyTorch mappings
│   │       ├── _layers.py            # Layer type mappings
│   │       └── _operations.py        # Operation type mappings
│   ├── optimize/                     # Stage 4: IR optimization
│   │   ├── __init__.py
│   │   └── optimizer.py              # IR-level optimizations
│   ├── generate/                     # Stage 5: Code generation
│   │   ├── __init__.py
│   │   ├── code_generator.py         # Main code generator
│   │   ├── _init_gen.py              # __init__ generation
│   │   ├── _forward_gen.py           # forward() generation
│   │   ├── _state_dict_gen.py        # State dict building
│   │   ├── _templates.py             # Code templates
│   │   ├── _utils.py                 # Helper utilities
│   │   └── _handlers/                # Operation-specific handlers
│   │       ├── __init__.py
│   │       ├── _registry.py          # Handler registry
│   │       ├── _layers.py            # Layer handlers
│   │       ├── _operators.py         # Operator handlers
│   │       └── _operations.py        # Function handlers
│   └── simplify/                     # Stage 6: Code optimization
│       ├── __init__.py
│       ├── _optimizer.py             # Main optimizer
│       ├── _line_optimizer.py        # Line optimizations
│       └── _rules.py                 # Optimization rules
├── tests/                            # Testing infrastructure
│   ├── benchmarks/                   # Original ONNX files
│   ├── baselines/                    # Expected outputs
│   ├── results/                      # Generated outputs
│   ├── analyze_model_nodes.py        # Model node analyzer
│   ├── build_benchmarks.py           # Benchmark builder
│   ├── test_benchmarks.py            # VNNCOMP 2024 tests
│   └── utils.py                      # Test utilities
└── README.md
```

## Installation

**Note:** TorchONNX is not published on PyPI. Install locally from source.

### Quick Install

Clone the repository and install in development mode:

```bash
git clone https://github.com/ZhongkuiMa/torchonnx.git
cd torchonnx
pip install -e .
```

### Requirements

- Python >= 3.11
- PyTorch 2.3.1
- ONNX 1.16.0
- ONNXRuntime 1.22.0
- NumPy 1.26.4

### Development Installation

For development with linting and testing tools:

```bash
pip install -e ".[dev]"
```

This installs additional dependencies:
- pytest >= 7.0
- pytest-cov >= 4.0
- ruff >= 0.14.0
- mypy >= 1.0

## Usage

### Basic Example

```python
from torchonnx import TorchONNX

if __name__ == "__main__":
    # Create converter instance
    converter = TorchONNX(verbose=True)

    # Convert ONNX model to PyTorch
    converter.convert(
        onnx_path="model.onnx",
        benchmark_name="mymodel",  # Optional: for module naming
        target_py_path="model.py",  # Optional: defaults to model.py
        target_pth_path="model.pth"  # Optional: defaults to model.pth
    )
```

### Advanced Example: ViT Model Conversion

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
__all__ = ["Vit2023Ibp338"]

import torch
import torch.nn as nn


def dynamic_slice(data, starts, ends, axes=None, steps=None):
    """Dynamic slice helper for ONNX Slice operation."""
    # Ensure tensor
    starts = torch.as_tensor(starts, device=data.device)
    ends = torch.as_tensor(ends, device=data.device)
    if axes is None:
        axes = torch.arange(starts.numel(), device=data.device)
    else:
        axes = torch.as_tensor(axes, device=data.device)
    if steps is None:
        steps = torch.ones_like(starts)
    else:
        steps = torch.as_tensor(steps, device=data.device)

    # Normalize negative starts/ends
    dims = torch.as_tensor(data.shape, device=data.device)
    # axes tells where to read dim size
    dim_sizes = dims[axes]

    starts = torch.where(starts < 0, dim_sizes + starts, starts)
    ends = torch.where(ends < 0, dim_sizes + ends, ends)

    # Clip to bounds (ONNX semantics)
    # Use tensors for both min and max to avoid type mismatch
    zero = torch.zeros_like(dim_sizes)
    starts = torch.clamp(starts, min=zero, max=dim_sizes)
    ends = torch.clamp(ends, min=zero, max=dim_sizes)

    # Build index tuple dynamically
    index = [slice(None)] * data.ndim
    for i in range(axes.shape[0]):
        ax = axes[i].item()
        idx = torch.arange(starts[i], ends[i], steps[i], device=data.device)
        index[ax] = idx

    return data[tuple(index)]


class Vit2023Ibp338(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("c4", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c6", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c7", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c8", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c9", torch.empty([17, 48], dtype=torch.float32))
        self.register_buffer("c11", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c12", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c13", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c14", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c15", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c16", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c18", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c19", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c20", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c22", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c23", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c24", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c26", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c27", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c28", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c31", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c32", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c33", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c34", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c35", torch.empty([48, 96], dtype=torch.float32))
        self.register_buffer("c36", torch.empty([96], dtype=torch.float32))
        self.register_buffer("c37", torch.empty([96, 48], dtype=torch.float32))
        self.register_buffer("c38", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c40", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c41", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c42", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c43", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c44", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c45", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c47", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c48", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c49", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c51", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c52", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c53", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c55", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c56", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c57", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c60", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c61", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c62", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c63", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c64", torch.empty([48, 96], dtype=torch.float32))
        self.register_buffer("c65", torch.empty([96], dtype=torch.float32))
        self.register_buffer("c66", torch.empty([96, 48], dtype=torch.float32))
        self.register_buffer("c67", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c69", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c70", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c71", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c72", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c73", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c74", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c76", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c77", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c78", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c80", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c81", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c82", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c84", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c85", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c86", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c89", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c90", torch.empty([1], dtype=torch.int64))
        self.register_buffer("c91", torch.empty([48, 48], dtype=torch.float32))
        self.register_buffer("c92", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c93", torch.empty([48, 96], dtype=torch.float32))
        self.register_buffer("c94", torch.empty([96], dtype=torch.float32))
        self.register_buffer("c95", torch.empty([96, 48], dtype=torch.float32))
        self.register_buffer("c96", torch.empty([48], dtype=torch.float32))
        self.register_buffer("c97", torch.empty([1], dtype=torch.int64))
        self.conv2d1 = nn.Conv2d(3, 48, 8, stride=8)
        self.batchnorm2d1 = nn.BatchNorm2d(
            48, eps=9.999999747378752e-06, momentum=0.10000002384185791
        )
        self.flatten1 = nn.Flatten(3)
        self.softmax1 = nn.Softmax(-1)
        self.batchnorm2d2 = nn.BatchNorm2d(
            48, eps=9.999999747378752e-06, momentum=0.10000002384185791
        )
        self.relu1 = nn.ReLU()
        self.batchnorm2d3 = nn.BatchNorm2d(
            48, eps=9.999999747378752e-06, momentum=0.10000002384185791
        )
        self.flatten2 = nn.Flatten(3)
        self.softmax2 = nn.Softmax(-1)
        self.batchnorm2d4 = nn.BatchNorm2d(
            48, eps=9.999999747378752e-06, momentum=0.10000002384185791
        )
        self.relu2 = nn.ReLU()
        self.batchnorm2d5 = nn.BatchNorm2d(
            48, eps=9.999999747378752e-06, momentum=0.10000002384185791
        )
        self.flatten3 = nn.Flatten(3)
        self.softmax3 = nn.Softmax(-1)
        self.batchnorm2d6 = nn.BatchNorm2d(
            48, eps=9.999999747378752e-06, momentum=0.10000002384185791
        )
        self.relu3 = nn.ReLU()
        self.batchnorm2d7 = nn.BatchNorm2d(
            48, eps=9.999999747378752e-06, momentum=0.10000002384185791
        )
        self.linear1 = nn.Linear(48, 10)

    def forward(self, x0):
        x1 = torch.tensor(x0.shape, dtype=torch.int64)
        x2 = x1[0]
        x3 = self.conv2d1(x0)
        x4 = torch.tensor(x3.shape, dtype=torch.int64)
        x5 = x4[0:2]
        x6 = torch.cat([x5, self.c4])
        x7 = x3.reshape([int(x) for x in x6.tolist()])
        x8 = x7.permute((0, 2, 1))
        x9 = x2.unsqueeze(0)
        x10 = torch.cat([x9, self.c6, self.c7])
        x11 = torch.full(x10.tolist(), 0.0, dtype=torch.float32)
        x12 = x11 + self.c8
        x13 = torch.cat([x12, x8], dim=1)
        x14 = x13 + self.c9
        x15 = x14.permute((0, 2, 1))
        x16 = self.batchnorm2d1(x15.unsqueeze(2)).squeeze(2)
        x17 = x16.permute((0, 2, 1))
        x18 = torch.tensor(x17.shape, dtype=torch.int64)
        x19 = x18[0]
        x20 = x17 @ self.c11
        x21 = self.c12 + x20
        x22 = x17 @ self.c13
        x23 = self.c14 + x22
        x24 = x17 @ self.c15
        x25 = self.c16 + x24
        x26 = x19.unsqueeze(0)
        x27 = torch.cat([x26, self.c18, self.c19, self.c20])
        x28 = x19.unsqueeze(0)
        x29 = torch.cat([x28, self.c22, self.c23, self.c24])
        x30 = x19.unsqueeze(0)
        x31 = torch.cat([x30, self.c26, self.c27, self.c28])
        x32 = x21.reshape([int(x) for x in x27.tolist()])
        x33 = x32.permute((0, 2, 1, 3))
        x34 = x23.reshape([int(x) for x in x29.tolist()])
        x35 = x25.reshape([int(x) for x in x31.tolist()])
        x36 = x35.permute((0, 2, 1, 3))
        x37 = x34.permute((0, 2, 3, 1))
        x38 = x33 @ x37
        x39 = x38 * 0.25
        x40 = torch.tensor(x39.shape, dtype=torch.int64)
        x41 = self.flatten1(x39)
        x42 = self.softmax1(x41)
        x43 = x42.reshape([int(x) for x in x40.tolist()])
        x44 = x43 @ x36
        x45 = x44.permute((0, 2, 1, 3))
        x46 = x19.unsqueeze(0)
        x47 = torch.cat([x46, self.c31, self.c32])
        x48 = x45.reshape([int(x) for x in x47.tolist()])
        x49 = x48 @ self.c33
        x50 = self.c34 + x49
        x51 = x50 + x14
        x52 = x51.permute((0, 2, 1))
        x53 = self.batchnorm2d2(x52.unsqueeze(2)).squeeze(2)
        x54 = x53.permute((0, 2, 1))
        x55 = x54 @ self.c35
        x56 = self.c36 + x55
        x57 = self.relu1(x56)
        x58 = x57 @ self.c37
        x59 = self.c38 + x58
        x60 = x59 + x51
        x61 = x60.permute((0, 2, 1))
        x62 = self.batchnorm2d3(x61.unsqueeze(2)).squeeze(2)
        x63 = x62.permute((0, 2, 1))
        x64 = torch.tensor(x63.shape, dtype=torch.int64)
        x65 = x64[0]
        x66 = x63 @ self.c40
        x67 = self.c41 + x66
        x68 = x63 @ self.c42
        x69 = self.c43 + x68
        x70 = x63 @ self.c44
        x71 = self.c45 + x70
        x72 = x65.unsqueeze(0)
        x73 = torch.cat([x72, self.c47, self.c48, self.c49])
        x74 = x65.unsqueeze(0)
        x75 = torch.cat([x74, self.c51, self.c52, self.c53])
        x76 = x65.unsqueeze(0)
        x77 = torch.cat([x76, self.c55, self.c56, self.c57])
        x78 = x67.reshape([int(x) for x in x73.tolist()])
        x79 = x78.permute((0, 2, 1, 3))
        x80 = x69.reshape([int(x) for x in x75.tolist()])
        x81 = x71.reshape([int(x) for x in x77.tolist()])
        x82 = x81.permute((0, 2, 1, 3))
        x83 = x80.permute((0, 2, 3, 1))
        x84 = x79 @ x83
        x85 = x84 * 0.25
        x86 = torch.tensor(x85.shape, dtype=torch.int64)
        x87 = self.flatten2(x85)
        x88 = self.softmax2(x87)
        x89 = x88.reshape([int(x) for x in x86.tolist()])
        x90 = x89 @ x82
        x91 = x90.permute((0, 2, 1, 3))
        x92 = x65.unsqueeze(0)
        x93 = torch.cat([x92, self.c60, self.c61])
        x94 = x91.reshape([int(x) for x in x93.tolist()])
        x95 = x94 @ self.c62
        x96 = self.c63 + x95
        x97 = x96 + x60
        x98 = x97.permute((0, 2, 1))
        x99 = self.batchnorm2d4(x98.unsqueeze(2)).squeeze(2)
        x100 = x99.permute((0, 2, 1))
        x101 = x100 @ self.c64
        x102 = self.c65 + x101
        x103 = self.relu2(x102)
        x104 = x103 @ self.c66
        x105 = self.c67 + x104
        x106 = x105 + x97
        x107 = x106.permute((0, 2, 1))
        x108 = self.batchnorm2d5(x107.unsqueeze(2)).squeeze(2)
        x109 = x108.permute((0, 2, 1))
        x110 = torch.tensor(x109.shape, dtype=torch.int64)
        x111 = x110[0]
        x112 = x109 @ self.c69
        x113 = self.c70 + x112
        x114 = x109 @ self.c71
        x115 = self.c72 + x114
        x116 = x109 @ self.c73
        x117 = self.c74 + x116
        x118 = x111.unsqueeze(0)
        x119 = torch.cat([x118, self.c76, self.c77, self.c78])
        x120 = x111.unsqueeze(0)
        x121 = torch.cat([x120, self.c80, self.c81, self.c82])
        x122 = x111.unsqueeze(0)
        x123 = torch.cat([x122, self.c84, self.c85, self.c86])
        x124 = x113.reshape([int(x) for x in x119.tolist()])
        x125 = x124.permute((0, 2, 1, 3))
        x126 = x115.reshape([int(x) for x in x121.tolist()])
        x127 = x117.reshape([int(x) for x in x123.tolist()])
        x128 = x127.permute((0, 2, 1, 3))
        x129 = x126.permute((0, 2, 3, 1))
        x130 = x125 @ x129
        x131 = x130 * 0.25
        x132 = torch.tensor(x131.shape, dtype=torch.int64)
        x133 = self.flatten3(x131)
        x134 = self.softmax3(x133)
        x135 = x134.reshape([int(x) for x in x132.tolist()])
        x136 = x135 @ x128
        x137 = x136.permute((0, 2, 1, 3))
        x138 = x111.unsqueeze(0)
        x139 = torch.cat([x138, self.c89, self.c90])
        x140 = x137.reshape([int(x) for x in x139.tolist()])
        x141 = x140 @ self.c91
        x142 = self.c92 + x141
        x143 = x142 + x106
        x144 = x143.permute((0, 2, 1))
        x145 = self.batchnorm2d6(x144.unsqueeze(2)).squeeze(2)
        x146 = x145.permute((0, 2, 1))
        x147 = x146 @ self.c93
        x148 = self.c94 + x147
        x149 = self.relu3(x148)
        x150 = x149 @ self.c95
        x151 = self.c96 + x150
        x152 = x151 + x143
        x153 = torch.mean(x152, self.c97.tolist(), keepdim=False)
        x154 = self.batchnorm2d7(x153.unsqueeze(2).unsqueeze(3)).squeeze(2).squeeze(2)
        x155 = self.linear1(x154)
        return x155


```

## API Reference

### TorchONNX Class

Main API for ONNX to PyTorch conversion.

**Constructor:**

```python
TorchONNX(verbose: bool = False)
```

Parameters:
- `verbose`: Enable detailed logging during conversion (default: False)

**Methods:**

```python
convert(
    onnx_path: str,
    benchmark_name: str | None = None,
    target_py_path: str = "model.py",
    target_pth_path: str = "model.pth",
    vmap_mode: bool = True
) -> None
```

Converts ONNX model to PyTorch.

Parameters:
- `onnx_path`: Path to input ONNX model
- `benchmark_name`: Optional name for module (defaults to filename)
- `target_py_path`: Output .py file path for model structure (default: "model.py")
- `target_pth_path`: Output .pth file path for model parameters (default: "model.pth")
- `vmap_mode`: Generate torch.vmap-compatible code for vectorized evaluation (default: True)

```python
@staticmethod
preprocess(onnx_path: str, target_opset: int = 20) -> onnx.ModelProto
```

Preprocesses ONNX model (normalization, shape inference, validation).

Parameters:
- `onnx_path`: Path to input ONNX model
- `target_opset`: Target ONNX opset version (default: 20)

Returns:
- Normalized ONNX model

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
│   └── tests/
└── ...
vnncomp2024/
│   ├── benchmarks/
└── ...
```

Then run the test suite:

```bash
cd torchonnx/tests
python test_benchmarks.py
```

## Supported Operations

The tool implements most commonly used operations in feedforward neural networks and transformers:

### Layers (nn.Module)

- **Convolution**: Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d
- **Pooling**: MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- **Normalization**: BatchNorm2d (with automatic dimension handling)
- **Activation**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, ELU, GELU
- **Linear**: Linear
- **Dropout**: Dropout
- **Upsampling**: Upsample
- **Shape Operations**: Flatten

### Functions (F.* and torch.*)

- **Convolution**: F.conv, F.conv_transpose
- **Linear**: F.linear
- **Pooling**: F.interpolate
- **Padding**: F.pad
- **Concatenation**: torch.cat
- **Indexing**: torch.gather, scatter_nd
- **Reduction**: torch.mean, torch.sum, torch.min, torch.max, torch.argmax
- **Clipping**: torch.clamp
- **Conditional**: torch.where
- **Generation**: torch.full, torch.arange

### Operators (torch.*)

- **Arithmetic**: add (+), sub (-), mul (*), div (/), matmul (@), pow (pow), neg (neg)
- **Comparison**: equal (==)

### Tensor Operations

- **Shape**: reshape, permute, squeeze, unsqueeze, shape, expand, cast
- **Slicing**: slice, split
- **Math**: sign, cos, sin, floor

Transformer-based architectures are decomposed into basic operations and handled correctly.

## Testing & Validation

TorchONNX is extensively tested to ensure correctness and reliability. The test suite includes unit tests, integration tests, performance benchmarks, and VNNCOMP 2024 benchmark validation.

### Test Results Summary

**Latest Test Run (2026-01-03):**

| Test Suite | Passed | Skipped | Warnings | Time |
|------------|--------|---------|----------|------|
| Unit Tests | 853 | 8 | 5 | 2.53s |
| Benchmark Tests | 5 | 54 | 4 | 1.00s |
| **Total** | **858** | **62** | **9** | **3.53s** |

**Coverage:**
- Python 3.11 and 3.12
- All 6 pipeline stages (normalize, build, analyze, optimize, generate, simplify)
- End-to-end conversion tests with numerical accuracy validation
- Error handling for edge cases and invalid inputs
- Performance benchmarks on complex models

### Test Structure

```
tests/
├── test_units/test_torchonnx/              # Unit tests (853 tests)
│   ├── test_normalize.py                   # ONNX normalization and shape inference
│   ├── test_build.py                       # Structural IR construction
│   ├── test_analyze.py                     # Semantic IR and type mapping
│   ├── test_generate.py                    # Code generation and forward methods
│   ├── test_simplify.py                    # Code optimization and formatting
│   ├── test_pipeline.py                    # End-to-end conversion pipeline
│   ├── test_integration.py                 # Integration with PyTorch models
│   ├── test_error_handling.py              # Error handling and edge cases
│   ├── test_attr_validation.py             # Attribute validation rules
│   ├── test_conv_operations.py             # Convolution operation details
│   ├── test_operation_handlers.py          # Operation handler validation
│   ├── test_optimize.py                    # IR optimization passes
│   ├── test_remaining_gaps.py              # Code formatting edge cases
│   └── fixtures/                           # Test fixtures and utilities
└── test_benchmarks/                         # Benchmark tests (5 tests, 54 skipped)
    ├── test_performance.py                 # Performance benchmarks
    ├── test_vmap_mode.py                   # vmap-compatible code generation
    ├── test_torchonnx.py                   # Model conversion benchmarks
    ├── test_torchonnx_regression.py        # Regression baseline tests
    ├── test_vnncomp2024_benchmarks.py      # VNNCOMP 2024 benchmark validation
    ├── build_benchmarks.py                 # Benchmark data generation
    └── baselines/                          # Expected outputs for benchmarks
```

### Test Categories

**Unit Tests (853 tests):**
- **Normalize Stage:** 67 tests - ONNX model loading, preprocessing, shape inference
- **Build Stage:** 11 tests - Structural IR construction and validation
- **Analyze Stage:** 73 tests - Semantic analysis, type mapping, attribute extraction
- **Generate Stage:** 162 tests - Code generation, forward methods, state dict handling
- **Simplify Stage:** 47 tests - Code formatting, optimization, file headers
- **Pipeline Tests:** 54 tests - End-to-end conversion, numerical accuracy
- **Integration Tests:** 46 tests - Real PyTorch models, error handling
- **Operation Handlers:** 52 tests - Individual operation validation
- **Error Handling:** 58 tests - Invalid inputs, edge cases, type errors

**Benchmark Tests (5 tests):**
- **Performance:** 4 tests - normalize, build, format_code, model_creation
- **vmap Compatibility:** 1 test - vmap-compatible code generation validation

**Skipped Tests (62 tests):**
- Benchmark data tests (54 skipped): Requires external benchmark data via `build_benchmarks.py`
- Memory-intensive tests (1 skipped): Excluded from default test runs
- Complex model tests (3 skipped): Intentionally skipped for stability
- vmap mode tests (4 skipped): Requires full benchmark environment

### Running Tests

**Run all tests:**
```bash
pytest tests/
```

**Run unit tests only:**
```bash
pytest tests/test_units/test_torchonnx/ -v
```

**Run benchmark tests only:**
```bash
pytest tests/test_benchmarks/ -v
```

**Run with verbose output and short tracebacks:**
```bash
pytest tests/ -v --tb=short
```

**Run with coverage report:**
```bash
pytest tests/ --cov=src/torchonnx --cov-report=term-missing --cov-report=html
```

**Run specific test file:**
```bash
pytest tests/test_units/test_torchonnx/test_pipeline.py -v
```

**Run specific test class:**
```bash
pytest tests/test_units/test_torchonnx/test_pipeline.py::TestConvertAPI -v
```

**Run specific test:**
```bash
pytest tests/test_units/test_torchonnx/test_pipeline.py::TestConvertAPI::test_convert_numerical_accuracy -v
```

**Run performance benchmarks:**
```bash
pytest tests/test_benchmarks/test_performance.py -v
```

### Continuous Integration

Tests run automatically via GitHub Actions:
- **Schedule:** Daily at 8 AM UTC
- **Manual:** Workflow dispatch trigger available
- **Matrix:** Python 3.11 and 3.12 on ubuntu-latest
- **Coverage:** Codecov integration for coverage reporting
- **Linting:** Ruff (check and format)
- **Type Checking:** MyPy static analysis

See `.github/workflows/unit-tests.yml` for full CI/CD configuration.

### Test Quality Metrics

- **Pass Rate:** 99.3% (858 passed / 920 total tests)
- **Test Execution:** ~3.5 seconds for full suite
- **Code Coverage:** Comprehensive coverage of all pipeline stages
- **Error Handling:** 58 dedicated error handling tests
- **Numerical Accuracy:** Validated against ONNX Runtime and PyTorch
- **Type Safety:** Full type hint coverage with MyPy validation

## Related Projects

- **[ShapeONNX](https://github.com/ZhongkuiMa/shapeonnx)**: Advanced shape inference for ONNX models. SlimONNX uses ShapeONNX for shape-dependent optimizations.
- **[TorchVNNLIB](https://github.com/ZhongkuiMa/torchvnnlib)**: PyTorch library for neural network verification. Often used in conjunction with SlimONNX for model verification tasks. This convert the VNNLIB data files to `.pth` format for PyTorch or `.npz` format for NumPy.
- **[SlimONNX](https://github.com/ZhongkuiMa/slimonnx)**: ONNX model simplification tool that removes redundant operations and optimizes the graph before conversion.
- **[VNN-COMP](https://sites.google.com/view/vnn2024)**: International Verification of Neural Networks Competition. SlimONNX is tested on all VNN-COMP 2024 benchmarks.
- **[ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)**: Alternative ONNX optimization tool with different optimization strategies.

## Troubleshooting

### Model Conversion Fails

**Unsupported Operations:**
- Use [SlimONNX](https://github.com/ZhongkuiMa/slimonnx) to simplify model first
- Check the "Supported Operations" list above
- Open an issue for missing operations

**Shape Inference Errors:**
- Ensure input ONNX model is valid and complete
- Try using [ShapeONNX](https://github.com/ZhongkuiMa/shapeonnx) externally to infer shapes
- Check ONNX opset version compatibility

**Invalid ONNX Model:**
- Validate model using ONNX checker: `onnx.checker.check_model(model)`
- Ensure all inputs have defined shapes in the ONNX graph

### Generated Code Issues

**Import Errors:**
- Verify PyTorch 2.x is installed and compatible with your system
- Check that all dependencies listed in Installation section are installed
- Ensure correct Python path and virtual environment

**Numerical Differences:**
- Verify original ONNX model produces expected outputs in onnxruntime
- Check for floating-point precision differences (use allclose with tolerance)
- Inspect generated code for potential precision issues

**vmap Compatibility Issues:**
- Disable vmap mode if conversion fails: `converter.convert(..., vmap_mode=False)`
- Check for in-place operations or `.item()` calls in generated code
- Some operations may not be vmap-compatible; see vmap Mode section

### Performance Issues

**Slow Conversion:**
- Large models may take time to normalize and analyze
- Enable verbose mode to monitor progress: `TorchONNX(verbose=True)`
- Simplify model using SlimONNX first to reduce complexity

**Large Generated Files:**
- Many constants embedded in code; this is normal
- Use parameter quantization in ONNX before conversion if size is critical
- Generated .pth files contain only parameters, not code

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing procedures, code quality standards, and pull request guidelines.

## Changelog

### Version 2026.1.0 (2026-01-03)

**Initial public release with clean commit history**

- Complete ONNX to PyTorch conversion pipeline
- Support for 50+ ONNX operations
- vmap-compatible code generation mode
- Extensive testing on VNNCOMP 2024 benchmarks
- 853 passing tests with comprehensive coverage
- Clean public API with proper visibility controls
- Full CI/CD pipeline with GitHub Actions

## License

This project is licensed under the MIT License. See the LICENSE file for details.
