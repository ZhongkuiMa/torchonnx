# TorchONNX

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/ZhongkuiMa/torchonnx/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/ZhongkuiMa/torchonnx/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/ZhongkuiMa/torchonnx/branch/main/graph/badge.svg)](https://codecov.io/gh/ZhongkuiMa/torchonnx)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

Compile ONNX models into standalone PyTorch code (`.py` module + `.pth` state dict) with zero ONNX dependencies at inference time. Generated code supports `torch.vmap` for batched verification.

## Installation

```bash
git clone https://github.com/ZhongkuiMa/torchonnx.git
cd torchonnx
pip install -e ".[dev]"
```

**Requirements:** Python 3.11+, PyTorch, onnx, onnxruntime, numpy

## Quick Start

```python
from torchonnx import TorchONNX

converter = TorchONNX(verbose=True)
converter.convert("model.onnx", target_py_path="model.py", target_pth_path="model.pth")
```

Load and run the generated model:

```python
import torch
from model import Model

model = Model()
model.load_state_dict(torch.load("model.pth"))
output = model(torch.randn(1, 3, 32, 32))
```

## Usage Guide

### vmap-compatible output

By default (`vmap_mode=True`), `TorchONNX` replaces in-place ops with functional equivalents and `.item()` calls with tensor-based indexing, making the generated code compatible with `torch.vmap`:

```python
model = Model()
model.load_state_dict(torch.load("model.pth"))
batch_outputs = torch.vmap(model)(torch.randn(100, 3, 32, 32))
```

Disable with `vmap_mode=False` if vmap compatibility is not needed.

### Preprocessing

Use `TorchONNX.preprocess()` to load, validate, and normalize an ONNX model without generating code:

```python
model_proto = TorchONNX.preprocess("model.onnx", target_opset=20)
```

### Benchmark presets

`BENCHMARKS_WITHOUT_BATCH_DIM` is a set of benchmark names that lack a batch dimension. Use `if_has_batch_dim(benchmark_name)` to check whether a model from a given benchmark includes one:

```python
from torchonnx import BENCHMARKS_WITHOUT_BATCH_DIM, if_has_batch_dim

has_batch = if_has_batch_dim("acasxu")
```

### API reference

| Symbol | Description |
|--------|-------------|
| `TorchONNX(verbose=False, use_shapeonnx=False)` | Create converter instance |
| `.convert(onnx_path, benchmark_name, target_py_path, target_pth_path, vmap_mode=True)` | Compile ONNX to PyTorch (`.py` + `.pth`) |
| `.preprocess(onnx_path, target_opset, infer_shapes, clear_docstrings)` | Load, validate, and normalize ONNX model (static method) |
| `BENCHMARKS_WITHOUT_BATCH_DIM` | Set of benchmark names without a batch dimension |
| `if_has_batch_dim(benchmark_name)` | Return whether the benchmark model includes a batch dimension |

## Compiler Pipeline

TorchONNX is a 6-stage compiler:

| Stage | Directory | Purpose |
|-------|-----------|---------|
| 1. Normalize | `normalize/` | Load, validate, convert opset, infer shapes |
| 2. Build | `build/` | Extract structural IR (`ModelIR`, `NodeIR`) from ONNX graph |
| 3. Analyze | `analyze/` | Classify tensors, map ONNX ops to PyTorch types |
| 4. Optimize | `optimize/` | IR-level optimizations |
| 5. Generate | `generate/` | Emit `__init__`, `forward()`, state dict, imports |
| 6. Simplify | `simplify/` | Remove unused buffers, default args, format code |

## Supported Operations

| Category | Operations |
|----------|------------|
| Layers | Conv1d/2d, ConvTranspose1d/2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, BatchNorm2d, Linear, Flatten, Dropout, Upsample |
| Activations | ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, ELU, GELU |
| Functions | F.pad, F.interpolate, torch.cat, torch.gather, torch.mean/sum/min/max/argmax, torch.clamp, torch.where, torch.full, torch.arange |
| Operators | add, sub, mul, div, matmul, pow, neg, equal |
| Tensor ops | reshape, permute, squeeze, unsqueeze, expand, slice, split, cast, sign, cos, sin, floor |

Transformer architectures are decomposed into these basic operations.

## Project Structure

```
torchonnx/
├── src/torchonnx/
│   ├── _torchonnx.py     # TorchONNX class (main API)
│   ├── presets.py        # Benchmark-specific settings
│   ├── normalize/        # Stage 1: ONNX normalization
│   ├── build/            # Stage 2: Structural IR
│   ├── analyze/          # Stage 3: Semantic IR + type mapping
│   ├── optimize/         # Stage 4: IR optimization
│   ├── generate/         # Stage 5: Code generation + handlers
│   └── simplify/         # Stage 6: Code optimization
└── tests/
    ├── test_units/       # Unit tests
    └── test_benchmarks/  # VNNCOMP 2024 benchmark validation
```

## Tests

```bash
pytest tests/ -v                              # all tests
pytest tests/test_units/test_torchonnx/ -v   # unit tests
pytest tests/test_benchmarks/ -v             # benchmark tests
```

Benchmark tests require [vnncomp2024_benchmarks](https://github.com/ChristopherBrix/vnncomp2024_benchmarks) cloned as a sibling directory.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).
