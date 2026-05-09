# TorchONNX Architecture

ONNX-to-PyTorch compiler: 6-stage pipeline converting `.onnx` models into standalone `.py` modules + `.pth` state dicts.

## Package Tree

```
src/torchonnx/
├── _torchonnx.py          Entry point (TorchONNX class); modify when: changing pipeline orchestration
├── presets.py             Benchmark presets; modify when: adding benchmark-specific behavior
├── normalize/             Stage 1: ONNX loading/validation/opset conversion/shape inference
│   ├── normalize.py       Core preprocessing logic
│   └── utils.py           ONNX utility helpers
├── build/                 Stage 2: Structural IR extraction (graph topology only)
│   ├── builder.py         Graph traversal, NodeIR/ModelIR construction
│   └── types.py           NodeIR, ModelIR frozen dataclasses
├── analyze/               Stage 3: Semantic IR (classify tensors, map ONNX ops to PyTorch types)
│   ├── builder.py         Semantic IR construction
│   ├── tensor_classifier.py  Classify tensors as params/constants/arguments
│   ├── types.py           Semantic IR types
│   └── type_mapping/      ONNX-to-PyTorch type/op mapping tables
│       ├── _layers.py     nn.Module mappings (Conv, BatchNorm, Linear...)
│       └── _operations.py Functional op mappings (reshape, cat, pad...)
├── optimize/              Stage 4: IR-level optimizations
│   └── optimizer.py       Optimization passes on semantic IR
├── generate/              Stage 5: Python code generation
│   ├── code_generator.py  Orchestrates __init__ + forward + state_dict generation
│   ├── _init_gen.py       Emits __init__ body (layer declarations)
│   ├── _forward_gen.py    Emits forward() body (operation calls)
│   ├── _state_dict_gen.py Extracts state dict from IR
│   ├── _templates.py      Code templates (imports, class skeleton)
│   ├── _utils.py          Code-gen helpers
│   └── _handlers/         Per-op code generators (NOT IR; generates Python strings)
│       ├── _registry.py   Handler registry (HANDLERS dict, get_handler)
│       ├── _layers.py     nn.Module handlers (Conv2d, Linear, BatchNorm2d...)
│       ├── _operations.py Functional handlers (reshape, cat, gather...)
│       └── _operators.py  Binary/unary operator handlers (add, mul, matmul...)
└── simplify/              Stage 6: Post-processing on generated code strings
    ├── _optimizer.py      Dead-code removal, unused buffer cleanup
    ├── _rules.py          Simplification rules
    ├── _line_optimizer.py Line-level optimizations
    ├── _formatter.py      Black-compatible formatting
    └── _decorations.py    File headers, metadata
```

## Modification Map

| Intent | Primary Modify | Follow-ups | Avoid | Constraints | Failure Signal |
|--------|---------------|------------|-------|-------------|----------------|
| Add ONNX op support | `generate/_handlers/` | `analyze/type_mapping/`, tests | `build/`, `normalize/` | Must add both mapping AND handler (enforced) | `ValueError: Unsupported ONNX operator` |
| Change IR structure | `build/types.py` | `analyze/builder.py`, `generate/` | `normalize/` | Dataclasses are frozen (enforced) | `TypeError` on construction |
| Fix generated code quality | `simplify/` | None | `generate/_handlers/` | String-level transforms only (observed) | Ruff lint failures on output |
| Change preprocessing | `normalize/normalize.py` | None | `build/`, `analyze/` | Must preserve ModelProto output (observed) | Stage 2 build failure |
| Modify code-gen structure | `generate/code_generator.py` | `_init_gen.py`, `_forward_gen.py` | `simplify/` | Template changes in `_templates.py` (observed) | Generated code won't import |

## Dependency Rules

| Rule | Source | Failure |
|------|--------|---------|
| Stages flow forward only (1->2->3->4->5->6) | (enforced) | Circular import |
| Absolute imports only | (enforced) | Ruff TID error |
| IR dataclasses are frozen | (enforced) | `FrozenInstanceError` |
| `generate/_handlers/` must not import from `simplify/` | (observed) | N/A |

## Common Mistakes

| Mistake | Detection Signal | Fix |
|---------|-----------------|-----|
| Adding handler without type mapping | `ValueError: Unsupported ONNX operator` at runtime | Add entry to `analyze/type_mapping/_layers.py` or `_operations.py` |
| Generated code uses `.item()` in vmap mode | `torch.vmap` raises `RuntimeError` | Use tensor indexing; check `vmap_mode` parameter in handler |
| Modifying IR dataclass fields without updating downstream | `AttributeError` in later stages | Grep all usages of the field across stages 3-6 |

## Conventions

- `__docformat__ = "restructuredtext"` and `__all__` in every module
- Handler functions return `list[str]` (lines of generated Python code)
- Type mapping returns PyTorch type strings (`"nn.Conv2d"`, `"torch.add"`, `"F.pad"`)
- Three op categories: layers (nn.Module), operations (F.*/torch.*), operators (binary/unary)

## Related Documents

- [README.md](README.md) -- usage, API, supported operations
- [CONTRIBUTING.md](CONTRIBUTING.md) -- setup, workflow, adding operations
- Root [ARCHITECTURE.md](../ARCHITECTURE.md) -- rover system architecture, submodule relationships
