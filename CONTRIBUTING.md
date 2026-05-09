# Contributing to TorchONNX

Shared conventions (imports, formatting, docstrings) are in the root [CONTRIBUTING.md](../CONTRIBUTING.md).
This file covers torchonnx-specific workflow only.

## Setup

```bash
cd torchonnx
pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
pre-commit run --all-files  # lint, format, type-check
pytest tests/ -v            # tests
```

## Adding a New ONNX Operation

Most PRs add support for a new ONNX op. Follow this pattern:

1. **Type mapping** -- add entry to `src/torchonnx/analyze/type_mapping/_layers.py` (if nn.Module) or `_operations.py` (if functional)
2. **Code generation handler** -- add handler function in `src/torchonnx/generate/_handlers/_layers.py`, `_operations.py`, or `_operators.py`
3. **Register handler** -- call `register_handler("OpName", handler_fn)` in the appropriate registration function
4. **Test** -- add unit test in `tests/test_units/test_torchonnx/`
5. **Verify** -- `pytest tests/ -v -k "new_op_name"`

## Constraints

| Rule | Details |
|------|---------|
| Absolute imports only | `from torchonnx.generate._handlers._registry import ...` (no relative) |
| Frozen dataclasses | All IR types use `frozen=True` |
| vmap compatibility | Generated code must avoid `.item()` and in-place ops when `vmap_mode=True` |
| Handler registry | Every supported op needs both type mapping AND generation handler |
| ReST docstrings | `__docformat__ = "restructuredtext"` and `__all__` in every module |
