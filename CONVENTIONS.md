# Torchonnx Conventions

This file defines style and documentation conventions for the torchonnx package.
Use it as a **checklist** — when writing or reviewing code, check each item below
one by one.

---

## 1. Module Docstrings

Every `.py` file begins with a module docstring.

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 1.1 | **First line**: short summary of the module's purpose (one sentence) | ☐ |
| 1.2 | **Extended description** (optional): 1-2 paragraphs after a blank line, covering the module's role in the 5-stage pipeline | ☐ |
| 1.3 | **Format**: ReST plain text, no `:param:` or `:return:` tags at module level | ☐ |
| 1.4 | Always followed by `__docformat__ = "restructuredtext"` | ☐ |
| 1.5 | **No author, date, or version lines** — git history is authoritative | ☐ |
| 1.6 | **No non-ASCII characters** in docstrings — use ASCII equivalents | ☐ |

### Patterns

| File type | Style | Example |
|-----------|-------|---------|
| Entry point (`_torchonnx.py`) | One line describing the converter | `"""ONNX-to-PyTorch converter using a 5-stage pipeline."""` |
| Stage modules (`normalize/`, `build/`, `analyze/`) | One line | `"""Normalize ONNX model for conversion."""` |
| Handler module (`_handlers/_layers.py`) | One line | `"""Layer handler implementations for code generation."""` |
| Template module (`_templates.py`) | One line | `"""Code generation templates for PyTorch modules."""` |
| `__init__.py` | Summary of subpackage with listed exports | `"""Semantic analysis of ONNX model tensors."""` |

---

## 2. Class Docstrings

torchonnx uses a main orchestrator class with per-stage subpackages.

### 2.1 Structure

```python
class TorchONNX:
    """ONNX-to-PyTorch converter using a 5-stage pipeline.

    Converts ONNX models to standalone PyTorch nn.Module code by:
    normalize -> build IR -> analyze semantics -> generate code -> simplify -> export.
    """
```

### 2.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 2.1 | **First line**: describes what the class converts/computes, ends with period | ☐ |
| 2.2 | Pipeline stages documented in class docstring with `->` arrow notation | ☐ |
| 2.3 | Constructor parameters documented in class docstring with `:param name:` | ☐ |
| 2.4 | No docstring on `__init__` when class docstring covers constructor params | ☐ |
| 2.5 | Data classes (`types.py`) use class docstring to describe the data model | ☐ |

---

## 3. Method/Function Docstrings

### 3.1 Structure

```python
def convert(self, onnx_path: str, output_dir: str | None = None) -> str:
    """
    Convert ONNX model to PyTorch nn.Module code.

    Runs the full 5-stage pipeline: normalize, build IR, analyze,
    generate, simplify, optimize.

    :param onnx_path: Path to input ONNX model.
    :param output_dir: Directory for generated code (default: cwd).

    :return: Path to generated Python module file.
    :raises ValueError: If the ONNX model cannot be parsed.
    """
```

### 3.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 3.1 | **First line**: imperative mood, describes what the method computes, ends with period | ☐ |
| 3.2 | Use `:param name:`, `:return:`, and `:raises ExceptionType:` tags — no `:type:` tags | ☐ |
| 3.3 | `:param` descriptions: **capitalized, end with period**, describe semantics | ☐ |
| 3.4 | `:return` description: **capitalized, end with period** | ☐ |
| 3.5 | Private helpers may use a single-line docstring without `:param:` tags | ☐ |
| 3.6 | **No non-ASCII characters** in docstrings | ☐ |
| 3.7 | Stage-dispatch methods document which pipeline stage they implement | ☐ |

---

## 4. Inline Comments

| # | Rule | Pass/Fail |
|---|------|-----------|
| 4.1 | Comment **why**, not what — the code already says what | ☐ |
| 4.2 | Only add comments when the reasoning is non-obvious (code generation edge cases, IR transformations) | ☐ |
| 4.3 | Stage markers: `# Stage 1: Normalize ONNX model`, `# Stage 2: Build IR` for pipeline tracking in orchestrator methods | ☐ |
| 4.4 | No commented-out code — delete it | ☐ |
| 4.5 | `# TODO:` comments require an associated issue reference (enforced by ruff TD001) | ☐ |
| 4.6 | Generation comments: `# Generate forward method`, `# Generate __init__` in code generation modules | ☐ |

---

## 5. Naming Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 5.1 | **Classes**: PascalCase — `TorchONNX`, `CodeGenerator`, `IRBuilder` | ☐ |
| 5.2 | **Functions/methods**: snake_case — `convert`, `normalize_model`, `build_ir`, `generate_code` | ☐ |
| 5.3 | **Private functions**: `_` prefix — `_forward_gen`, `_init_gen`, `_state_dict_gen` | ☐ |
| 5.4 | **Private modules**: `_` prefix — `_torchonnx.py`, `_forward_gen.py`, `_templates.py`, `_utils.py`, `_handlers/` | ☐ |
| 5.5 | **Constants**: UPPER_CASE — `BENCHMARKS_WITHOUT_BATCH_DIM` | ☐ |
| 5.6 | **Handler functions**: `_handle_<op>` — `_handle_conv`, `_handle_gemm`, `_handle_relu`, `_handle_add` | ☐ |
| 5.7 | **Template files**: `_<concern>_gen.py` — `_forward_gen.py`, `_init_gen.py`, `_state_dict_gen.py` | ☐ |

---

## 6. Code Style

| # | Rule | Pass/Fail |
|---|------|-----------|
| 6.1 | **100-char line length** (enforced by ruff) | ☐ |
| 6.2 | **Double quotes** for strings and docstrings | ☐ |
| 6.3 | **Absolute imports only** — `from torchonnx.normalize import normalize_model` | ☐ |
| 6.4 | `__docformat__ = "restructuredtext"` after module docstring, before imports | ☐ |
| 6.5 | `__all__` in every source module, alphabetically sorted | ☐ |
| 6.6 | **Import order**: stdlib → third-party (`torch`, `onnx`) → first-party (`torchonnx.*`). Groups separated by blank lines. | ☐ |
| 6.7 | **Lazy imports inside methods**: `from torchonnx.normalize import ...` inside `convert()` body — permitted to avoid circular imports between pipeline stages | ☐ |
| 6.8 | `import torch` at module level; `from onnx import ModelProto, NodeProto` for type annotations | ☐ |
| 6.9 | **McCabe complexity ≤ 10** (enforced by ruff C90) | ☐ |
| 6.10 | **Only import what you use** — clean up unused imports (enforced by ruff F401) | ☐ |
| 6.11 | **No string annotations** when type is already imported | ☐ |

---

## 7. 6-Stage Pipeline Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 7.1 | Pipeline order is fixed: **normalize -> build IR -> analyze -> generate -> simplify (optimize + format + header)** | [x] |
| 7.2 | Each stage is a separate subpackage under `torchonnx/` | ☐ |
| 7.3 | Stages communicate via well-defined intermediate formats (ONNX ModelProto -> ModelIR -> SemanticModelIR -> generated code str -> formatted code str) | [x] |
| 7.4 | `TorchONNX.convert()` orchestrates all stages. The simplify stage has three entry points called sequentially: `optimize_generated_code()`, `format_code()`, `add_file_header()` | ☐ |
| 7.5 | New stages are added by: create subpackage → add to pipeline in `_torchonnx.py` → update this section | ☐ |
| 7.6 | Stage entry functions accept the previous stage's output and return the next stage's input | ☐ |

---

## 8. Code Generation Template Patterns

| # | Rule | Pass/Fail |
|---|------|-----------|
| 8.1 | Templates in `generate/_templates.py` use `str.format()` with `{placeholder}` syntax | ☐ |
| 8.2 | `_forward_gen.py` generates the `forward()` method body | ☐ |
| 8.3 | `_init_gen.py` generates the `__init__()` method with layer definitions | ☐ |
| 8.4 | `_state_dict_gen.py` builds `dict[str, Tensor]` mapping parameter names to initial values (in-memory state dict, not code generation) | ☐ |
| 8.5 | Generated code uses consistent indentation (4 spaces); `INDENT` constant defined in `_templates.py` | ☐ |
| 8.6 | Large helper function templates (DYNAMIC_SLICE, SCATTER_ND, EXPAND, and vmap variants) are defined as module-level constants in `code_generator.py`, not in `_templates.py` | ☐ |
| 8.6 | Generated variable names follow PyTorch conventions — `self.conv1`, `self.bn1`, `x`, `out` | ☐ |

---

## 9. Handler Registry Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 9.1 | `generate/_handlers/_registry.py` maps PyTorch layer type names to handler functions via `HANDLERS: dict[str, Handler]` | ☐ |
| 9.2 | Handlers are split by `OperatorClass`: `_layers.py` (LAYER → nn.Module), `_operations.py` (OPERATION → functional), `_operators.py` (OPERATOR → math) | ☐ |
| 9.3 | Handler type: `Handler = Callable[[SemanticLayerIR, dict[str, str]], str]` — takes layer IR and `layer_name_mapping`, returns code string | ☐ |
| 9.4 | New ONNX ops are added by: create `_handle_<op>` in appropriate category file → call `register_handler(name, handler)` in the corresponding `register_*_handlers()` | ☐ |
| 9.5 | Lazy registration: `_ensure_handlers_registered()` defers handler registration to first use, avoiding module-level side effects | ☐ |
| 9.6 | `ForwardGenContext` global singleton accessed via `get_forward_gen_context()` / `set_forward_gen_context()` — not passed as a parameter | ☐ |

---

## 10. Semantic IR Type Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 10.1 | `analyze/types.py` defines the typed IR: `VariableInfo` (inputs), `ParameterInfo` (weights/biases), `ConstantInfo` (constants), `ArgumentInfo` (op args), `OperatorClass` (LAYER/OPERATION/OPERATOR), `SemanticLayerIR`, `SemanticModelIR` | ☐ |
| 10.2 | `analyze/attr_extractor.py` extracts ONNX attributes into typed Python dicts via `extract_onnx_attrs()` | ☐ |
| 10.3 | `analyze/tensor_classifier.py` classifies tensors by role: `classify_inputs()` → `VariableInfo`, `classify_outputs()` → output specs | ☐ |
| 10.4 | `analyze/type_mapping/_layers.py` and `_operations.py` map ONNX op types to PyTorch equivalents with arg extraction dispatch dicts | ☐ |
| 10.5 | All IR types use `@dataclass(frozen=True)` without `slots=True` | ☐ |
| 10.6 | Code name convention: `VariableInfo.code_name` = `xN`, `ParameterInfo.code_name` = `pN`, `ConstantInfo.code_name` = `cN` | ☐ |

---

## 11. Simplify Stage Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 11.1 | `simplify/_optimizer.py` applies post-generation code optimizations via `optimize_generated_code()` | ☐ |
| 11.2 | `simplify/_rules.py` defines `LAYER_DEFAULTS`, `FUNCTION_DEFAULTS`, `POSITIONAL_ONLY_ARGS` — rule tables for removing redundant default arguments and converting to positional form | ☐ |
| 11.3 | `simplify/_formatter.py` applies consistent formatting via `format_code()` — normalizes blank lines, wraps to 88 chars (Black default for generated code, distinct from 100-char source limit) | ☐ |
| 11.4 | `simplify/_line_optimizer.py` optimizes individual lines via `optimize_line()` → `_optimize_layer_instantiation()` + `_optimize_function_call()` | ☐ |
| 11.5 | Simplification rules are idempotent — running simplify twice produces the same output | ☐ |
| 11.6 | `simplify/_decorations.py` adds file headers (copyright, metadata, `__all__`) via `add_file_header()` | ☐ |

---

## 12. Presets and Configuration

| # | Rule | Pass/Fail |
|---|------|-----------|
| 12.1 | `presets.py` defines benchmark model lists and utility functions | ☐ |
| 12.2 | Module-level tuples for model name constants — `BENCHMARKS_WITHOUT_BATCH_DIM` | ☐ |
| 12.3 | `if_has_batch_dim(model_name)` helper for batch dimension detection | ☐ |

---

## 13. Test Style

### 13.1 Directory Layout

```
tests/
├── test_benchmarks/           # integration tests (opt-in)
│   └── baselines/test/
└── test_units/
    └── test_torchonnx/
        ├── __init__.py
        └── test_<concern>.py
```

### 13.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 13.1 | **Test file naming**: `test_<concern>.py` — `test_convert.py`, `test_generate.py` | ☐ |
| 13.2 | `__init__.py` at leaf `test_torchonnx/` level (collision avoidance) | ☐ |
| 13.3 | **No pytest markers** except `@pytest.mark.parametrize` | ☐ |
| 13.4 | Test ONNX models built with `onnx.helper.make_model()` or loaded from test fixtures | ☐ |
| 13.5 | Integration tests for full pipeline go in `test_benchmarks/` | ☐ |
| 13.6 | Generated code tests: write to temp dir (`tmp_path` fixture), import dynamically, verify output | ☐ |
| 13.7 | Test module docstrings: 1-3 lines max summarizing what the file validates | ☐ |
| 13.8 | **Default test suite**: `pytest` runs `tests/test_units/` by default. Benchmark tests are opt-in | ☐ |
| 13.9 | **No `@pytest.mark.skip`** in committed code — use conditional early return with `[REVIEW]` comment | ☐ |

---

## 14. Enum Conventions

torchonnx uses `StrEnum` for operator classification (`OperatorClass` in `analyze/types.py`).

| # | Rule | Pass/Fail |
|---|------|-----------|
| 14.1 | **StrEnum for user-facing values**: Use `StrEnum` when values are exposed as strings. Use `IntEnum` with `@unique` for internal numeric enums | ☐ |
| 14.2 | **Placement**: Domain-specific enums in the owning subpackage (e.g., `analyze/types.py` for `OperatorClass`) | ☐ |
| 14.3 | **Class naming**: PascalCase with categorical suffix — `Mode`, `Type`, `Status`, `Strategy`. Never suffix with `Enum` | ☐ |
| 14.4 | **Member naming**: `UPPER_SNAKE_CASE`, 1-3 words. StrEnum values are lowercase member names: `LAYER = "layer"` | ☐ |
| 14.5 | **`@unique` decorator**: Required on IntEnum, optional on StrEnum (values are inherently unique) | ☐ |
| 14.6 | **Member docstrings**: Every enum member has a one-line ReST docstring after the value assignment | ☐ |

---

## 15. Constants Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 15.1 | **Naming**: `UPPER_SNAKE_CASE`, 2-4 words. Use prefixes (`DEFAULT_`, `MAX_`, `MIN_`) and suffixes for clarity | ☐ |
| 15.2 | **Scope levels**: Place at narrowest scope — function-level → file-level → subfolder → package-level. Promote when a second consumer appears | ☐ |
| 15.3 | **Extraction trigger**: Extract a literal when it appears 2+ times. Never duplicate a constant across files | ☐ |
| 15.4 | **When NOT to extract**: Self-documenting single-use values, test data, function defaults | ☐ |
| 15.5 | **Type annotations**: Annotate only when the type is not obvious from the literal | ☐ |
| 15.6 | **Frozen collections**: Use `frozenset` or `tuple` for constant collections — never mutable | ☐ |

---

## 16. Vmap Mode (Spec-Dimension Batching)

| # | Rule | Pass/Fail |
|---|------|-----------|
| 13.1 | `TorchONNX.convert()` accepts `vmap_mode: bool = False` parameter | ☐ |
| 13.2 | When `vmap_mode=True`, generated code uses vmap-compatible operations with an extra batch dimension | ☐ |
| 13.3 | Vmap-compatible helper templates defined as module-level constants in `code_generator.py`: `DYNAMIC_SLICE_VMAP_HELPER`, `SCATTER_ND_VMAP_HELPER`, `EXPAND_VMAP_HELPER` | ☐ |
| 13.4 | `_detect_static_slice_lengths()` in `code_generator.py` analyzes IR to distinguish static vs dynamic slice sizes for vmap compatibility | ☐ |
| 13.5 | Handler `slice_length_hints` mechanism provides per-layer slice size information to code generation | ☐ |

---

## 17. Cross-Cutting Patterns

| # | Rule | Pass/Fail |
|---|------|-----------|
| 17.1 | **Forward-gen context**: `_forward_gen_context` lives in `generate/_context.py` (re-exported through `_forward_gen.py` for back-compat) with `set_forward_gen_context()` / `get_forward_gen_context()` for sharing state between `code_generator.py` and handlers | [x] |
| 17.2 | **Helper lifecycle**: IR analyzed via `_get_helper_needs_from_ir()` to determine needed helpers; only needed helpers are generated via `_generate_helpers_from_context()` | [x] |
| 17.3 | **Fail-fast in generation**: `_generate_forward_body()` re-raises any handler exception as a `RuntimeError` so callers learn about unsupported ops at compile time instead of getting a module that ast-parses but crashes on the first forward pass | [x] |
| 17.4 | **`OperatorClass` enum**: `LAYER` (nn.Module), `OPERATION` (torch.nn.functional), `OPERATOR` (torch operators). Classification by `_classify_operator_class()` in `analyze/builder.py` | ☐ |
| 17.5 | **Stage subpackages may import from earlier stages**: `build/` imports from `normalize/` for utility functions; `analyze/` imports from `build/` for IR types. This is an accepted cross-stage coupling for shared data structures | ☐ |

---

## 18. Architecture Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 18.1 | **`__init__.py` facade pattern**: import from subpackages, re-export public classes via `__all__` | ☐ |
| 18.2 | **Root modules** (`_torchonnx.py`, `presets.py`) import from all stage subpackages | ☐ |
| 18.3 | **shapeonnx and slimonnx are optional dependencies** — guarded or used only in specific stages (normalize) | ☐ |
| 18.4 | **Stage subpackages may import from earlier stages** for shared data structures (e.g., `build/` imports from `normalize/`, `analyze/` imports from `build/`). This is accepted for IR type sharing; avoid importing stage entry points | ☐ |

---

## 19. Logging Conventions

Pipeline tool: use `logging` package with `_enable_verbose()` helper.

### Setup

```python
import logging

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
    def __init__(self, verbose: bool = False, ...):
        if verbose:
            _enable_verbose()
```

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 19.1 | **`_enable_verbose()` in `_torchonnx.py`** — single configuration point for the package | ☐ |
| 19.2 | **Direct `_logger.info(f"...")` calls** — no `isEnabledFor` guards, no `%`-formatting | ☐ |
| 19.3 | **f-strings for all log messages** — `_logger.info(f"  Normalize: opset 20 (0.0123s)")` | ☐ |
| 19.4 | **Output format**: first line `TorchONNX: converting <path>`, stage lines `  Stage: description (0.XXXXs)`, final line `  Total: 0.XXXXs` | ☐ |
| 19.5 | **Per-stage timing**: `t = time.perf_counter()` before each stage, elapsed after | ☐ |
| 19.6 | **`warnings.warn()` for recoverable errors** — never `logger.warning()`. Warnings independent of verbose flag | ☐ |
| 19.7 | **`raise ValueError/RuntimeError` for fatal errors** -- never `logger.error()` | [x] |
| 19.8 | **No `print()` for diagnostic output** in source code | [x] |

---

## 20. Known Numerical Limits

| # | Rule / observation | Pass/Fail |
|---|------|-----------|
| 20.1 | **300-node ML4ACOPF models exceed tier-3 tolerance.** Both `ml4acopf_2023/onnx/300_ieee_ml4acopf.onnx` and `ml4acopf_2024/onnx/300_ieee_ml4acopf.onnx` produce float32 outputs that diverge from `onnxruntime` by up to ``max_abs = 2.15e-02`` / ``max_rel = 5.58e+01`` even though the generated PyTorch module is structurally identical to the (passing) 14- and 118-node variants. The cause is float32 precision accumulation across the 168-node graph on outputs whose magnitude is near zero -- not a handler bug. A future float64 conversion path is the right fix; today these two are tracked as hard fails in `test_verify_model_against_original`. | [x] |
| 20.2 | **Tolerance tiers**: see `tests/test_benchmarks/test_torchonnx.py` -- `TOLERANCE_TIER1_ABS=1e-6`, `TOLERANCE_TIER2_REL=1e-5`, `TOLERANCE_TIER3_REL=1e-3`, `TOLERANCE_TIER3_ABS=1e-4`. Anything beyond tier 3 is either `TOLERANCE_MISMATCH` (xfail) or `NUMERICAL_MISMATCH` (hard fail). | [x] |
