"""Self-contained convert -> load -> numerical-verify over vnncomp2025 models.

The baseline 2024 suite (``test_torchonnx.py``) is npz/baseline driven and
only exercised the vnncomp2024 corpus, which missed conversion bugs that
the vnncomp2025 model variants expose (e.g. a generated module that inlines
Reshape shapes still exported orphan ``c0``/``c1`` constants, failing a
strict ``load_state_dict``). This module discovers the 2025 corpus (symlinked
by ``build_benchmarks.py``), converts the first model of each benchmark, then
loads the generated module -- catching ``load_state_dict`` failures -- and
checks its output matches ONNX Runtime on a random input.

One model per benchmark keeps the suite fast; oversized-input models
(VGG/YOLO) are skipped. Skips cleanly when the 2025 corpus is absent.
"""

__docformat__ = "restructuredtext"

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from tests.test_benchmarks.benchmark_utils import find_benchmarks, find_models
from tests.test_benchmarks.test_torchonnx import _run_onnx_model, _run_pytorch_module
from torchonnx import TorchONNX

# Skip models whose flat input exceeds this many elements (VGG/YOLO etc.);
# they convert slowly and are not needed to exercise conversion correctness.
_MAX_INPUT_ELEMENTS = 200_000

# Skip models whose ONNX file is larger than this; their weight tensors make
# conversion slow (nn4sys ~100MB, VGG ~550MB) without adding coverage value.
_MAX_MODEL_BYTES = 30 * 1024 * 1024

# Benchmarks torchonnx cannot yet convert (unsupported ops / graph shapes).
# Documented skips, not silent -- revisit as torchonnx gains coverage.
_SKIP_BENCHMARKS: frozenset[str] = frozenset()


def _discover_2025_models() -> list[str | None]:
    """Collect the first ONNX model of each vnncomp2025 benchmark."""
    corpus = Path(__file__).parent / "vnncomp2025_benchmarks"
    if not corpus.exists():
        return [None]
    benchmarks = [b for b in find_benchmarks(str(corpus)) if b.name not in _SKIP_BENCHMARKS]
    models = find_models(benchmarks, max_per_benchmark=1)
    return [str(m) for m in models] or [None]


def _resolved_input_shape(onnx_path: Path) -> tuple[int, ...]:
    """Return the real ONNX input shape with dynamic dims resolved to 1."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    raw = session.get_inputs()[0].shape
    return tuple(d if isinstance(d, int) and d > 0 else 1 for d in raw)


@pytest.mark.parametrize("model_path", _discover_2025_models())
def test_convert_load_verify_2025(model_path: str | None) -> None:
    """Convert a 2025 model, load its state dict, and match ONNX Runtime."""
    if model_path is None:
        pytest.skip("vnncomp2025 benchmarks not available (run build_benchmarks.py)")
    assert model_path is not None

    onnx_path = Path(model_path)
    if onnx_path.stat().st_size > _MAX_MODEL_BYTES:
        pytest.skip(f"model too large to convert quickly: {onnx_path.stat().st_size} bytes")
    shape = _resolved_input_shape(onnx_path)
    if int(np.prod(shape)) > _MAX_INPUT_ELEMENTS:
        pytest.skip(f"input too large to convert quickly: {shape}")

    rng = np.random.default_rng(0)
    sample = rng.standard_normal(shape).astype(np.float32)

    out_dir = Path(tempfile.mkdtemp(prefix="torchonnx_2025_"))
    py_path = out_dir / f"{onnx_path.stem}.py"
    pth_path = out_dir / f"{onnx_path.stem}.pth"
    TorchONNX().convert(str(onnx_path), target_py_path=str(py_path), target_pth_path=str(pth_path))

    # _run_pytorch_module performs load_state_dict -- this is where an orphan
    # constant key (c0/c1) would raise before the numerical check.
    onnx_out = _run_onnx_model(str(onnx_path), sample)
    torch_out = _run_pytorch_module(str(py_path), str(pth_path), sample)

    onnx_first = next(iter(onnx_out.values()))
    torch_first = next(iter(torch_out.values()))
    np.testing.assert_allclose(
        np.asarray(torch_first).reshape(-1),
        np.asarray(onnx_first).reshape(-1),
        rtol=1e-3,
        atol=1e-4,
    )
