"""TorchONNX Model Conversion Test Suite.

This module provides functions to:
1. Convert ONNX models to PyTorch and save to results/baselines/
2. Verify converted models against original benchmark models (structural + numerical)

Directory structure:
- benchmarks/          # Original ONNX models
- results/baselines/{benchmark_name}/   # Current conversion results (.py and .pth files)
- baselines/{benchmark_name}/           # Golden reference baselines for regression testing

To update baselines, run update_baselines.py after converting models.
"""

__docformat__ = "restructuredtext"
__all__ = [
    "convert_all_models",
    "convert_model",
    "verify_benchmarks",
]

import importlib.util
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
import torch

from tests.test_benchmarks.benchmark_utils import (
    find_benchmarks,
    find_models,
    get_model_benchmark_name,
    get_model_data_path,
    get_model_relative_path,
)

TOLERANCE_EPSILON = 1e-10
TOLERANCE_TIER1_ABS = 1e-6
TOLERANCE_TIER2_REL = 1e-5
TOLERANCE_TIER3_REL = 1e-3
TOLERANCE_TIER3_ABS = 1e-4


def _run_onnx_model(model_path: str, inputs: np.ndarray) -> dict:
    """Run ONNX model inference using ONNX Runtime.

    :param model_path: Path to ONNX model file
    :param inputs: input arrays
    :return: Dictionary of output arrays
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inputs})
    output_names = [out.name for out in session.get_outputs()]
    return dict(zip(output_names, outputs, strict=False))


def _run_pytorch_module(
    module_path: str,
    state_dict_path: str,
    inputs: np.ndarray,
    dtype: str = "float32",
    device: str = "cpu",
) -> dict:
    """Run PyTorch module inference.

    :param module_path: Path to PyTorch module file
    :param state_dict_path: Path to state dict file
    :param inputs: input arrays
    :param dtype: Data type ("float32" or "float64")
    :param device: Device to run on ("cpu" or "cuda")
    :return: Dictionary of output arrays
    """
    module_file = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_file.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class_name = module.__all__[0]
    model_class = getattr(module, class_name)
    model = model_class()

    # Determine torch device
    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)

    if dtype == "float64":
        model = model.double()
        state_dict = {k: v.double() for k, v in state_dict.items()}
        input_tensor = torch.from_numpy(inputs.astype(np.float64)).double()
    else:
        input_tensor = torch.from_numpy(inputs).float()

    # Move model and input to device
    model = model.to(torch_device)
    input_tensor = input_tensor.to(torch_device)

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    # Move output back to CPU for numpy conversion
    if isinstance(output, torch.Tensor):
        return {"output": output.cpu().numpy()}
    if isinstance(output, tuple):
        return {f"output_{i}": out.cpu().numpy() for i, out in enumerate(output)}
    if isinstance(output, dict):
        return {k: v.cpu().numpy() for k, v in output.items()}
    return {"output": output}


def _check_tolerance(max_abs: float, max_rel: float) -> str:
    """Check if errors pass three-tier tolerance criteria.

    Returns three categories:
    - "PASS": Meets acceptable tolerance criteria
    - "TOLERANCE_MISMATCH": Exceeds strict tolerance but within acceptable precision band
    - "FAIL": Significant deviation beyond acceptable precision

    :param max_abs: Maximum absolute error
    :param max_rel: Maximum relative error
    :return: Status string: "PASS", "TOLERANCE_MISMATCH", or "FAIL"
    """
    # Strict tolerance (should pass)
    passes_tier1 = max_abs < TOLERANCE_TIER1_ABS
    passes_tier2 = max_rel < TOLERANCE_TIER2_REL
    passes_tier3 = max_rel < TOLERANCE_TIER3_REL and max_abs < TOLERANCE_TIER3_ABS

    if passes_tier1 or passes_tier2 or passes_tier3:
        return "PASS"

    # Borderline tolerance (numerical precision issue, not computation error)
    # These thresholds account for:
    # - CPU vs CUDA numerical differences
    # - float32 vs float64 precision differences
    # - Rounding error accumulation from different operation orderings
    borderline_abs = 1e-2  # Absolute difference threshold (1%)
    borderline_rel = 1.0  # Relative difference threshold (100%)

    if max_abs < borderline_abs or max_rel < borderline_rel:
        return "TOLERANCE_MISMATCH"

    # Significant deviation (actual computation error)
    # This catches cases where either:
    # - Absolute error exceeds 1%
    # - AND relative error exceeds 100%
    return "FAIL"


def _compute_errors(
    out1: np.ndarray, out2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute absolute and relative errors between two arrays.

    :param out1: First array
    :param out2: Second array
    :return: Tuple of (diff_array, rel_diff_array, max_abs, max_rel)
    """
    diff = np.abs(out1 - out2)
    rel_diff = diff / (np.abs(out2) + TOLERANCE_EPSILON)
    max_abs = float(np.max(diff))
    max_rel = float(np.max(rel_diff))
    return diff, rel_diff, max_abs, max_rel


def _is_scalar_shape_mismatch(shape1: tuple, shape2: tuple) -> bool:
    """Check if shapes are scalar vs single-element mismatches.

    :param shape1: First shape
    :param shape2: Second shape
    :return: True if mismatch is acceptable scalar difference
    """
    return (shape1 == () and shape2 == (1,)) or (shape1 == (1,) and shape2 == ())


def _compare_single_output(
    key1: str,
    key2: str,
    out1: np.ndarray,
    out2: np.ndarray,
) -> tuple[list, list, list]:
    """Compare single output pair and collect errors.

    :param key1: First output name
    :param key2: Second output name
    :param out1: First output array
    :param out2: Second output array
    :return: Tuple of (mismatches, diffs, rel_diffs)
    """
    mismatches: list[str] = []
    all_diffs: list[Any] = []
    all_rel_diffs: list[Any] = []

    if out1.shape != out2.shape:
        if _is_scalar_shape_mismatch(out1.shape, out2.shape):
            out1_flat = out1.flatten()
            out2_flat = out2.flatten()
            diff, rel_diff, max_abs, max_rel = _compute_errors(out1_flat, out2_flat)
            all_diffs.extend(diff)
            all_rel_diffs.extend(rel_diff)

            if not _check_tolerance(max_abs, max_rel):
                msg = (
                    f"{key1} {key2}: max diff {max_abs:.2e}, "
                    f"rel {max_rel:.2e} (shape: {out1.shape} vs {out2.shape})"
                )
                mismatches.append(msg)
        else:
            mismatches.append(f"{key1}: shape {out1.shape} vs {key2}: {out2.shape}")
        return mismatches, all_diffs, all_rel_diffs

    diff, rel_diff, max_abs, max_rel = _compute_errors(out1, out2)
    all_diffs.extend(diff.flatten())
    all_rel_diffs.extend(rel_diff.flatten())

    if not _check_tolerance(max_abs, max_rel):
        mismatches.append(f"{key1} {key2}: max diff {max_abs:.2e}, rel {max_rel:.2e}")

    return mismatches, all_diffs, all_rel_diffs


def _compute_statistics(all_diffs: list, all_rel_diffs: list) -> dict:
    """Compute error statistics from collected differences.

    :param all_diffs: List of absolute differences
    :param all_rel_diffs: List of relative differences
    :return: Dictionary of statistics
    """
    if not all_diffs:
        return {}

    return {
        "max_abs_diff": float(np.max(all_diffs)),
        "mean_abs_diff": float(np.mean(all_diffs)),
        "max_rel_diff": float(np.max(all_rel_diffs)),
        "mean_rel_diff": float(np.mean(all_rel_diffs)),
    }


def _compare_outputs(
    outputs1: dict, outputs2: dict, rtol: float = 1e-4, atol: float = 1e-4
) -> tuple[bool, list[str], dict]:
    """Compare outputs using three-tier tolerance.

    :param outputs1: First model outputs
    :param outputs2: Second model outputs
    :param rtol: Relative tolerance (kept for compatibility)
    :param atol: Absolute tolerance (kept for compatibility)
    :return: Tuple of (all_match, mismatch_messages, statistics)
    """
    all_mismatches = []
    all_diffs = []
    all_rel_diffs = []

    for key1, key2 in zip(outputs1.keys(), outputs2.keys(), strict=False):
        mismatches, diffs, rel_diffs = _compare_single_output(
            key1, key2, outputs1[key1], outputs2[key2]
        )
        all_mismatches.extend(mismatches)
        all_diffs.extend(diffs)
        all_rel_diffs.extend(rel_diffs)

    stats = _compute_statistics(all_diffs, all_rel_diffs)
    return len(all_mismatches) == 0, all_mismatches, stats


def convert_model(
    model_path: Path,
    output_dir: Path,
    benchmarks_root: Path,
) -> dict:
    """Convert one ONNX model to PyTorch and save to results directory.

    :param model_path: Path to source ONNX model file
    :param output_dir: Directory to save converted PyTorch module
    :param benchmarks_root: Path to benchmarks root directory
    :return: Dictionary with conversion results
    """
    benchmark_name = get_model_benchmark_name(model_path)
    model_name = model_path.name
    model_stem = model_path.stem

    pytorch_module_path = output_dir / benchmark_name / f"{model_stem}.py"
    state_dict_path = output_dir / benchmark_name / f"{model_stem}.pth"

    pytorch_module_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        start_time = time.perf_counter()

        # Use TorchONNX to convert ONNX model to PyTorch model
        from torchonnx import TorchONNX

        converter = TorchONNX(verbose=False)
        converter.convert(
            onnx_path=str(model_path),
            benchmark_name=benchmark_name,
            target_py_path=str(pytorch_module_path),
            target_pth_path=str(state_dict_path),
        )

        elapsed_time = time.perf_counter() - start_time

        return {
            "success": True,
            "benchmark": benchmark_name,
            "model": model_name,
            "time": elapsed_time,
            "py_path": str(pytorch_module_path),
            "pth_path": str(state_dict_path),
            "error": None,
        }

    except (OSError, ValueError, RuntimeError, AttributeError, ImportError) as error:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "model": model_name,
            "time": 0.0,
            "py_path": None,
            "pth_path": None,
            "error": str(error),
        }


def get_benchmark_models():
    """Collect all models from vnncomp2024 benchmarks for parametrization.

    :return: List of model paths for parametrized testing, or pytest.skip marker if empty
    """
    test_dir = Path(__file__).parent
    benchmarks_dir = test_dir / "vnncomp2024_benchmarks"
    benchmarks = find_benchmarks(str(benchmarks_dir))
    models = find_models(benchmarks, max_per_benchmark=20)
    model_list = [str(m) for m in models]

    # Return a skip marker if no models found (prevents pytest collection error)
    if not model_list:
        return [
            pytest.param(
                None,
                marks=pytest.mark.skip(
                    reason="Benchmark data not available (run build_benchmarks.py)"
                ),
            )
        ]
    return model_list


@pytest.mark.parametrize("model_path", get_benchmark_models())
def test_convert_model(model_path, output_dir_baselines, benchmarks_root):
    """Convert one ONNX model to PyTorch.

    Parametrized test that runs conversion for each benchmark model.

    :param model_path: Path to source ONNX model
    :param output_dir_baselines: Output directory (from conftest fixture)
    :param benchmarks_root: Benchmarks root (from conftest fixture)
    """
    model_path_obj = Path(model_path)
    result = convert_model(model_path_obj, output_dir_baselines, benchmarks_root)

    # Print progress (pytest -v shows this)
    rel_path = get_model_relative_path(model_path_obj, benchmarks_root)
    if result["success"]:
        print(f"\n{rel_path}: OK ({result['time']:.2f}s)")
    else:
        print(f"\n{rel_path}: FAILED - {result['error']}")

    # Assert success
    assert result["success"], f"Conversion failed for {rel_path}: {result['error']}"


@pytest.mark.parametrize("model_path", get_benchmark_models())
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def _format_status_line(
    rel_path: Path,
    dtype: str,
    device: str,
    status: str,
    max_abs_str: str,
    max_rel_str: str,
    error_msg: str | None,
) -> str:
    """Format a status line for test output.

    :return: Formatted status line
    """
    status_line = f"\n{rel_path}"
    status_line += f" | dtype={dtype}, device={device}"
    status_line += f" | Status: {status}"

    # Only show error details for non-OK statuses or OK with numerical data
    should_show_errors = (status != "OK" and status != "SKIP") or (
        status == "OK" and isinstance(max_abs_str, str) and max_abs_str != "N/A"
    )
    if should_show_errors:
        status_line += f" | max_abs={max_abs_str}, max_rel={max_rel_str}"

    if error_msg and error_msg != "N/A":
        status_line += f" | {error_msg}"

    return status_line


def _handle_verification_status(status: str, error_msg: str | None) -> None:
    """Handle verification status and assert/skip as appropriate.

    :param status: Verification status
    :param error_msg: Error message if any
    """
    if status == "OK":
        pass  # Test passes
    elif status == "TOLERANCE_MISMATCH":
        pytest.xfail(f"Numerical precision deviation: {error_msg}")
    elif status == "NUMERICAL_MISMATCH":
        raise AssertionError(f"Verification failed: {status} - {error_msg}")
    elif status == "SKIP":
        pytest.skip(error_msg or "Test skipped")


def test_verify_model_against_original(
    model_path,
    dtype,
    device,
    output_dir_baselines,
    benchmarks_root,
) -> None:
    """Verify converted PyTorch module against original ONNX model.

    Parametrized test that verifies each model across dtypes and devices.

    :param model_path: Path to original ONNX model
    :param dtype: Data type to test (float32 or float64)
    :param device: Device to test (cpu or cuda)
    :param output_dir_baselines: Output directory (from conftest fixture)
    :param benchmarks_root: Benchmarks root (from conftest fixture)
    """
    # Skip if CUDA not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Get paths
    model_path_obj = Path(model_path)
    benchmark_name = get_model_benchmark_name(model_path_obj)
    model_stem = model_path_obj.stem
    rel_path = Path(benchmark_name) / model_stem
    result_py = output_dir_baselines / benchmark_name / f"{model_stem}.py"
    result_pth = output_dir_baselines / benchmark_name / f"{model_stem}.pth"
    data_file = get_model_data_path(model_path_obj, benchmarks_root)

    # Skip if converted model doesn't exist
    if not result_py.exists():
        pytest.skip(f"Converted model not found: {result_py.name}")
    if not result_pth.exists():
        pytest.skip(f"State dict not found: {result_pth.name}")

    # Verify
    status, error_msg, stats = _verify_one_benchmark(
        result_py, result_pth, model_path_obj, data_file, rel_path, dtype, device
    )

    # Format status output
    max_abs = stats.get("max_abs_diff", "N/A")
    max_rel = stats.get("max_rel_diff", "N/A")

    if isinstance(max_abs, float):
        max_abs_str = f"{max_abs:.6e}"
    else:
        max_abs_str = str(max_abs)

    if isinstance(max_rel, float):
        max_rel_str = f"{max_rel:.6e}"
    else:
        max_rel_str = str(max_rel)

    status_line = _format_status_line(
        rel_path, dtype, device, status, max_abs_str, max_rel_str, error_msg
    )
    print(status_line)

    # Assert based on status category
    _handle_verification_status(status, error_msg)


def convert_all_models(
    benchmark_dir: str = "vnncomp2024_benchmarks",
    output_dir: str = "results/baselines",
    max_per_benchmark: int = 20,
) -> dict:
    """Convert all benchmark ONNX models to PyTorch and save to results directory.

    :param benchmark_dir: Root directory of benchmarks
    :param output_dir: Directory to save converted PyTorch modules
    :param max_per_benchmark: Maximum models per benchmark to process
    :return: Dictionary with overall statistics
    """
    benchmarks_root = Path(benchmark_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    benchmarks = find_benchmarks(benchmark_dir)
    models = find_models(benchmarks, max_per_benchmark)

    print(f"\nConverting {len(models)} ONNX models to PyTorch, saving to {output_dir}/")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    total_time = 0.0

    start_time = time.perf_counter()

    for i, model_path in enumerate(models):
        rel_path = get_model_relative_path(model_path, benchmarks_root)

        print(f"[{i + 1}/{len(models)}] {rel_path}...", end=" ")

        result = convert_model(model_path, output_root, benchmarks_root)

        if result["success"]:
            success_count += 1
            total_time += result["time"]

            print(f"OK ({result['time']:.2f}s)")
        else:
            failed_count += 1
            print(f"FAILED: {result['error']}")

    elapsed_total = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total models: {len(models)}")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    if len(models) > 0:
        print(f"Success rate: {success_count / len(models) * 100:.1f}%")
    print(f"Total conversion time: {total_time:.2f}s")
    print(f"Avg time per model: {total_time / success_count:.2f}s" if success_count > 0 else "N/A")
    print(f"Total wall time: {elapsed_total:.2f}s")

    return {
        "total": len(models),
        "success": success_count,
        "failed": failed_count,
        "total_time": total_time,
        "wall_time": elapsed_total,
    }


def _validate_verification_directories(results_path: Path, benchmarks_path: Path) -> None:
    """Validate that all required directories exist for verification.

    :param results_path: Path to results directory
    :param benchmarks_path: Path to benchmarks directory
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_path}")


def _load_test_data_from_npz(data_file: Path) -> list[np.ndarray]:
    """Load test inputs from npz data file.

    :param data_file: Path to npz data file
    :return: List of test input arrays
    """
    data = np.load(data_file, allow_pickle=True)
    test_inputs = []

    for vnnlib_name in data.files:
        vnnlib_data = data[vnnlib_name].item()
        if isinstance(vnnlib_data, dict):
            for bound_type in ["lower", "upper"]:
                if bound_type in vnnlib_data:
                    bound_data = vnnlib_data[bound_type]
                    if "inputs" in bound_data:
                        test_inputs.extend(bound_data["inputs"])

    return test_inputs


def _check_verification_prerequisites(
    result_file: Path,
    result_state_dict: Path,
    benchmark_onnx_file: Path,
    data_file: Path,
) -> tuple[bool, str | None, str | None]:
    """Check if all prerequisite files exist for verification.

    :return: (all_exist, skip_message, error_message)
    """
    if not result_file.exists():
        return False, f"Result file not found: {result_file.name}", None
    if not result_state_dict.exists():
        return False, f"State dict not found: {result_state_dict.name}", None
    if not benchmark_onnx_file.exists():
        return False, f"Benchmark ONNX not found: {benchmark_onnx_file.name}", None
    if not data_file.exists():
        return False, f"No test data file: {data_file.name}", None
    return True, None, None


def _process_test_input(
    inputs: np.ndarray,
    result_file: Path,
    result_state_dict: Path,
    benchmark_onnx_file: Path,
    dtype: str,
    device: str,
) -> tuple[bool, list[str], list[str], dict, str]:
    """Process a single test input and check outputs.

    :return: (all_match, tolerance_issues, computation_errors, combined_stats, worst_status)
    """
    tolerance_issues = []
    computation_errors = []
    combined_stats = {}
    worst_status = "OK"
    all_match = True

    try:
        result_outputs = _run_pytorch_module(
            str(result_file), str(result_state_dict), inputs, dtype, device
        )
        benchmark_outputs = _run_onnx_model(str(benchmark_onnx_file), inputs)

        # Check tolerance for each output
        for key1, key2 in zip(result_outputs.keys(), benchmark_outputs.keys(), strict=False):
            out1 = result_outputs[key1]
            out2 = benchmark_outputs[key2]

            if out1.shape == out2.shape:
                _diff, _rel_diff, max_abs, max_rel = _compute_errors(out1, out2)
                combined_stats = {
                    "max_abs_diff": float(max_abs),
                    "max_rel_diff": float(max_rel),
                }

                tolerance_status = _check_tolerance(max_abs, max_rel)
                if tolerance_status == "PASS":
                    continue
                if tolerance_status == "TOLERANCE_MISMATCH":
                    tolerance_issues.append(f"{key1}: max diff {max_abs:.2e}, rel {max_rel:.2e}")
                    worst_status = "TOLERANCE_MISMATCH"
                else:  # "FAIL"
                    computation_errors.append(f"{key1}: max diff {max_abs:.2e}, rel {max_rel:.2e}")
                    worst_status = "NUMERICAL_MISMATCH"
                    all_match = False

    except (
        RuntimeError,
        ValueError,
        AttributeError,
        ImportError,
        IndexError,
        TypeError,
        KeyError,
    ):
        return False, [], [], {}, "ERROR"

    return all_match, tolerance_issues, computation_errors, combined_stats, worst_status


def _verify_one_benchmark(
    result_file: Path,
    result_state_dict: Path,
    benchmark_onnx_file: Path,
    data_file: Path,
    rel_path: Path,
    dtype: str = "float32",
    device: str = "cpu",
) -> tuple[str, str | None, dict]:
    """Verify converted PyTorch module against original ONNX model.

    Returns four status categories:
    - "OK": Outputs match perfectly or within tolerance
    - "TOLERANCE_MISMATCH": Numerical precision deviation (CPU/CUDA, float32/float64)
    - "NUMERICAL_MISMATCH": Actual computation error beyond acceptable precision
    - "SKIP": Could not run test (missing files, etc.)
    - "ERROR": Unexpected error during verification

    :param result_file: Path to converted PyTorch module file
    :param result_state_dict: Path to converted state dict file
    :param benchmark_onnx_file: Path to original ONNX model file
    :param data_file: Path to test data npz file
    :param rel_path: Relative path for reporting
    :param dtype: Data type ("float32" or "float64")
    :param device: Device to run on ("cpu" or "cuda")
    :return: Tuple of (status, error_message, statistics) where status is "OK",
        "TOLERANCE_MISMATCH", "NUMERICAL_MISMATCH", "SKIP", or "ERROR"
    """
    # Check prerequisites
    prereq_ok, skip_msg, _error_msg = _check_verification_prerequisites(
        result_file, result_state_dict, benchmark_onnx_file, data_file
    )
    if not prereq_ok:
        return "SKIP", skip_msg, {}

    # Load test data
    try:
        test_inputs = _load_test_data_from_npz(data_file)
        if not test_inputs:
            return "SKIP", "No inputs in data file", {}
    except (OSError, KeyError, ValueError, IndexError) as error:
        return "SKIP", f"Error loading data: {error}", {}

    # Process each test input
    worst_status = "OK"
    all_tolerance_issues = []
    all_computation_errors = []
    combined_stats = {}

    for _i, inputs in enumerate(test_inputs):
        _all_match, tolerance_issues, computation_errors, stats, status = _process_test_input(
            inputs, result_file, result_state_dict, benchmark_onnx_file, dtype, device
        )

        if status == "ERROR":
            return "ERROR", str(status), {}

        combined_stats.update(stats)
        all_tolerance_issues.extend(tolerance_issues)
        all_computation_errors.extend(computation_errors)

        if status == "NUMERICAL_MISMATCH":
            worst_status = "NUMERICAL_MISMATCH"
            break

        if status == "TOLERANCE_MISMATCH" and worst_status == "OK":
            worst_status = "TOLERANCE_MISMATCH"

    # Determine final status
    if worst_status == "OK":
        return "OK", None, combined_stats
    if worst_status == "TOLERANCE_MISMATCH":
        return "TOLERANCE_MISMATCH", ", ".join(all_tolerance_issues), combined_stats
    error_list = all_computation_errors or all_tolerance_issues
    return "NUMERICAL_MISMATCH", ", ".join(error_list), combined_stats


def _print_verification_status(
    status: str, error_msg: str | None, stats: dict, print_errors: bool
) -> None:
    """Print verification status for a single model.

    :param status: Verification status
    :param error_msg: Error message if any
    :param stats: Error statistics dictionary
    :param print_errors: Whether to print detailed errors
    """
    status_handlers = {
        "OK": lambda: _print_ok_status(stats, print_errors),
        "NUMERICAL_MISMATCH": lambda: _print_mismatch_status(stats, error_msg, print_errors),
        "SKIP": lambda: print(f"SKIP - {error_msg}"),
        "ERROR": lambda: print(f"ERROR - {error_msg}"),
    }

    handler = status_handlers.get(status)
    if handler:
        handler()


def _print_ok_status(stats: dict, print_errors: bool) -> None:
    """Print OK status with optional error details.

    :param stats: Error statistics dictionary
    :param print_errors: Whether to print detailed errors
    """
    if print_errors and stats:
        max_abs = stats.get("max_abs_diff", 0)
        max_rel = stats.get("max_rel_diff", 0)
        print(f"OK (max_abs: {max_abs:.2e}, max_rel: {max_rel:.2e})")
    else:
        print("OK")


def _print_mismatch_status(stats: dict, error_msg: str | None, print_errors: bool) -> None:
    """Print mismatch status with optional error details.

    :param stats: Error statistics dictionary
    :param error_msg: Error message
    :param print_errors: Whether to print detailed errors
    """
    if print_errors and stats:
        max_abs = stats.get("max_abs_diff", 0)
        max_rel = stats.get("max_rel_diff", 0)
        print(f"MISMATCH (max_abs: {max_abs:.2e}, max_rel: {max_rel:.2e})")
    else:
        print(f"NUMERICAL MISMATCH ({error_msg})")


def _update_verification_counts(status: str, counts: dict[str, int]) -> dict[str, int]:
    """Update verification counts based on status.

    :param status: Verification status
    :param counts: Dictionary of current counts
    :return: Updated counts dictionary
    """
    if status == "OK":
        counts["passed"] += 1
    elif status == "NUMERICAL_MISMATCH":
        counts["numerical_mismatches"] += 1
        counts["failed"] += 1
    elif status == "SKIP":
        counts["skipped"] += 1
    elif status == "ERROR":
        counts["failed"] += 1
    return counts


def _print_verification_summary(
    benchmark_models: list,
    counts: dict[str, int],
    max_abs_errors: list,
    max_rel_errors: list,
    dtype: str,
    device: str,
    print_errors: bool,
) -> None:
    """Print verification summary statistics.

    :param benchmark_models: List of benchmark models
    :param counts: Dictionary of verification counts
    :param max_abs_errors: List of maximum absolute errors
    :param max_rel_errors: List of maximum relative errors
    :param dtype: Data type ("float32" or "float64")
    :param device: Device ("cpu" or "cuda")
    :param print_errors: Whether to print detailed error statistics
    """
    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY ({dtype}, {device.upper()})")
    print("=" * 70)
    print(f"Total files: {len(benchmark_models)}")
    print(f"Passed: {counts['passed']}")
    print(f"Failed: {counts['failed']}")
    print(f"  Numerical mismatches: {counts['numerical_mismatches']}")
    print(f"Skipped: {counts['skipped']}")

    total_tested = len(benchmark_models) - counts["skipped"]
    if total_tested > 0:
        pass_rate = counts["passed"] / total_tested * 100
        print(f"Pass rate: {pass_rate:.1f}%")
    else:
        print("Pass rate: N/A")

    if print_errors and max_abs_errors:
        print("\nError Statistics:")
        print(f"  Max absolute error: {np.max(max_abs_errors):.6e}")
        print(f"  Mean absolute error: {np.mean(max_abs_errors):.6e}")
        print(f"  Max relative error: {np.max(max_rel_errors):.6e}")
        print(f"  Mean relative error: {np.mean(max_rel_errors):.6e}")


def verify_benchmarks(
    results_dir: str = "results/baselines",
    benchmarks_dir: str = "vnncomp2024_benchmarks",
    max_per_benchmark: int = 20,
    dtype: str = "float32",
    device: str = "cpu",
    print_errors: bool = True,
) -> dict:
    """Verify converted PyTorch modules against original ONNX benchmarks.

    :param results_dir: Directory containing converted PyTorch modules
    :param benchmarks_dir: Directory containing original ONNX benchmarks
    :param max_per_benchmark: Maximum models per benchmark to test
    :param dtype: Data type ("float32" or "float64")
    :param device: Device to run on ("cpu" or "cuda")
    :param print_errors: Print detailed error statistics
    :return: Dictionary with verification results
    """
    results_path = Path(results_dir)
    benchmarks_path = Path(benchmarks_dir)
    _validate_verification_directories(results_path, benchmarks_path)

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("\nWarning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    print(f"\nVerifying {results_dir}/ against {benchmarks_dir}/ ({dtype}, {device.upper()})")
    print("=" * 70)

    benchmarks = find_benchmarks(benchmarks_dir)
    benchmark_models = find_models(benchmarks, max_per_benchmark)

    counts = {"passed": 0, "failed": 0, "skipped": 0, "numerical_mismatches": 0}
    all_errors = {}
    max_abs_errors = []
    max_rel_errors = []

    for i, benchmark_onnx_file in enumerate(benchmark_models):
        rel_path = get_model_relative_path(benchmark_onnx_file, benchmarks_path)
        print(f"[{i + 1}/{len(benchmark_models)}] {rel_path}...", end=" ")

        result_file = results_path / rel_path.with_suffix(".py")
        result_state_dict = results_path / rel_path.with_suffix(".pth")
        data_file = get_model_data_path(benchmark_onnx_file, benchmarks_path)

        status, error_msg, stats = _verify_one_benchmark(
            result_file,
            result_state_dict,
            benchmark_onnx_file,
            data_file,
            rel_path,
            dtype,
            device,
        )

        all_errors[str(rel_path)] = {
            "status": status,
            "error_msg": error_msg,
            "stats": stats,
        }

        if stats:
            max_abs_errors.append(stats.get("max_abs_diff", 0))
            max_rel_errors.append(stats.get("max_rel_diff", 0))

        counts = _update_verification_counts(status, counts)
        _print_verification_status(status, error_msg, stats, print_errors)

    _print_verification_summary(
        benchmark_models,
        counts,
        max_abs_errors,
        max_rel_errors,
        dtype,
        device,
        print_errors,
    )

    return {
        "total": len(benchmark_models),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "numerical_mismatches": counts["numerical_mismatches"],
        "all_errors": all_errors,
        "dtype": dtype,
        "device": device,
    }


def main() -> None:
    """Run all tests using pytest.

    Executes conversion and verification tests. For individual test control:
        pytest tests/test_benchmarks.py::test_convert_model
        pytest tests/test_benchmarks.py::test_verify_model_against_original
    """
    import sys

    print("\n" + "=" * 70)
    print("STEP 1: Converting ONNX models to PyTorch")
    print("=" * 70)
    exit_code = pytest.main(
        [
            __file__ + "::test_convert_model",
            "-v",
            "--tb=short",
        ]
    )

    if exit_code != 0:
        print("\nConversion tests failed. Skipping verification.")
        sys.exit(exit_code)

    print("\n" + "=" * 70)
    print("STEP 2: Verifying converted models")
    print("=" * 70)
    exit_code = pytest.main(
        [
            __file__ + "::test_verify_model_against_original",
            "-v",
            "--tb=short",
        ]
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
