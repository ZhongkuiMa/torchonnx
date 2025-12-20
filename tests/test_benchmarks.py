"""Baseline management for SlimONNX regression testing.

This module provides functions to:
1. Optimize models and save to results/baselines/
2. Save results as archived baselines to baselines/
3. Verify results against baselines (structural + numerical)
4. Verify results against original benchmark models

Directory structure:
- benchmarks/          # Original unoptimized models
- results/baselines/   # Current optimization results
- baselines/           # Archived good baselines for regression testing
"""

__docformat__ = "restructuredtext"
__all__ = [
    "convert_model",
    "convert_all_models",
    "save_as_baseline",
    "verify_benchmarks",
]

import importlib.util
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from torchonnx.tests.benchmark_utils import (
    find_benchmarks,
    find_models,
    get_model_benchmark_name,
    get_model_relative_path,
    get_model_data_path,
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
    return {name: output for name, output in zip(output_names, outputs)}


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
    elif isinstance(output, tuple):
        return {f"output_{i}": out.cpu().numpy() for i, out in enumerate(output)}
    elif isinstance(output, dict):
        return {k: v.cpu().numpy() for k, v in output.items()}
    else:
        return {"output": output}


def _check_tolerance(max_abs: float, max_rel: float) -> bool:
    """Check if errors pass three-tier tolerance criteria.

    :param max_abs: Maximum absolute error
    :param max_rel: Maximum relative error
    :return: True if passes any tier
    """
    passes_tier1 = max_abs < TOLERANCE_TIER1_ABS
    passes_tier2 = max_rel < TOLERANCE_TIER2_REL
    passes_tier3 = max_rel < TOLERANCE_TIER3_REL and max_abs < TOLERANCE_TIER3_ABS
    return passes_tier1 or passes_tier2 or passes_tier3


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
    mismatches = []
    all_diffs = []
    all_rel_diffs = []

    if out1.shape != out2.shape:
        if _is_scalar_shape_mismatch(out1.shape, out2.shape):
            out1_flat = out1.flatten()
            out2_flat = out2.flatten()
            diff, rel_diff, max_abs, max_rel = _compute_errors(out1_flat, out2_flat)
            all_diffs.extend(diff)
            all_rel_diffs.extend(rel_diff)

            if not _check_tolerance(max_abs, max_rel):
                mismatches.append(
                    f"{key1} {key2}: max diff {max_abs:.2e}, rel {max_rel:.2e} (shape: {out1.shape} vs {out2.shape})"
                )
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

    for key1, key2 in zip(outputs1.keys(), outputs2.keys()):
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

    rel_path = get_model_relative_path(model_path, benchmarks_root)
    pytorch_module_path = output_dir / rel_path.with_suffix(".py")
    state_dict_path = output_dir / rel_path.with_suffix(".pth")

    pytorch_module_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        start_time = time.perf_counter()

        # Use TorchONNX to convert ONNX model to PyTorch model
        from torchonnx.torchonnx import TorchONNX

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

    except (
        IOError,
        OSError,
        ValueError,
        RuntimeError,
        AttributeError,
        ImportError,
    ) as error:
        return {
            "success": False,
            "benchmark": benchmark_name,
            "model": model_name,
            "time": 0.0,
            "py_path": None,
            "pth_path": None,
            "error": str(error),
        }


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

        print(f"[{i+1}/{len(models)}] {rel_path}...", end=" ")

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
        print(f"Success rate: {success_count/len(models)*100:.1f}%")
    print(f"Total conversion time: {total_time:.2f}s")
    print(
        f"Avg time per model: {total_time/success_count:.2f}s"
        if success_count > 0
        else "N/A"
    )
    print(f"Total wall time: {elapsed_total:.2f}s")

    return {
        "total": len(models),
        "success": success_count,
        "failed": failed_count,
        "total_time": total_time,
        "wall_time": elapsed_total,
    }


def save_as_baseline(
    results_dir: str = "results/baselines", baselines_dir: str = "baselines"
) -> tuple[int, int]:
    """Save current results as archived baselines.

    Copies entire results directory structure to baselines directory.

    :param results_dir: Source directory containing current results
    :param baselines_dir: Target directory for archived baselines
    :return: Tuple of (num_copied, num_failed)
    """
    results_path = Path(results_dir)
    baselines_path = Path(baselines_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    print(f"Saving {results_dir}/ as baselines to {baselines_dir}/")
    print("=" * 70)

    # Remove old baselines if they exist
    if baselines_path.exists():
        print(f"Removing old baselines {baselines_path}...")
        shutil.rmtree(baselines_path)

    # Copy results to baselines
    shutil.copytree(results_path, baselines_path)

    # Count files
    onnx_count = len(list(baselines_path.rglob("*.onnx")))
    json_count = len(list(baselines_path.rglob("*.json")))

    print(f"Copied {onnx_count} ONNX files and {json_count} JSON files")
    print("=" * 70)

    return onnx_count, json_count


def _validate_verification_directories(
    results_path: Path, benchmarks_path: Path
) -> None:
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

    :param result_file: Path to converted PyTorch module file
    :param result_state_dict: Path to converted state dict file
    :param benchmark_onnx_file: Path to original ONNX model file
    :param data_file: Path to test data npz file
    :param rel_path: Relative path for reporting
    :param dtype: Data type ("float32" or "float64")
    :param device: Device to run on ("cpu" or "cuda")
    :return: Tuple of (status, error_message, statistics) where status is "OK", "NUMERICAL_MISMATCH", "SKIP", "ERROR"
    """
    if not result_file.exists():
        return "SKIP", f"Result file not found: {result_file.name}", {}

    if not result_state_dict.exists():
        return "SKIP", f"State dict not found: {result_state_dict.name}", {}

    if not benchmark_onnx_file.exists():
        return "SKIP", f"Benchmark ONNX not found: {benchmark_onnx_file.name}", {}

    if not data_file.exists():
        return "SKIP", f"No test data file: {data_file.name}", {}

    try:
        test_inputs = _load_test_data_from_npz(data_file)
        if not test_inputs:
            return "SKIP", "No inputs in data file", {}
    except (IOError, KeyError, ValueError, IndexError) as error:
        return "SKIP", f"Error loading data: {error}", {}

    all_match = True
    mismatch_info = []
    combined_stats = {}

    for i, inputs in enumerate(test_inputs):
        try:
            result_outputs = _run_pytorch_module(
                str(result_file), str(result_state_dict), inputs, dtype, device
            )
            benchmark_outputs = _run_onnx_model(str(benchmark_onnx_file), inputs)

            match, mismatches, stats = _compare_outputs(
                result_outputs, benchmark_outputs
            )
            combined_stats = stats  # Use stats from last input
            if not match:
                all_match = False
                mismatch_info.extend(mismatches)
                break
        except (
            RuntimeError,
            ValueError,
            AttributeError,
            ImportError,
            IndexError,
            TypeError,
            KeyError,
        ) as error:
            return "ERROR", str(error), {}

    if all_match:
        return "OK", None, combined_stats
    else:
        return "NUMERICAL_MISMATCH", ", ".join(mismatch_info), combined_stats


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
        "NUMERICAL_MISMATCH": lambda: _print_mismatch_status(
            stats, error_msg, print_errors
        ),
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


def _print_mismatch_status(
    stats: dict, error_msg: str | None, print_errors: bool
) -> None:
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
        print(f"\nError Statistics:")
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
        print(f"\nWarning: CUDA requested but not available. Falling back to CPU.")
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
        print(f"[{i+1}/{len(benchmark_models)}] {rel_path}...", end=" ")

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
    """Main entry point for script execution."""
    convert_all_models()
    save_as_baseline()

    # Test all dtype and device combinations
    dtypes = ["float32", "float64"]
    devices = ["cpu"]

    # Add CUDA if available
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"\nCUDA is available. Testing on: {devices}")
    else:
        print(f"\nCUDA is not available. Testing on: {devices}")

    for dtype in dtypes:
        for device in devices:
            verify_benchmarks(dtype=dtype, device=device)


if __name__ == "__main__":
    main()
