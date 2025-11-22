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
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, "../..")

from benchmark_utils import (
    find_benchmarks,
    find_models,
    get_model_benchmark_name,
    get_model_relative_path,
    get_model_data_path,
)


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
    module_path: str, state_dict_path: str, inputs: np.ndarray
) -> dict:
    """Run PyTorch module inference.

    :param module_path: Path to PyTorch module file
    :param state_dict_path: Path to state dict file
    :param inputs: input arrays
    :return: Dictionary of output arrays
    """
    module_file = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_file.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class_name = module.__all__[0]
    model_class = getattr(module, class_name)
    model = model_class()

    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    input_tensor = torch.from_numpy(inputs).float()
    with torch.no_grad():
        output = model(input_tensor)

    if isinstance(output, torch.Tensor):
        return {"output": output.numpy()}
    elif isinstance(output, tuple):
        return {f"output_{i}": out.numpy() for i, out in enumerate(output)}
    elif isinstance(output, dict):
        return {k: v.numpy() for k, v in output.items()}
    else:
        return {"output": output}


def _compare_outputs(
    outputs1: dict, outputs2: dict, rtol: float = 1e-4, atol: float = 1e-5
) -> tuple[bool, list[str]]:
    """Compare outputs from two models.

    :param outputs1: First model outputs
    :param outputs2: Second model outputs
    :param rtol: Relative tolerance (default 1e-4 for numerical precision)
    :param atol: Absolute tolerance (default 1e-5 for numerical precision)
    :return: Tuple of (all_match, mismatch_messages)
    """
    mismatches = []
    for key1, key2 in zip(outputs1.keys(), outputs2.keys()):
        out1 = outputs1[key1]
        out2 = outputs2[key2]

        if out1.shape != out2.shape:
            # Accept scalar () vs [1] shape difference
            is_scalar_vs_single = (out1.shape == () and out2.shape == (1,)) or (
                out1.shape == (1,) and out2.shape == ()
            )

            if is_scalar_vs_single:
                # Reshape to compare values
                out1_flat = out1.flatten()
                out2_flat = out2.flatten()
                if not np.allclose(out1_flat, out2_flat, rtol=rtol, atol=atol):
                    max_diff = np.max(np.abs(out1_flat - out2_flat))
                    mismatches.append(
                        f"{key1} {key2}: max diff {max_diff:.2e} (acceptable shape difference: {out1.shape} vs {out2.shape})"
                    )
                # Otherwise OK - scalar vs [1] with matching values is acceptable
            else:
                mismatches.append(f"{key1}: shape {out1.shape} vs {key2}: {out2.shape}")
            continue

        if not np.allclose(out1, out2, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(out1 - out2))
            mismatches.append(f"{key1} {key2}: max diff {max_diff:.2e}")

    return len(mismatches) == 0, mismatches


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
    benchmark_dir: str = "benchmarks",
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
) -> tuple[str, str | None]:
    """Verify converted PyTorch module against original ONNX model.

    :param result_file: Path to converted PyTorch module file
    :param result_state_dict: Path to converted state dict file
    :param benchmark_onnx_file: Path to original ONNX model file
    :param data_file: Path to test data npz file
    :param rel_path: Relative path for reporting
    :return: Tuple of (status, error_message) where status is "OK", "NUMERICAL_MISMATCH", "SKIP", "ERROR"
    """
    if not result_file.exists():
        return "SKIP", f"Result file not found: {result_file.name}"

    if not result_state_dict.exists():
        return "SKIP", f"State dict not found: {result_state_dict.name}"

    if not benchmark_onnx_file.exists():
        return "SKIP", f"Benchmark ONNX not found: {benchmark_onnx_file.name}"

    if not data_file.exists():
        return "SKIP", f"No test data file: {data_file.name}"

    try:
        test_inputs = _load_test_data_from_npz(data_file)
        if not test_inputs:
            return "SKIP", "No inputs in data file"
    except (IOError, KeyError, ValueError, IndexError) as error:
        return "SKIP", f"Error loading data: {error}"

    all_match = True
    mismatch_info = []

    for i, inputs in enumerate(test_inputs):
        try:
            result_outputs = _run_pytorch_module(
                str(result_file), str(result_state_dict), inputs
            )
            benchmark_outputs = _run_onnx_model(str(benchmark_onnx_file), inputs)

            match, mismatches = _compare_outputs(result_outputs, benchmark_outputs)
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
            return "ERROR", str(error)

    if all_match:
        return "OK", None
    else:
        return "NUMERICAL_MISMATCH", ", ".join(mismatch_info)


def verify_benchmarks(
    results_dir: str = "results/baselines",
    benchmarks_dir: str = "benchmarks",
    max_per_benchmark: int = 20,
) -> dict:
    """Verify converted PyTorch modules against original ONNX benchmarks.

    Compares numerical outputs between original ONNX and converted PyTorch.

    :param results_dir: Directory containing converted PyTorch modules
    :param benchmarks_dir: Directory containing original ONNX benchmarks with test data
    :param max_per_benchmark: Maximum models per benchmark to test
    :return: Dictionary with verification results
    """
    results_path = Path(results_dir)
    benchmarks_path = Path(benchmarks_dir)

    _validate_verification_directories(results_path, benchmarks_path)

    print(
        f"\nVerifying converted models in {results_dir}/ against original ONNX in {benchmarks_dir}/"
    )
    print("=" * 70)

    benchmarks = find_benchmarks(benchmarks_dir)
    benchmark_models = find_models(benchmarks, max_per_benchmark)

    passed = 0
    failed = 0
    skipped = 0
    numerical_mismatches = 0

    for i, benchmark_onnx_file in enumerate(benchmark_models):
        rel_path = get_model_relative_path(benchmark_onnx_file, benchmarks_path)
        rel_path_py = rel_path.with_suffix(".py")
        rel_path_pth = rel_path.with_suffix(".pth")

        print(f"[{i+1}/{len(benchmark_models)}] {rel_path}...", end=" ")

        result_file = results_path / rel_path_py
        result_state_dict = results_path / rel_path_pth
        data_file = get_model_data_path(benchmark_onnx_file, benchmarks_path)

        status, error_msg = _verify_one_benchmark(
            result_file, result_state_dict, benchmark_onnx_file, data_file, rel_path
        )

        if status == "OK":
            passed += 1
            print("OK")
        elif status == "NUMERICAL_MISMATCH":
            numerical_mismatches += 1
            failed += 1
            print(f"NUMERICAL MISMATCH ({error_msg})")
        elif status == "SKIP":
            skipped += 1
            print(f"SKIP - {error_msg}")
        elif status == "ERROR":
            failed += 1
            print(f"ERROR - {error_msg}")

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total files: {len(benchmark_models)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"  Numerical mismatches: {numerical_mismatches}")
    print(f"Skipped: {skipped}")
    total_tested = len(benchmark_models) - skipped
    if total_tested > 0:
        print(f"Pass rate: {passed/total_tested*100:.1f}%")
    else:
        print("Pass rate: N/A")

    return {
        "total": len(benchmark_models),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "numerical_mismatches": numerical_mismatches,
    }


def main() -> None:
    """Main entry point for script execution."""
    convert_all_models()
    # save_as_baseline()
    verify_benchmarks()


if __name__ == "__main__":
    main()
