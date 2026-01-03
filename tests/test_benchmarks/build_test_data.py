"""Build test data from vnncomp2024_benchmarks for torchonnx conversion tests.

Generates test inputs and outputs for ONNX models to support the test_benchmarks.py
conversion pipeline. This reduces model processing time by pre-computing reference outputs.
"""

import csv
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort


def get_models_from_benchmarks(benchmark_dir: Path, max_models: int = 20):
    """Get list of ONNX models from benchmark directory.

    :param benchmark_dir: Benchmark directory path
    :param max_models: Maximum number of models to process
    :return: List of ONNX model paths
    """
    instances_csv = benchmark_dir / "instances.csv"
    if not instances_csv.exists():
        return []

    seen_models = set()
    models = []

    with instances_csv.open() as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) < 1:
                continue

            onnx_path = row[0].strip()

            # Only process each model once
            if onnx_path not in seen_models:
                seen_models.add(onnx_path)
                model_full_path = benchmark_dir / onnx_path
                if model_full_path.exists():
                    models.append(model_full_path)

                if len(models) >= max_models:
                    break

    return models


def _get_model_input_info(onnx_path: Path) -> tuple[str, list[int]] | None:
    """Get model input name and expected shape.

    :param onnx_path: Path to ONNX model
    :return: Tuple of (input_name, expected_shape) or None if invalid
    """
    try:
        model = onnx.load(str(onnx_path))
        input_names = [
            inp.name
            for inp in model.graph.input
            if not any(init.name == inp.name for init in model.graph.initializer)
        ]

        if not input_names:
            return None

        input_name = input_names[0]
        input_tensor = next((inp for inp in model.graph.input if inp.name == input_name), None)

        if not input_tensor:
            return None

        expected_shape = [
            dim.dim_value if dim.dim_value > 0 else 1
            for dim in input_tensor.type.tensor_type.shape.dim
        ]

        return input_name, expected_shape
    except (AttributeError, ValueError, IndexError):
        return None


def generate_test_data(
    onnx_path: Path,
    output_dir: Path,
    num_samples: int = 5,
) -> bool:
    """Generate test data from ONNX model by running inference.

    :param onnx_path: Path to ONNX model
    :param output_dir: Output directory for .npz file
    :param num_samples: Number of test samples to generate
    :return: True if successful, False otherwise
    """
    try:
        # Get model input info
        input_info = _get_model_input_info(onnx_path)
        if not input_info:
            return False

        input_name, expected_shape = input_info

        # Create ONNX Runtime session
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs_list = []
        outputs_list = []

        # Generate test samples with random inputs in reasonable range
        rng = np.random.default_rng()
        for _ in range(num_samples):
            # Generate random input in [-1, 1] range
            input_array = rng.uniform(-1.0, 1.0, expected_shape).astype(np.float32)

            try:
                output = session.run(None, {input_name: input_array})
                inputs_list.append(input_array)
                outputs_list.append(output[0])
            except (ValueError, RuntimeError, KeyError):
                continue

        if not inputs_list:
            return False

        # Save results in format compatible with load_test_inputs
        # Format: {vnnlib_name: {"lower": {"inputs": [...], "outputs": [...]}, ...}}
        vnnlib_name = "generated"
        results: dict[str, Any] = {
            vnnlib_name: {
                "lower": {
                    "inputs": inputs_list,
                    "outputs": outputs_list,
                }
            }
        }

        output_file = output_dir / f"{onnx_path.stem}.npz"
        np.savez_compressed(output_file, **cast(dict[str, Any], results))

        return True

    except (OSError, ValueError, RuntimeError, AttributeError, KeyError) as e:
        print(f"  Error processing {onnx_path.name}: {e}")
        return False


def build_test_data(
    benchmarks_root: str = "vnncomp2024_benchmarks",
    max_per_benchmark: int = 20,
):
    """Build test data from benchmarks.

    Data files are saved in vnncomp2024_benchmarks/benchmark_name/data/
    to match expected structure from load_test_inputs().

    :param benchmarks_root: Root directory with benchmarks (symlink)
    :param max_per_benchmark: Maximum models per benchmark
    """
    benchmarks_path = Path(benchmarks_root)

    if not benchmarks_path.exists():
        print(f"Error: {benchmarks_root} not found")
        return

    # Get all benchmark directories
    benchmark_dirs = sorted(
        [d for d in benchmarks_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    print(f"Building test data from {len(benchmark_dirs)} benchmarks")
    print(f"Max {max_per_benchmark} models per benchmark")
    print("=" * 70)

    total_models = 0
    total_success = 0
    start_time = time.perf_counter()

    for benchmark_dir in benchmark_dirs:
        benchmark_name = benchmark_dir.name

        # Get models
        models = get_models_from_benchmarks(benchmark_dir, max_per_benchmark)

        if not models:
            continue

        # Create data subdirectory inside benchmark directory
        output_dir = benchmark_dir / "data"
        output_dir.mkdir(exist_ok=True)

        success = 0
        for model_path in models:
            total_models += 1

            if generate_test_data(model_path, output_dir):
                success += 1
                total_success += 1

        print(f"[{benchmark_name}] Processed {success}/{len(models)} models")

    elapsed = time.perf_counter() - start_time

    print("=" * 70)
    print(f"Total: {total_success}/{total_models} models processed")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nData saved in: {benchmarks_path.absolute()}/*/data/")
    npz_files = list(benchmarks_path.rglob("data/*.npz"))
    if npz_files:
        total_size = sum(f.stat().st_size for f in npz_files) / 1024 / 1024
        print(f"Files: {len(npz_files)}")
        print(f"Size: {total_size:.1f} MB")


if __name__ == "__main__":
    import sys

    max_models = 20
    if len(sys.argv) > 1:
        max_models = int(sys.argv[1])

    print(f"\nBuilding test data with max_per_benchmark={max_models}")
    build_test_data(max_per_benchmark=max_models)
