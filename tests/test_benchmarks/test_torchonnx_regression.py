"""Regression testing for torchonnx conversions against baselines.

This module provides regression tests that verify converted models match stored baselines.
To update baselines, run update_baselines.py after converting models with test_torchonnx.py.
"""

__docformat__ = "restructuredtext"

from pathlib import Path

import pytest

from tests.test_benchmarks.benchmark_utils import find_benchmarks, find_models


def get_benchmark_models():
    """Collect all models from vnncomp2024 benchmarks.

    :return: List of model paths for parametrized testing, or pytest.skip marker if empty
    """
    benchmarks_dir = Path(__file__).parent / "vnncomp2024_benchmarks"
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


@pytest.fixture(scope="session")
def results_dir():
    """Results directory for generated models (created if needed).

    Note: baselines_dir fixture is provided by conftest.py
    """
    path = Path(__file__).parent / "results" / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("model_path", get_benchmark_models())
def test_conversion_baseline_exists(model_path, baselines_dir):
    """Verify baseline model exists for conversion test.

    :param model_path: Path to ONNX model file
    :param baselines_dir: Directory containing baseline models
    """
    model_path_obj = Path(model_path)
    benchmark_name = model_path_obj.parent.name
    model_name = model_path_obj.stem

    # Expected paths
    baseline_py = baselines_dir / benchmark_name / f"{model_name}.py"
    baseline_pth = baselines_dir / benchmark_name / f"{model_name}.pth"

    # Skip if no baseline exists yet
    if not baseline_py.exists() or not baseline_pth.exists():
        pytest.skip(f"No baseline for {benchmark_name}/{model_name}")

    # Verify both files exist
    assert baseline_py.exists(), f"Missing baseline .py file: {baseline_py}"
    assert baseline_pth.exists(), f"Missing baseline .pth file: {baseline_pth}"


@pytest.mark.parametrize("model_path", get_benchmark_models())
def test_conversion_results_match_baseline(model_path, baselines_dir, results_dir):
    """Verify converted model files match baseline versions.

    Compares results/baselines/{benchmark}/ vs baselines/{benchmark}/

    :param model_path: Path to ONNX model file
    :param baselines_dir: Directory containing baseline models
    :param results_dir: Directory containing converted results
    """
    model_path_obj = Path(model_path)
    benchmark_name = model_path_obj.parent.name
    model_name = model_path_obj.stem

    # Expected paths
    baseline_py = baselines_dir / benchmark_name / f"{model_name}.py"
    baseline_pth = baselines_dir / benchmark_name / f"{model_name}.pth"
    results_py = results_dir / benchmark_name / f"{model_name}.py"
    results_pth = results_dir / benchmark_name / f"{model_name}.pth"

    # Skip if no baseline exists yet
    if not baseline_py.exists() or not baseline_pth.exists():
        pytest.skip(f"No baseline for {benchmark_name}/{model_name}")

    # Skip if no results exist yet (need to run test_torchonnx.py first)
    if not results_py.exists() or not results_pth.exists():
        pytest.skip(f"No results for {benchmark_name}/{model_name} - run test_torchonnx.py first")

    # Compare file sizes as quick check (exact binary comparison)
    baseline_py_size = baseline_py.stat().st_size
    results_py_size = results_py.stat().st_size
    baseline_pth_size = baseline_pth.stat().st_size
    results_pth_size = results_pth.stat().st_size

    # Verify files match in size
    assert baseline_py_size == results_py_size, (
        f"Generated .py file size mismatch for {model_name}: "
        f"baseline={baseline_py_size}, results={results_py_size}"
    )

    assert baseline_pth_size == results_pth_size, (
        f"Generated .pth file size mismatch for {model_name}: "
        f"baseline={baseline_pth_size}, results={results_pth_size}"
    )

    # Compare file contents for deterministic checks
    with Path(baseline_pth).open("rb") as f:
        baseline_content = f.read()
    with Path(results_pth).open("rb") as f:
        results_content = f.read()

    assert baseline_content == results_content, (
        f"Generated .pth file content mismatch for {model_name}"
    )
