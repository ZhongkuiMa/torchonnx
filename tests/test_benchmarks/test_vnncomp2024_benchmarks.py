"""Parametrized tests for all vnncomp2024 benchmarks."""

__docformat__ = "restructuredtext"

from pathlib import Path

import pytest

VNNCOMP2024_BENCHMARKS = [
    "acasxu_2023",
]


@pytest.mark.parametrize("benchmark_name", VNNCOMP2024_BENCHMARKS)
def test_benchmark_directory_exists(benchmark_name: str) -> None:
    """Test that each benchmark directory exists via symlink."""
    tests_dir = Path(__file__).parent
    benchmarks_dir = tests_dir / "vnncomp2024_benchmarks"
    benchmark_path = benchmarks_dir / benchmark_name

    assert benchmarks_dir.exists(), "Symlink not found. Run build_benchmarks.py"
    assert benchmark_path.exists(), f"Benchmark {benchmark_name} not found"
    assert benchmark_path.is_dir(), f"Benchmark {benchmark_name} is not a directory"


@pytest.mark.parametrize("benchmark_name", VNNCOMP2024_BENCHMARKS)
def test_benchmark_has_instances_csv(benchmark_name: str) -> None:
    """Test that each benchmark has instances.csv."""
    tests_dir = Path(__file__).parent
    instances_csv = tests_dir / "vnncomp2024_benchmarks" / benchmark_name / "instances.csv"
    assert instances_csv.exists(), f"instances.csv not found for {benchmark_name}"
