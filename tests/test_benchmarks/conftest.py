"""Pytest configuration and fixtures for torchonnx benchmark tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def benchmarks_root():
    """Root directory for vnncomp2024 benchmarks.

    :return: Path to benchmarks root directory
    """
    return Path(__file__).parent / "vnncomp2024_benchmarks"


@pytest.fixture(scope="session")
def output_dir_baselines():
    """Output directory for baseline converted models.

    :return: Path to baselines output directory (created if needed)
    """
    path = Path(__file__).parent / "results" / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def output_dir_vmap():
    """Output directory for vmap mode converted models.

    :return: Path to vmap output directory (created if needed)
    """
    path = Path(__file__).parent / "results" / "vmap"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def baselines_dir():
    """Baselines directory for storing reference models.

    :return: Path to baselines directory (created if needed)
    """
    path = Path(__file__).parent / "baselines"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def max_per_benchmark():
    """Maximum models per benchmark to test.

    :return: Maximum model count per benchmark
    """
    return 20


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark (slow, run with: pytest -m benchmark)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip benchmark-dependent tests if data is missing or parameters are empty."""
    benchmarks_path = Path(__file__).parent / "vnncomp2024_benchmarks"
    has_benchmark_dirs = bool(list(benchmarks_path.glob("*/instances.csv")))

    if not has_benchmark_dirs:
        skip_marker = pytest.mark.skip(
            reason="Benchmark data not available (run build_benchmarks.py)"
        )
        for item in items:
            # Skip all benchmark-dependent tests
            if "test_vnncomp2024_benchmarks" in str(item.fspath) or (
                "model_path" in item.fixturenames
                or ("dtype" in item.fixturenames and "device" in item.fixturenames)
            ):
                item.add_marker(skip_marker)
