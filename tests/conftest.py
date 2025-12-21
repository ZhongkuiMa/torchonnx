"""Pytest configuration and fixtures for torchonnx tests."""

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
