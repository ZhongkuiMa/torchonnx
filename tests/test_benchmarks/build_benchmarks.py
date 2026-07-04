#!/usr/bin/env python3
"""Build script to set up benchmark symlinks for torchonnx tests.

Links the vnncomp_benchmarks corpus (cloned as a sibling of the rover_alpha
repo) into ``tests/test_benchmarks/`` so the benchmark conversion / verification
suites can discover ONNX models.
"""

__docformat__ = "restructuredtext"

import shutil
import sys
from pathlib import Path

BENCHMARK_CORPORA = ("vnncomp_benchmarks",)


def setup_benchmarks_symlink() -> int:
    """Set up symlink for the merged vnncomp benchmark corpus.

    :return: Exit code (0 when the corpus is linked, 1 on failure).
    """
    tests_dir = Path(__file__).parent
    symlink_path = tests_dir / "vnncomp_benchmarks"
    # corpus at <rover_project>/vnncomp_benchmarks/benchmarks
    benchmarks_dir = (
        tests_dir / ".." / ".." / ".." / ".." / "vnncomp_benchmarks" / "benchmarks"
    ).resolve()

    if not benchmarks_dir.exists():
        print(f"Error: benchmarks directory not found at {benchmarks_dir}")
        return 1

    if symlink_path.is_symlink():
        if symlink_path.resolve() == benchmarks_dir:
            print(f"Symlink up to date: {symlink_path} -> {benchmarks_dir}")
            return 0
        symlink_path.unlink()
    elif symlink_path.exists():
        if symlink_path.is_dir():
            shutil.rmtree(symlink_path)
        else:
            symlink_path.unlink()

    try:
        symlink_path.symlink_to(benchmarks_dir)
    except OSError as e:
        print(f"Error creating symlink {symlink_path} -> {benchmarks_dir}: {e}")
        return 1
    print(f"Created symlink: {symlink_path} -> {benchmarks_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(setup_benchmarks_symlink())
