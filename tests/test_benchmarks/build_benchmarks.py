#!/usr/bin/env python3
"""Build script to set up benchmark symlinks for torchonnx tests.

Links the vnncomp2024 and vnncomp2025 benchmark corpora (cloned as
siblings of the torchonnx repo) into ``tests/test_benchmarks/`` so the
benchmark conversion / verification suites can discover ONNX models.
"""

__docformat__ = "restructuredtext"

import shutil
import sys
from pathlib import Path

BENCHMARK_CORPORA = ("vnncomp2024_benchmarks", "vnncomp2025_benchmarks")


def _setup_one_symlink(tests_dir: Path, corpus_name: str) -> int:
    """Create one ``tests_dir/<corpus_name>`` -> ``<corpus>/benchmarks`` link.

    :param tests_dir: The ``test_benchmarks`` directory holding the links.
    :param corpus_name: Benchmark corpus folder name (e.g.
        ``vnncomp2025_benchmarks``).
    :return: Exit code (0 on success or when the corpus is absent, 1 on
        symlink failure).
    """
    symlink_path = tests_dir / corpus_name
    # The corpora are cloned as siblings of the rover_alpha repo, i.e.
    # ``<rover_project>/<corpus_name>/benchmarks``. From this file
    # (``rover_alpha/torchonnx/tests/test_benchmarks``) that is four
    # levels up.
    benchmarks_dir = (tests_dir / ".." / ".." / ".." / ".." / corpus_name / "benchmarks").resolve()

    if not benchmarks_dir.exists():
        print(f"Skip {corpus_name}: not found at {benchmarks_dir}")
        return 0

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


def setup_benchmarks_symlink() -> int:
    """Set up symlinks for every supported benchmark corpus.

    :return: Exit code (0 when all present corpora linked, 1 on any
        symlink failure).
    """
    tests_dir = Path(__file__).parent
    codes = [_setup_one_symlink(tests_dir, corpus) for corpus in BENCHMARK_CORPORA]
    return 1 if any(code != 0 for code in codes) else 0


if __name__ == "__main__":
    sys.exit(setup_benchmarks_symlink())
