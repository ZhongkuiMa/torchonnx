#!/usr/bin/env python3
"""Update baseline outputs for torchonnx regression testing.

This script copies converted models from results/ to baselines/ to update
golden references for regression testing.

Workflow:
    1. Run test_torchonnx.py to generate results/baselines/
    2. Run this script to copy results/baselines/ -> baselines/
    3. Run test_torchonnx_regression.py to verify

Usage:
    python update_baselines.py                     # Copy all results to baselines
    python update_baselines.py --benchmark acasxu_2023  # Copy specific benchmark
    python update_baselines.py --dry-run           # Show what would be copied
"""

import argparse
import shutil
import sys
from pathlib import Path


def copy_baseline(
    result_path: Path, baseline_path: Path, dry_run: bool = False
) -> tuple[bool, int]:
    """Copy a single result file/directory to baseline.

    :param result_path: Source path in results/baselines/
    :param baseline_path: Destination path in baselines/
    :param dry_run: If True, only show what would be copied
    :return: Tuple of (success, file_count)
    """
    if dry_run:
        if result_path.exists():
            is_file = result_path.is_file()
            file_type = "file" if is_file else "directory"
            print(f"  [DRY-RUN] Would copy: {result_path} -> {baseline_path} ({file_type})")
            return True, 1
        print(f"  [DRY-RUN] Would skip (missing): {result_path}")
        return False, 0

    # Check source exists
    if not result_path.exists():
        print(f"  [SKIP] Missing result: {result_path}")
        return False, 0

    # Create baseline directory
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old baseline if exists
    if baseline_path.exists():
        if baseline_path.is_dir():
            shutil.rmtree(baseline_path)
        else:
            baseline_path.unlink()

    # Copy result to baseline (handles both files and directories)
    if result_path.is_dir():
        shutil.copytree(result_path, baseline_path)
    else:
        shutil.copy2(result_path, baseline_path)

    print(f"  [OK] Copied: {baseline_path}")
    return True, 1


def update_benchmark(
    benchmark_name: str,
    results_dir: Path,
    baselines_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Update baselines for one benchmark.

    :param benchmark_name: Name of benchmark
    :param results_dir: Root results directory (results/baselines/)
    :param baselines_dir: Root baselines directory
    :param dry_run: If True, only show what would be copied
    :return: Tuple of (success_count, total_count)
    """
    result_bench_dir = results_dir / benchmark_name
    baseline_bench_dir = baselines_dir / benchmark_name

    if not result_bench_dir.exists():
        print(f"  [SKIP] No results: {benchmark_name}")
        return 0, 0

    # Count .py files as indicator of work
    py_files = list(result_bench_dir.glob("*.py"))
    if not py_files:
        print(f"  [SKIP] No converted models: {result_bench_dir}")
        return 0, 0

    # Copy the entire benchmark directory
    success = 0
    total = 1

    copied, _ = copy_baseline(result_bench_dir, baseline_bench_dir, dry_run)
    if copied:
        success += 1

    return success, total


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Update baselines by copying results/baselines/ to baselines/",
        epilog="Example: python update_baselines.py --benchmark acasxu_2023",
    )
    parser.add_argument(
        "--benchmark",
        help="Update specific benchmark only (e.g., acasxu_2023)",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without making changes",
    )
    return parser.parse_args()


def _get_benchmarks_to_update(args: argparse.Namespace, results_dir: Path) -> list[str] | None:
    """Determine which benchmarks to update.

    :param args: Parsed arguments
    :param results_dir: Path to results directory
    :return: List of benchmark names or None if error
    """
    if args.benchmark:
        benchmark_path = results_dir / args.benchmark
        if not benchmark_path.exists():
            print(f"Error: Benchmark '{args.benchmark}' not found in results/baselines/")
            print(f"\nAvailable benchmarks in {results_dir}:")
            for bench in sorted(results_dir.iterdir()):
                if bench.is_dir():
                    print(f"  - {bench.name}")
            return None
        return [args.benchmark]

    benchmarks = [bench.name for bench in sorted(results_dir.iterdir()) if bench.is_dir()]
    return benchmarks if benchmarks else None


def main() -> int:
    """Update baselines from results directory."""
    args = _parse_arguments()

    # Setup paths
    test_dir = Path(__file__).parent
    results_dir = test_dir / "results" / "baselines"
    baselines_dir = test_dir / "baselines"

    # Check results directory exists
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("\nRun test_torchonnx.py first to generate results/baselines/")
        return 1

    # Create baselines directory if missing
    if not args.dry_run:
        baselines_dir.mkdir(exist_ok=True)

    # Find benchmarks to update
    benchmarks_to_update = _get_benchmarks_to_update(args, results_dir)
    if not benchmarks_to_update:
        print("No benchmarks found in results/baselines/")
        return 1

    # Update baselines
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Updating baselines from results/baselines/")
    print(f"Benchmarks: {len(benchmarks_to_update)}")
    print("=" * 80)

    total_success = 0
    total_count = 0

    for benchmark_name in benchmarks_to_update:
        print(f"\n{benchmark_name}:")
        success, count = update_benchmark(
            benchmark_name, results_dir, baselines_dir, dry_run=args.dry_run
        )
        total_success += success
        total_count += count

        if count > 0:
            print(f"  {success}/{count} benchmarks copied")

    print("\n" + "=" * 80)
    print(f"{'[DRY-RUN] ' if args.dry_run else ''}Baseline update complete!")
    print(f"Total: {total_success}/{total_count} benchmarks copied")

    if not args.dry_run and total_success > 0:
        print(f"\nBaselines updated in: {baselines_dir}")
        print("Run test_torchonnx_regression.py to verify")

    return 0


if __name__ == "__main__":
    sys.exit(main())
