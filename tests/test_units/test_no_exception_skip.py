"""Guard unit tests against hiding implementation failures as skips."""

from __future__ import annotations

import ast
from pathlib import Path

TEST_ROOT = Path(__file__).parent


def _is_pytest_skip(node: ast.AST) -> bool:
    """Return whether *node* calls ``pytest.skip``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "skip"
    )


def test_unit_tests_do_not_skip_caught_exceptions() -> None:
    """Require implementation exceptions to fail their unit test."""
    offenders: list[str] = []
    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if any(_is_pytest_skip(child) for child in ast.walk(node)):
                offenders.append(f"{path.relative_to(TEST_ROOT)}:{node.lineno}")

    assert not offenders, "exception-to-skip handlers hide regressions: " + ", ".join(offenders)
