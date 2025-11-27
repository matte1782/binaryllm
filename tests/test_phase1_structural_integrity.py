"""Structural integrity guardrails for the Phase 1 tree (T14).

These tests do not execute runtime logic. Instead, they act as canaries that
fail immediately if someone reintroduces previously observed structural issues:

- duplicate test suites created via copy/paste mishaps,
- multiple `run_phase1_experiment` entrypoints in the runner module,
- regression to pre-v2 logging schemas that omitted the system metadata block.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = PROJECT_ROOT / "src" / "experiments" / "runners" / "phase1_binary_embeddings.py"

_V1_REQUIRED_FIELDS = frozenset({"encoder_name", "dataset_name", "code_bits", "metrics", "hypotheses"})


def _iter_test_modules() -> List[Path]:
    return sorted(path for path in TESTS_ROOT.rglob("test_*.py") if path.is_file())


def _collect_duplicate_tests(module_path: Path) -> List[Tuple[str, int, int]]:
    """Return a list of duplicate test qualnames within a module."""

    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    duplicates: List[Tuple[str, int, int]] = []
    seen: Dict[str, int] = {}
    stack: List[str] = []

    def register(name: str, lineno: int) -> None:
        qualname = ".".join(stack + [name]) if stack else name
        if qualname in seen:
            duplicates.append((qualname, seen[qualname], lineno))
        else:
            seen[qualname] = lineno

    class _Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802 - pytest-style classes
            if node.name.lower().startswith("test"):
                register(node.name, node.lineno)
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802 - pytest free funcs
            if node.name.startswith("test_"):
                register(node.name, node.lineno)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # pragma: no cover
            self.visit_FunctionDef(node)  # type: ignore[arg-type]

    _Visitor().visit(tree)
    return duplicates


def test_test_modules_have_unique_test_definitions() -> None:
    """Detect accidental duplication of entire pytest modules."""

    failures: Dict[Path, List[Tuple[str, int, int]]] = defaultdict(list)
    for module_path in _iter_test_modules():
        dups = _collect_duplicate_tests(module_path)
        if dups:
            failures[module_path].extend(dups)

    if failures:
        formatted = []
        for path, dup_entries in failures.items():
            for qualname, first_line, duplicate_line in dup_entries:
                formatted.append(
                    f"{path}: duplicate definition for '{qualname}' (lines {first_line} and {duplicate_line})"
                )
        pytest.fail("Duplicate pytest definitions detected:\n" + "\n".join(formatted))


def test_runner_has_single_phase1_entrypoint() -> None:
    """Ensure only one run_phase1_experiment is defined in the runner module."""

    source = RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER_PATH))
    entrypoints = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_phase1_experiment"
    ]
    assert len(entrypoints) == 1, f"duplicate run_phase1_experiment definitions detected: {len(entrypoints)}"


def test_logging_required_fields_reflect_v2_contract() -> None:
    """Fail fast if REQUIRED_LOG_FIELDS silently regresses to the v1 schema."""

    from src.utils import logging as logging_utils  # noqa: WPS433 - runtime import for assertions

    required = logging_utils.REQUIRED_LOG_FIELDS
    assert "system" in required, "logging schema must expose system metadata per phase1-v2 spec"
    assert "git_hash" in required, "logs must capture git hash for reproducibility"
    assert set(required) != _V1_REQUIRED_FIELDS, "REQUIRED_LOG_FIELDS regressed to the v1 minimal schema"
    assert len(required) >= 10, "phase1-v2 logs expect the expanded field set (>=10 keys)"
























