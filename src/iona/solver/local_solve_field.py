"""Placeholder for future local solve-field integration."""

from __future__ import annotations

from iona.pipeline.result_schema import PlateSolveResult


def solve_with_local_solve_field(*args, **kwargs) -> PlateSolveResult:
    return PlateSolveResult(
        success=False,
        failure_reason="local_solve_field_not_implemented",
        diagnostics={"backend": "solve-field"},
    )

