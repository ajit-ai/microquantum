"""Conformance level A - every official example executes for real.

Each example program under ``examples/`` is run as a real subprocess
against the locally installed package.  A failure inside any official
example fails the quality gate.
"""

from __future__ import annotations

from pathlib import Path

import conformance_helpers as h
import pytest

_EXAMPLES: list[tuple[str, Path]] = [
    (str(p.relative_to(h.REPO_ROOT)).replace("\\", "/"), p)
    for p in h.all_example_files()
]


def _ids() -> list[str]:
    return [rel for rel, _ in _EXAMPLES]


@pytest.mark.parametrize("example", _EXAMPLES, ids=_ids())
def test_official_example_executes(example) -> None:
    rel, path = example
    proc = h.run_example(path)
    combined = f"{proc.stdout}\n{proc.stderr}" if proc.stderr else proc.stdout
    assert proc.returncode == 0, (
        f"official example failed: {rel}\nexit={proc.returncode}\n"
        f"--- stderr ---\n{proc.stderr[-6000:]}\n--- stdout tail ---\n{proc.stdout[-6000:]}"
    )
    assert not h.has_traceback(proc), (
        f"official example raised an exception: {rel}\n{combined[-6000:]}"
    )