"""Shared helpers for the MicroQuantum public-conformance suite.

The suite is the executable quality gate for the public API and every
official example.  Everything here is designed to be 100% green: no skips,
no xfails, no quarantines, no weakened assertions.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
DOCS_DIR = REPO_ROOT / "docs"

_TIMEOUT_S = 300


def all_example_files() -> list[Path]:
    """Every official example script under ``examples/``, deterministic order."""
    return sorted(EXAMPLES_DIR.rglob("*.py"))


def run_python(args: list[str], *, cwd: Path | None = None, timeout: int = _TIMEOUT_S):
    """Run a python subprocess against the local (source) package."""
    env = None
    import os

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def run_example(example: Path):
    """Execute a real example program (Level A conformance)."""
    return run_python(["-X", "utf8", str(example)])


def has_traceback(proc) -> bool:
    combined = (proc.stdout or "") + (proc.stderr or "")
    return "Traceback (most recent call last)" in combined


def parse_docs_python_blocks(path: Path) -> list[str]:
    """Extract ``.. code-block:: python`` (and ``.. code:: python``) bodies.

    Returns the raw text of each block (inner indentation stripped to the
    directive's content level).
    """
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\.\. (?:code-block|code):: (?:py|python)\s*\n(?:\n?)((?:\s{3,}.*\n|^\s*\n)*)",
        re.MULTILINE,
    )
    blocks: list[str] = []
    for match in pattern.finditer(text):
        body = match.group(1)
        raw_lines = body.splitlines()
        nonblank = [ln for ln in raw_lines if ln.strip()]
        if not nonblank:
            continue
        common = min(len(ln) - len(ln.lstrip()) for ln in nonblank)
        lines = [ln[common:] if ln.strip() else "" for ln in raw_lines]
        code = "\n".join(lines).strip("\n")
        if code.strip():
            blocks.append(code)
    return blocks