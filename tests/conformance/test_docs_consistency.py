"""Conformance: docs code blocks and cross-references must match the SDK.

Two checks:
1. Executor: every ``.. code-block:: python`` in the hand-written docs runs
   end-to-end with the installed package, sharing one namespace per page.
2. Consistency: every ``microquantum[.submodule].name`` dotted reference used
   inline in the docs resolves to a real importable symbol.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
AUTOAPI = DOCS / "api" / "microquantum"


def iter_handwritten_rst():
    for path in sorted(DOCS.rglob("*.rst")):
        if AUTOAPI.resolve() in path.resolve().parents:
            continue
        yield path


def extract_code_blocks(text: str):
    blocks = []
    in_block = False
    indent = ""
    current: list[str] = []
    for line in text.splitlines():
        if re.match(r"^\s*\.\.\s+code-block::\s+python\s*$", line):
            in_block = True
            indent = None
            current = []
            continue
        if in_block:
            if indent is None:
                if not line.strip():
                    continue
                m = re.match(r"^(\s+)", line)
                indent = m.group(1) if m else ""
                current.append(line[len(indent) :])
                continue
            if line.startswith(indent) or (line.strip() == "" and current):
                current.append(line[len(indent) :])
            else:
                blocks.append("\n".join(current).rstrip())
                in_block = False
    if in_block and indent is not None:
        blocks.append("\n".join(current).rstrip())
    return blocks


FRAGMENT_MARKERS = (
    "  ...",
    "\t...",
    "...",
)


def _page_groups():
    """Group blocks by page, skipping explicit fragment markers."""
    groups: dict[str, list[str]] = {}
    for path in iter_handwritten_rst():
        key = path.relative_to(ROOT).as_posix()
        for code in extract_code_blocks(path.read_text(encoding="utf-8")):
            lines = code.splitlines()
            if lines and lines[0].strip() == "...":
                continue
            if any(re.match(r"^\s*\.\.\.\s*$", ln) for ln in lines):
                continue
            groups.setdefault(key, []).append(code)
    return groups


PAGES = sorted(_page_groups())


@pytest.mark.parametrize("page", PAGES)
def test_docs_page_code_blocks_run(page: str) -> None:
    blocks = _page_groups()[page]
    script_lines = ["import sys"]
    for i, code in enumerate(blocks):
        script_lines.append(
            "try:\n"
            f"    exec({code!r})\n"
            "except BaseException:\n"
            f"    print('@@BLOCK {i} {page}@@', flush=True)\n"
            "    raise\n"
        )
    script = "\n".join(script_lines)
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        cwd=str(DOCS),
        capture_output=True,
        text=True,
        timeout=240,
    )
    combined = (proc.stdout + proc.stderr).strip()
    assert proc.returncode == 0, (
        f"docs page {page} failed:\n{combined[-3000:]}"
    )
    assert "@@BLOCK" not in proc.stdout


# ---------------------------------------------------------------------------
# Doc reference consistency: dotted microquantum names must resolve
# ---------------------------------------------------------------------------

DOTTED = re.compile(r"microquantum(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


@pytest.mark.parametrize(
    "page", [p for p in PAGES if "api" not in p]
)
def test_doc_dotted_references_resolve(page: str) -> None:
    import importlib

    text = (ROOT / page).read_text(encoding="utf-8")
    seen = set()
    for match in DOTTED.finditer(text):
        ref = match.group(0)
        if ref in seen:
            continue
        seen.add(ref)
        parts = ref.split(".")
        assert len(parts) >= 2
        obj = importlib.import_module(parts[0])
        try:
            for attr in parts[1:]:
                obj = getattr(obj, attr)
        except (AttributeError, ImportError) as exc:
            raise AssertionError(f"broken doc reference `{ref}` in {page}: {exc}") from None


@pytest.mark.parametrize("name", ["microquantum"])
def test_top_level_package_imports(name: str) -> None:
    import importlib

    importlib.import_module(name)