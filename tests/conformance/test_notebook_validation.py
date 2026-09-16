"""Conformance - the consolidated developer notebook is valid and executable.

The notebook ``examples/MicroQuantum_Examples.ipynb`` is a convenience layer
over the authoritative ``.py`` examples: every code cell executes the
corresponding example file through its ``run_example`` helper.  These tests
make sure the notebook (1) is a well-formed notebook, (2) covers the whole
official example corpus, (3) never references a missing example file, and
(4) actually executes when its code cells are run in order (stdlib only, no
Jupyter runtime dependency).
"""

from __future__ import annotations

import json

import conformance_helpers as h
import pytest

_NOTEBOOK = h.EXAMPLES_DIR / "MicroQuantum_Examples.ipynb"

_EXPECTED_SECTIONS = [
    "1. Setup",
    "2. Core circuits & algorithms",
    "3. Parameters & gradients",
    "4. Simulators",
    "5. IR & compilation",
    "6. Execution runtime",
    "7. Backends & providers",
    "8. Problems & optimization",
    "9. Algorithms",
    "10. Variational algorithms",
    "11. Search",
    "12. Mathematical primitives",
    "13. Analysis",
    "14. Extending the SDK",
]


def _load_notebook() -> dict:
    return json.loads(_NOTEBOOK.read_text(encoding="utf-8"))


def _cell_blocks(nb: dict):
    markdown = [c for c in nb["cells"] if c["cell_type"] == "markdown"]
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return markdown, code


def _cell_text(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)


@pytest.fixture(scope="module")
def notebook() -> dict:
    return _load_notebook()


def test_notebook_file_exists() -> None:
    assert _NOTEBOOK.is_file(), f"missing notebook: {_NOTEBOOK}"


def test_notebook_is_valid_json(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["nbformat_minor"], int)
    assert isinstance(notebook["cells"], list) and notebook["cells"]


def test_cells_have_valid_shapes(notebook: dict) -> None:
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        assert isinstance(cell.get("source", ""), (str, list))
        if cell["cell_type"] == "code":
            assert isinstance(cell.get("execution_count"), (int, type(None)))
            assert isinstance(cell.get("outputs"), list)


def test_expected_sections_present(notebook: dict) -> None:
    markdown, _ = _cell_blocks(notebook)
    headings = [ln for c in markdown for ln in _cell_text(c).splitlines()
                if ln.startswith("## ")]
    for section in _EXPECTED_SECTIONS:
        assert f"## {section}" in headings, f"missing section: {section}"


def test_table_of_contents_present(notebook: dict) -> None:
    markdown, _ = _cell_blocks(notebook)
    toc = _cell_text(markdown[0]) if markdown else ""
    assert "## Table of contents" in toc
    for section in _EXPECTED_SECTIONS:
        assert f"- {section}" in toc, f"TOC missing entry: {section}"


def test_notebook_covers_every_official_example(notebook: dict) -> None:
    markdown, code = _cell_blocks(notebook)
    md_text = "\n".join(_cell_text(c) for c in markdown)
    code_lines = set()
    for cell in code:
        for ln in _cell_text(cell).splitlines():
            if ln.startswith('run_example(r"'):
                code_lines.add(ln.strip())
    for example in h.all_example_files():
        rel = str(example.relative_to(h.REPO_ROOT)).replace("\\", "/")
        assert f"**Source example:** ``{rel}``" in md_text, (
            f"notebook missing marker for {rel}"
        )
        assert f'run_example(r"{rel}")' in code_lines, (
            f"notebook missing code cell for {rel}"
        )


def test_no_missing_file_references(notebook: dict) -> None:
    markdown, _ = _cell_blocks(notebook)
    for cell in markdown:
        for ln in _cell_text(cell).splitlines():
            if "**Source example:**" in ln:
                marker = ln.split("**Source example:**", 1)[1].strip()
                rel = marker.strip("` ")
                assert (h.REPO_ROOT / rel).is_file(), (
                    f"notebook references missing file: {rel}"
                )


def test_notebook_code_cells_execute(monkeypatch, notebook: dict) -> None:
    _, code = _cell_blocks(notebook)
    assert code, "notebook has no code cells"
    monkeypatch.chdir(h.REPO_ROOT)
    namespace: dict = {}
    for i, cell in enumerate(code):
        try:
            exec(compile(_cell_text(cell), f"<notebook cell {i}>", "exec"),
                 namespace)
        except Exception as exc:  # noqa: BLE001 - failing cell must fail the gate
            raise AssertionError(
                f"notebook code cell {i} failed:\n{_cell_text(cell)}"
            ) from exc