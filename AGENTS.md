# AGENTS.md

## Intro
Public `microquantum` SDK (MIT). Published to PyPI as `microquantum`. Proprietary
vertical solvers live in the separate private `Microquantum-company` repository
and depend on this SDK from PyPI; the SDK must never import `microquantum_pro`
or `quantsmind`.

## Branch workflow (REQUIRED)
- After finishing any Phase, commit the work and make it available on **both branches**:
  1. Commit on `develop`
  2. Push `develop` to origin
  3. Merge `develop` into `main` (`git switch main; git merge develop --no-ff`)
  4. Push `main` to origin
- Do not leave local-only commits at the end of a session.

## Repository layout
- `src/microquantum/` — the package (`where = ["src"]` in root `pyproject.toml`).
- `tests/` — SDK test suite.
- `examples/` — demo scripts.
- `docs/` — Sphinx docs (autoapi reads `../src/microquantum`).

## Verification before finishing a Phase
- `uv sync --group dev`
- Tests: `uv run pytest tests/ -q`
- Type check: `uv run mypy src/microquantum/ --ignore-missing-imports`
- No release to PyPI without: push + green CI + `v<version>` tag + TestPyPI check first.