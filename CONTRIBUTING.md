# Contributing to microquantum

Thanks for contributing to the open-source `microquantum` SDK!

This project is MIT licensed and deliberately dependency-light (NumPy only).
Industry-specific vertical solvers are developed outside this repository and are
out of scope here.

## Ground rules

- **No new hard dependencies.** New features should build on NumPy. Optional
  capabilities (e.g. GPU) belong behind lazy imports and `[project.optional-dependencies]`.
- **No private/proprietary code.** This repository is the public SDK. Do not add
  domain-specific business analytics, platform code, credentials, or references
  to code from outside this repository.
- **Keep the contract stable.** Changes to `microquantum.analytics.result.Result`
  or other serialized result types are versioned; update `CHANGELOG.md`.

## Getting started

```bash
uv sync --group dev
```

## Before you submit changes

1. **Tests** — add or update tests under `tests/`:
   ```bash
   uv run pytest tests/
   ```
2. **Type check** — the SDK targets zero mypy errors:
   ```bash
   uv run mypy src/microquantum/ --ignore-missing-imports
   ```
3. **Lint** — keep new code ruff-clean:
   ```bash
   uv run ruff check src tests examples
   ```
4. **Docs** — update Sphinx docs under `docs/` for public API changes and
   verify a warning-free build (CI fails on warnings):
   ```bash
   uv run sphinx-build -E -W --keep-going -b html docs docs/_build/html
   ```

## Branch workflow (REQUIRED)

1. Work on the `develop` branch.
2. Commit your changes: `git commit -m "..."` (concise, imperative style).
3. Push `develop`: `git push origin develop`
4. Merge into `main` with a merge commit:
   ```bash
   git switch main
   git merge develop --no-ff
   git push origin main
   ```

## Versioning and releases

- The version is read dynamically from `microquantum/__init__.py` (`__version__`).
  Bump it there and in `docs/conf.py`, and add a `CHANGELOG.md` entry.
- Releases happen from `main` with a `v<version>` tag. No release to PyPI
  without: push + green CI + `v<version>` tag + TestPyPI check first.