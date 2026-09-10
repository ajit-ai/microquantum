Development
===========

This page is the developer quick start for working on the SDK itself.

Setup
-----

.. code-block:: console

   python -m venv .venv
   # activate, then:
   pip install -U uv
   uv sync --group dev          # installs numpy + dev deps

Verify
------

.. code-block:: console

   uv run pytest tests/ -q                       # full test suite
   uv run pytest tests/ -q --cov=microquantum    # with coverage
   uv run mypy src/microquantum/ --ignore-missing-imports
   uv run ruff check src/ tests/ examples/
   uv run ruff format --check src/ tests/

Build & install
---------------

.. code-block:: console

   uv build                                   # → dist/ (wheel + sdist)
   uv pip install --no-deps dist/microquantum-0.4.0-py3-none-any.whl

Docs
----

.. code-block:: console

   uv run sphinx-build -W --keep-going -b html docs docs/_build/html

Docstrings use Google format (``Args`` / ``Returns`` / ``Raises``) and are the
source of the autoapi reference — keep signatures and docstrings accurate.

Branch workflow
---------------

This repository follows a **develop → main** flow:

1. Commit work on ``develop``.
2. Push ``develop``.
3. Merge into ``main`` (``git switch main``; ``git merge develop --no-ff``).
4. Push ``main``.

Do not leave local-only commits at the end of a session.

Testing conventions
-------------------

* Tests live in ``tests/`` and are organized per subsystem.
* New public API needs tests covering happy paths, validation failures, and
  JSON round-trips.
* Examples in ``examples/`` are executed in CI against the installed package —
  keep them runnable.