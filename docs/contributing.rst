Contributing
============

Thanks for contributing to the open-source ``microquantum`` SDK!  This
project is MIT licensed and deliberately dependency-light (NumPy only).
Industry-specific vertical solvers are developed outside this repository and
are out of scope here.

Ground rules
------------

* **No new hard dependencies** — new features build on NumPy. Optional
  capabilities (e.g. GPU) belong behind lazy imports and
  ``[project.optional-dependencies]``.
* **No private/proprietary code** — this repository is the public SDK. Do not
  add domain-specific business analytics, platform code, credentials, or
  references to code from outside this repository.
* **Keep the contract stable** — changes to serialized result types are
  versioned; update ``CHANGELOG.md``.

Getting started
---------------

.. code-block:: console

   uv sync --group dev

Before you submit changes
-------------------------

1. **Tests** — add or update tests under ``tests/``:

   .. code-block:: console

      uv run pytest tests/

2. **Type check** — the SDK targets zero mypy errors:

   .. code-block:: console

      uv run mypy src/microquantum/ --ignore-missing-imports

3. **Lint** — keep new code ruff-clean:

   .. code-block:: console

      uv run ruff check src tests examples

4. **Docs** — update the Sphinx docs under ``docs/`` for public API changes:

   .. code-block:: console

      uv run sphinx-build -W --keep-going -b html docs docs/_build/html

Branch workflow (required)
--------------------------

1. Work on the ``develop`` branch.
2. Commit your changes with a concise, imperative message.
3. Push ``develop``.
4. Merge into ``main`` with a merge commit:

   .. code-block:: console

      git switch main
      git merge develop --no-ff
      git push origin main

Versioning and releases
-----------------------

The version is read dynamically from ``microquantum/__init__.py``
(``__version__``).  Bump it there and in ``docs/conf.py``, and add a
``CHANGELOG.md`` entry.  Releases happen from ``main`` with a ``v<version>``
tag.  No release to PyPI without: push + green CI + ``v<version>`` tag +
TestPyPI check first.