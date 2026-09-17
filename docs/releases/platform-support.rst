.. _platform-support:

Platform Support
================

MicroQuantum is **platform-independent at the Python source/package level**:
it is pure Python with NumPy as the only hard runtime dependency, and it ships
a pure ``py3-none-any`` wheel that installs on any platform where the required
Python + NumPy environment is available.

This page records, per platform, what is claimed and how it is verified
(CI-verified, manually verified, or designed but not currently verified).

Supported by design
-------------------

The SDK is designed for:

* Windows
* Linux
* macOS
* BSD

There are no platform-specific dependencies in the package's core dependency
set; the ``microquantum`` CLI is implemented with the Python standard library
only.  Platform-specific or optional functionality (such as the CuPy-backed
``gpu`` extra) is opt-in and only installs on platforms where the
corresponding wheels exist.

CI-verified
-----------

GitHub Actions runs the consolidated CI pipeline (:doc:`/developer-guide/development`)
on the following hosted runners for Python 3.10 - 3.13:

.. list-table::
   :widths: 20 35 45
   :header-rows: 1

   * - Platform
     - Runner
     - Verification
   * - Linux
     - ``ubuntu-latest``
     - Install ``uv sync``, pytest, import check, CLI smoke
   * - Windows
     - ``windows-latest``
     - Install ``uv sync``, pytest, import check, CLI smoke
   * - macOS
     - ``macos-latest``
     - Install ``uv sync``, pytest, import check, CLI smoke

Each CI job verifies installation, ``import microquantum``, the ``microquantum``
CLI (``--version`` / ``info`` / ``backends --json``) and the full test suite.

BSD
---

BSD systems (FreeBSD, OpenBSD, NetBSD, ...) are supported at the portable
Python/package level: the pure ``py3-none-any`` wheel installs wherever CPython
>= 3.10 and NumPy >= 1.20 are available.

BSD is **not** currently CI-verified: GitHub-hosted runners do not provide a
native BSD image, and no self-hosted or external BSD CI infrastructure is
available to this project.  BSD compatibility is therefore documented as
designed/supported by the portable packaging model, reported separately from
the CI-verified platforms, and not upgraded to "CI verified" unless a real BSD
CI run completes successfully.

Verification guidance
---------------------

The normal installation path is the same on every platform — see
:doc:`/getting-started/installation`.  A minimal functional check is:

.. code-block:: console

   python -c "import microquantum; print(microquantum.__version__)"
   microquantum --version

:ref:`Known limitations <ga>` such as simulator-only execution and ``2**n``
memory growth are platform-independent properties of the SDK, not per-platform
restrictions.