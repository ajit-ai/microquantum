Installation
============

Requirements
------------

* Python **3.10, 3.11, 3.12, or 3.13**
* NumPy ``>= 1.20`` (installed automatically)

MicroQuantum is pure Python + NumPy and is platform-independent at the package
level. It is designed for **Windows, Linux, macOS and BSD** systems with a
compatible Python + NumPy environment; see :doc:`/releases/platform-support`
for the CI-verified and manual verification status of each platform.

From PyPI
---------

.. code-block:: bash

   pip install microquantum

With `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv pip install microquantum

Optional GPU acceleration (NumPy remains the default; CuPy is opt-in, and the
``gpu`` extra is only available on platforms where CuPy ships wheels):

.. code-block:: bash

   pip install "microquantum[gpu]"

Verify the installation (on every platform):

.. code-block:: bash

   python -c "import microquantum; print(microquantum.__version__)"

You should see ``1.1.0``. When the ``microquantum`` entry point is on your
``PATH``:

.. code-block:: bash

   microquantum --version

Linux
-----

*(CI-verified on GitHub Actions ``ubuntu-latest``)*

Install a supported CPython on your distribution, typically already present or
available from the package manager, for example:

.. code-block:: bash

   # Debian / Ubuntu
   sudo apt install python3 python3-venv python3-pip

   # Fedora / RHEL / CentOS
   sudo dnf install python3 python3-pip

Create and activate a virtual environment, then install:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install microquantum

Or install into the user site without a virtual environment:

.. code-block:: bash

   python3 -m pip install --user microquantum

Windows
-------

*(CI-verified on GitHub Actions ``windows-latest``)*

Install a supported CPython for Windows from
`python.org <https://www.python.org/downloads/windows/>`_ (or the Windows
Store), then use the Microsoft Store / python.org ``python`` launcher ``py`` to
create a virtual environment:

.. code-block:: powershell

   py -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install microquantum

In ``cmd.exe`` the activation script is ``.venv\Scripts\activate.bat`` instead.
Installing when the environment is active places the ``microquantum.exe``
console script on ``PATH``, so verify with:

.. code-block:: powershell

   python -c "import microquantum; print(microquantum.__version__)"
   microquantum --version

macOS
-----

*(CI-verified on GitHub Actions ``macos-latest``)*

Install a supported CPython for macOS, for example with
`Homebrew <https://brew.sh/>`_:

.. code-block:: bash

   brew install python

Then create and activate a virtual environment and install:

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install microquantum

Apple Silicon and Intel macOS are covered by the CI runners used; the SDK is
pure Python + NumPy so it uses the prebuilt NumPy wheels for the installed
architecture.

BSD
---

*(Not CI-verified; supported by design — see
:doc:`/releases/platform-support`)*

On a BSD system (such as FreeBSD or OpenBSD) install a supported CPython from
the ports/packages system, for example:

.. code-block:: bash

   # FreeBSD
   sudo pkg install python311 py311-pip

   # OpenBSD
   doas pkg_add python-3.11 py3-pip

Then create a virtual environment and install with pip:

.. code-block:: bash

   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install microquantum

MicroQuantum is portable Python + NumPy and loads pure Python wheels
(``py3-none-any``), so no BSD-specific binaries are needed. NumPy must be able
to install on the BSD release in use; a source build is used where no wheel is
available for the platform/architecture.

From source
-----------

.. code-block:: bash

   git clone https://github.com/ajit-ai/microquantum.git
   cd microquantum
   pip install -e .

Development tooling (tests, lint, type check, docs) is managed with uv:

.. code-block:: bash

   uv sync --group dev

Documentation build extras:

.. code-block:: bash

   pip install "microquantum[docs]"

Next: :doc:`quickstart`.