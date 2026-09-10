Installation
============

Requirements
------------

* Python **3.10, 3.11, 3.12, or 3.13**
* NumPy ``>= 1.20`` (installed automatically)

From PyPI
---------

.. code-block:: bash

   pip install microquantum

With `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv pip install microquantum

Optional GPU acceleration (NumPy remains the default; CuPy is opt-in):

.. code-block:: bash

   pip install "microquantum[gpu]"

Verify the installation:

.. code-block:: bash

   python -c "import microquantum; print(microquantum.__version__)"

You should see ``0.4.0``.

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