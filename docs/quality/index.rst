Quality
=======

MicroQuantum runs a standing suite of automated quality gates on every
development phase: the full pytest suite (unit + conformance), mypy, ruff,
and the Sphinx documentation build with warnings treated as errors.

Verification is driven by the permanent conformance suite under
``tests/conformance/``. The historical record of the completed public
conformance gate is preserved in this section.

.. toctree::
   :maxdepth: 2
   :caption: Conformance

   conformance/index