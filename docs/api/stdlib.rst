Standard Library
================

The **System Standard Library** (``microquantum.stdlib``, Phase 117) is a
stable public layer of foundational, dependency-light building blocks shared
by user programs, the runtime and the compiler.  It adds no application
behaviour and never duplicates APIs defined elsewhere in the SDK.

* :mod:`microquantum.stdlib.bits` — MSB-first bitstring/integer conversions
  and Hamming distance/weight, matching the SDK's measurement-bitstring
  convention.
* :mod:`microquantum.stdlib.numbers` — angle normalization
  (:func:`~microquantum.stdlib.numbers.mod_2pi`,
  :func:`~microquantum.stdlib.numbers.wrap_angle`) and modulo-``2*pi``
  rotation comparisons.
* :mod:`microquantum.stdlib.states` — common state factories built on
  :class:`~microquantum.StateVector`:
  :func:`~microquantum.stdlib.states.basis_state`,
  :func:`~microquantum.stdlib.states.uniform_superposition`,
  :func:`~microquantum.stdlib.states.bell_state`,
  :func:`~microquantum.stdlib.states.ghz_state` and
  :func:`~microquantum.stdlib.states.w_state`.

Every function is importable both from ``microquantum.stdlib`` and from the
top-level package (``from microquantum import ghz_state``).

Generated reference (by module)
-------------------------------

.. toctree::
   :hidden:

   /api/microquantum/stdlib/index
   /api/microquantum/stdlib/bits/index
   /api/microquantum/stdlib/numbers/index
   /api/microquantum/stdlib/states/index