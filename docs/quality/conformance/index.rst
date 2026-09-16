Conformance
===========

The public API conformance gate for the MicroQuantum SDK is **complete**.
The suite lives in ``tests/conformance/`` and runs with the regular pytest
command; no tests are skipped, expected to fail, quarantined, or
suppressed.

The :doc:`microquantum A-Z Conformance Report <CONFORMANCE_AZ_REPORT>` is the
permanent, historical verification record of that gate. It states exactly
what was exercised, what was fixed as a result, and what remains as future
API cleanup. It reflects the state of the gate at the time it ran and does
**not** claim any form of external certification.

The report is a point-in-time snapshot. Quality gates continue to run on
every development phase; a new report is recorded whenever a major gate is
re-executed.

.. toctree::
   :maxdepth: 1

   CONFORMANCE_AZ_REPORT
   CONFORMANCE_MQ14_REPORT
   CONFORMANCE_MQ15_REPORT