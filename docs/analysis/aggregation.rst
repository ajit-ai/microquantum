Aggregation
===========

:class:`~microquantum.ResultAggregator` groups raw execution results —
without ever losing them.  The pattern is always:

``raw results -> aggregation -> derived analysis``
(never ``raw results -> replace with summary``).

Usage
-----

.. code-block:: python

   from microquantum import ResultAggregator

   # result: an ExperimentResult (or a sequence of records/results)
   agg = ResultAggregator(result)
   print(agg.record_count)                     # number of records

   # group by dotted-path accessor (attribute or nested field)
   by_backend = agg.group_by("backend")
   by_status  = agg.group_by("status")
   by_theta   = agg.group_by("parameter_bindings.theta")
   counts     = agg.group_counts(by_theta)

   # convenience groupings
   agg.group_by_parameter("theta")
   agg.group_by_backend()
   agg.group_by_status()

Callables as accessors
----------------------

A callable ``record -> value`` works too:

.. code-block:: python

   def key(record):
       return record.metadata.get("sweep_name", "fixed")

   groups = agg.group_by(key)

Derived summaries
-----------------

* ``mean_expectation(groups, "Z")`` — mean of an expectation label per group
  (groups without the label are omitted).
* ``parameter_expectations(parameter, "Z")`` / ``expectation_keys()`` —
  parameter-to-expectation surfaces.
* ``to_dict(accessor="backend")`` / ``to_json(...)`` — JSON-safe output that
  keeps counts and per-group summaries.

Design guarantee
----------------

Raw records are kept as-is (``agg.records`` is a read-only view of the
original objects), so any group can be re-analysed with the full result data
later.