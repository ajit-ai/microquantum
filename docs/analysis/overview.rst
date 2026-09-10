Overview
========

Analysis is backend-independent and consensus-based: it consumes the existing
result/record contracts rather than inventing new output formats.  The four
analysers share a consistent JSON-safe surface (``to_dict()`` /
``to_json()``).

The families
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Analyser
     - What it computes
   * - :class:`~microquantum.SamplingAnalysis`
     - Measurement counts: outcome probabilities, most/least likely outcome,
       Shannon entropy, marginals, observable mean/variance/std.
   * - :class:`~microquantum.ExpectationAnalysis`
     - The ``BackendResult.expectations`` ``{label: value}`` contract:
       per-label mean / variance / std / standard error and a
       parameter-to-expectation mapping.
   * - :class:`~microquantum.StateAnalysis`
     - Statevectors (normalization, probabilities, most probable state,
       diagonal observables) and density matrices (trace, purity, diagonal
       measurement probabilities).
   * - :class:`~microquantum.ResultAggregator`
     - Group records by parameter bindings, backend, status or dotted-path
       accessors — always preserving the original records.
   * - :mod:`microquantum.analysis.statistics`
     - Reusable population-vs-sample variance (``ddof``), standard error,
       confidence intervals and min/max/count.
   * - :mod:`microquantum.analysis.aggregation`
     - Grouping helpers that map raw results -> aggregation -> derived
       analysis (never replacing raw results with summaries).

Inputs accepted
---------------

Analysers accept a :class:`~microquantum.BackendResult`, an
:class:`~microquantum.ExecutionRecord`, an
:class:`~microquantum.ExperimentResult`, raw dictionaries, or the serialized
``to_dict()`` form — so analysis works no matter how the results were
produced or transported.

Design principle
----------------

``raw results -> aggregation -> derived analysis``, never
``raw results -> replace with summary``.  Experiments stay re-analysable
without re-execution.