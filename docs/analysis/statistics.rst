Statistics
==========

The :mod:`microquantum.analysis.statistics` module provides small,
dependency-light statistical helpers for repeated measurements.  They accept
any iterable of real numbers (lists, tuples, NumPy arrays) and reject
non-numeric / non-finite input with :class:`ValueError`.

Functions
---------

* :func:`~microquantum.mean` — arithmetic mean.
* :func:`~microquantum.variance` — ``ddof=0`` population variance, ``ddof=1``
  sample variance.
* :func:`~microquantum.standard_deviation` — sqrt of variance.
* :func:`~microquantum.standard_error` — *sample* standard error of the mean
  (``sample std / sqrt(n)``; requires >= 2 samples).
* :func:`~microquantum.confidence_interval` — normal-approximation CI for the
  mean (presets 0.90 / 0.95 / 0.99, or an explicit ``z``).
* :func:`~microquantum.minimum` / :func:`~microquantum.maximum` /
  :func:`~microquantum.count`.

Example
-------

.. code-block:: python

   from microquantum import (
       confidence_interval,
       mean,
       standard_deviation,
       standard_error,
       variance,
   )

   shots = [0.501, 0.504, 0.498, 0.501, 0.499, 0.497]
   print(mean(shots))                # ~0.500
   print(variance(shots))            # population variance
   print(variance(shots, ddof=1))    # sample variance
   print(standard_deviation(shots, ddof=1))
   print(standard_error(shots))
   print(confidence_interval(shots, confidence=0.95))

   print(mean([]))                   # ValueError: must not be empty
   print(variance([1.0]))            # ValueError: need > ddof samples

Usage
-----

These helpers back :class:`~microquantum.SamplingAnalysis`,
:class:`~microquantum.ExpectationAnalysis` and
:class:`~microquantum.ResultAggregator`; they are also exported at the package
top level for direct use.