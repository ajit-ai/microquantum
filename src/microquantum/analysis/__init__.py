"""Backend-independent result analysis (MQ-07).

The analysis layer operates on **result objects** — never on backends — so it
applies equally to simulator, mock and hardware outputs:

* :class:`SamplingAnalysis` — counts/samples: probabilities, most likely
  outcome, entropy, marginal distributions, observable mean/variance.
* :class:`ExpectationAnalysis` — repeated labeled expectation values: mean,
  variance, standard error, parameter→expectation mapping.
* :class:`StateAnalysis` — statevector/density-matrix inspection
  (normalization, purity, most probable state, diagonal observables).
* :class:`ResultAggregator` — grouping by parameter bindings / backend /
  metadata while preserving the raw results.
* :mod:`statistics` helpers — :func:`mean` :func:`variance`
  (:func:`standard_deviation`, :func:`standard_error`,
  :func:`confidence_interval`, :func:`minimum`, :func:`maximum`,
  :func:`count`) with an explicit population-vs-sample distinction.

This is a pure SDK analytics foundation: no database, dashboard, CLI or
visualization dependency.  Every public entry point also exposes JSON-safe
``to_dict()`` / ``to_json()`` output for downstream tooling.
"""

from .aggregation import ResultAggregator
from .expectation import ExpectationAnalysis
from .sampling import SamplingAnalysis
from .state import StateAnalysis
from .statistics import (
    confidence_interval,
    count,
    maximum,
    mean,
    minimum,
    standard_deviation,
    standard_error,
    variance,
)

__all__ = [
    "SamplingAnalysis",
    "ExpectationAnalysis",
    "StateAnalysis",
    "ResultAggregator",
    "mean",
    "variance",
    "standard_deviation",
    "standard_error",
    "confidence_interval",
    "minimum",
    "maximum",
    "count",
]