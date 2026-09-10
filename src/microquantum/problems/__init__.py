"""Generic problem abstractions for algorithm-driven workflows.

``Problem`` describes *what* to solve; algorithms describe *how* to solve
it.  The hierarchy is intentionally small:

* :class:`Problem` -- the universal base.
* :class:`SamplingProblem` -- request samples of a distribution.
* :class:`OptimizationProblem` -- minimize over binary variables.
* :class:`HamiltonianProblem` -- eigenstates of a Hamiltonian.
* :class:`EigenvalueProblem` -- seek the ``k`` lowest eigenvalues.
* :class:`SearchProblem` -- unstructured database search.
"""

from .base import Problem, SamplingProblem
from .eigenvalue import (
    EigenvalueProblem,
    Hamiltonian,
    HamiltonianProblem,
    hamiltonian_expectation,
)
from .optimization import OptimizationProblem, standard_binary_encoding
from .search import SearchProblem

__all__ = [
    "Problem",
    "SamplingProblem",
    "OptimizationProblem",
    "HamiltonianProblem",
    "EigenvalueProblem",
    "SearchProblem",
    "Hamiltonian",
    "hamiltonian_expectation",
    "standard_binary_encoding",
]