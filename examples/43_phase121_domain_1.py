"""Phase 121 W3 tour: domain methods I."""

from __future__ import annotations

import numpy as np

from microquantum.algorithms import QPEPhaseFilter, QuantumCounting
from microquantum.analysis import HypothesisTest, bootstrap_ci
from microquantum.optimization import PUBOBuilder, QUBOBuilder, qubo_to_pauli_sum
from microquantum.optimization.ising_pauli import pubo_to_qubo_projection
from microquantum.problems import SearchProblem


def main() -> None:
    result = QuantumCounting().count(2, [0, 3])
    print("count:", result.estimated_count, "fraction:", round(result.fraction, 4))
    print("solve:", QuantumCounting().solve(SearchProblem(num_qubits=2, target=1)).estimated_count)
    print("angles:", QPEPhaseFilter(kappa=1.0).rotation_angles(2))

    outcome = HypothesisTest(alpha=0.05, permutations=100, seed=0).compare(
        {"00": 90, "11": 10}, {"00": 10, "11": 90}
    )
    print("significant:", outcome.significant)
    low, high = bootstrap_ci([1.0, 2.0, 3.0, 4.0], resamples=100, seed=0)
    print("bootstrap CI:", round(low, 3), round(high, 3))

    builder = PUBOBuilder(2)
    builder.add_quadratic(0, 1, -2.0)
    builder.add_term((0, 1), 0.5)
    print("pubo energy:", builder.build().energy(np.array([1, 1])))
    qb = QUBOBuilder(2)
    qb.add_quadratic(0, 1, 2.0)
    print("pauli qubits:", qubo_to_pauli_sum(qb.build()).num_qubits)
    cubic = PUBOBuilder(3)
    cubic.add_term((0, 1, 2), 4.0)
    print("dropped:", pubo_to_qubo_projection(cubic.build()).metadata["dropped_higher_order"])


if __name__ == "__main__":
    main()
