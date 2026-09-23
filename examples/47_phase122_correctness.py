"""Phase 122 W1+W2 tour: estimator correctness and contracts."""

from __future__ import annotations

import microquantum
from microquantum.algorithms import QuantumCounting
from microquantum.algorithms.amplitude_estimation import AmplitudeEstimation
from microquantum.core.circuit import QuantumCircuit
from microquantum.core.operators import Operator
from microquantum.problems import SearchProblem
from microquantum.providers.replay import ReplayTransport
from microquantum.runtime import runtime_info


def main() -> None:
    print("version single-sourced:", microquantum.__version__ == runtime_info().version)

    oracle = QuantumCircuit(1)
    oracle.append(Operator.Z(), [0])
    preparation = QuantumCircuit(1)
    preparation.h(0)
    result = AmplitudeEstimation(num_evaluation_qubits=4).estimate(1, preparation, oracle)
    print("ae amplitude:", round(result.estimated_amplitude, 6))

    counter = QuantumCounting()
    print("spectral count:", counter.count(2, [0, 3]).estimated_count)
    print("qpe count:", counter.estimate_count(2, [0, 3]).estimated_count)
    print("solve:", counter.solve(SearchProblem(num_qubits=2, target=1)).estimated_count)

    transport = ReplayTransport(script=[(200, {"ok": True})])
    print("replay:", transport(method="GET", url="https://x", headers=None, body=None))
    print("bitstring target:", SearchProblem(num_qubits=3, target="101").target)


if __name__ == "__main__":
    main()
