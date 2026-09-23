"""Phase 121 W6 tour: foundations."""

from __future__ import annotations

import math

import numpy as np

from microquantum.core.gates import ControlledUnitary, H
from microquantum.core.information import concurrence, entanglement_entropy
from microquantum.stdlib import dicke_state, graph_state, gray_code


def main() -> None:
    print("gray:", gray_code(2))
    print("dicke norm:", dicke_state(3, 1).is_normalized)
    print("graph norm:", graph_state([(0, 1), (1, 2)], 3).is_normalized)
    print("cu qubits:", ControlledUnitary(H().to_matrix()).num_qubits)
    bell = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    print("entropy:", round(entanglement_entropy(bell, [0]), 6))
    print("concurrence:", round(concurrence(bell), 6))


if __name__ == "__main__":
    main()
