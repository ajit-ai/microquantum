"""Phase 121 W4 tour: domain methods II."""

from __future__ import annotations

from microquantum.chemistry import ActiveSpace, FermionicOp, jordan_wigner
from microquantum.mitigation import CliffordDataRegression
from microquantum.qec import LookupDecoder, SteaneCode, Syndrome
from microquantum.qml import DataReuploadingClassifier, QuantumKernel, kernel_alignment


def main() -> None:
    clf = DataReuploadingClassifier(num_features=1, layers=1)
    print("predictions:", clf.predict([[0.0], [3.0]]).predictions)
    print("accuracy:", round(clf.score([[0.0], [3.0]], [0, 1]), 3))
    matrix = QuantumKernel().evaluate([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]])
    print("alignment:", round(float(kernel_alignment(matrix, [0, 1])), 4))

    code = SteaneCode()
    print("decode:", code.decode_syndrome([0, 0, 1, 1, 0, 0]))
    decoder = LookupDecoder(table={(0, 0, 1): [(0, "X")]}, code_name="steane")
    print("lookup:", decoder.decode(Syndrome(bits=(0, 0, 1), code_name="steane")))

    selection = ActiveSpace(num_core_orbitals=1, num_active_orbitals=2).select(6, 4)
    print("active qubits:", selection.num_qubits)
    hopping = FermionicOp({(("+", 0), ("-", 1)): 1.0, (("+", 1), ("-", 0)): 1.0})
    print("jw terms:", jordan_wigner(hopping, 2).num_terms)

    cdr = CliffordDataRegression()
    summary = cdr.train([0.5, 0.7, 0.9], [0.6, 0.8, 1.0])
    print("r_squared:", round(summary["r_squared"], 6), "mitigated:", round(cdr.mitigate(0.7), 6))


if __name__ == "__main__":
    main()
