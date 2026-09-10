"""IR example 4: parameter binding.

Demonstrates carrying symbolic parameters into the IR, binding them
numerically and rebuilding an executable circuit.
"""

import numpy as np

from microquantum import BindParameters, Parameter, QuantumCircuit, from_ir


def main():
    theta = Parameter("theta")

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(theta, 1)
    qc.cx(0, 1)
    qc.rz(theta, 0)

    ir = qc.to_ir()
    print("=== Parameterized circuit -> IR ===")
    print(ir)
    print("unbound parameters:", sorted(p.name for p in ir.parameters))

    bound_ir = BindParameters({"theta": np.pi / 3}).run(ir)
    print("\n=== After BindParameters({'theta': pi/3}) ===")
    print(bound_ir)
    print("is_parameterized:", bound_ir.is_parameterized)

    executable = from_ir(bound_ir)
    print("\n=== Rebuilt executable circuit ===")
    print(executable.qasm(header=False))
    amp = executable.run().amplitudes
    print("normalized:", np.isclose(np.linalg.norm(amp), 1.0))


if __name__ == "__main__":
    main()