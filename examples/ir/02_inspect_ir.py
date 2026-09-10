"""IR example 2: inspecting IR.

Demonstrates the IRCircuit inspection API: qubits, classical bits,
parameters, depth, gate counts, measurements and JSON serialization
without ever invoking NumPy.
"""

import json

from microquantum import DynamicCircuit, QuantumCircuit, to_ir_dynamic


def main():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    ir = qc.to_ir(include_terminal_measurements=True)

    print("=== Static circuit IR ===")
    print("qubits:", sorted(ir.qubits))
    print("classical bits:", sorted(ir.cbits))
    print("gates:", ir.gate_names())
    print("depth:", ir.depth)
    print("total gates:", ir.num_gates)
    print("measurements:", [(m.qubit, m.classical) for m in ir.measurements()])
    json.loads(ir.to_json())
    print("serializes to JSON: ok")

    dc = DynamicCircuit(2, 2)
    dc.h(0)
    dc.measure(0, 0)
    dc.reset(1)
    dc.classical_if(0, lambda d: d.x(1))
    dir_ir = to_ir_dynamic(dc)

    print("\n=== Dynamic circuit IR ===")
    print(dir_ir)
    print("has resets:", dir_ir.has_reset())
    print("has conditions:", dir_ir.has_conditions())


if __name__ == "__main__":
    main()