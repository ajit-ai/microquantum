"""IR example 7: custom passes.

Demonstrates extending the compilation foundation with a user-defined
IRPass and slotting it into the standard pipeline.
"""

from microquantum import Compiler, Gate, IRCircuit, IRPass, QuantumCircuit


class ZeroRyRemover(IRPass):
    """Custom pass: drop RY(0) gates (angle exactly zero)."""

    name = "drop-zero-ry"

    def run(self, ir: IRCircuit) -> IRCircuit:
        kept = [
            op
            for op in ir.operations
            if not (
                isinstance(op, Gate)
                and op.name == "ry"
                and tuple(op.params) == (0.0,)
            )
        ]
        return IRCircuit(
            num_qubits=ir.num_qubits,
            num_classical_bits=ir.num_classical_bits,
            name=ir.name,
            operations=kept,
            metadata=dict(ir.metadata),
        )


def main():
    qc = QuantumCircuit(1)
    qc.ry(0.0, 0)
    qc.ry(0.25, 0)

    print("=== Source IR ===")
    print(qc.to_ir())

    result = Compiler(optimization_level=0).compile(
        qc,
        passes=[ZeroRyRemover()],
    )
    print("\n=== After custom pass 'drop-zero-ry' ===")
    print(result.result)
    print("passes applied:", result.passes_applied)


if __name__ == "__main__":
    main()