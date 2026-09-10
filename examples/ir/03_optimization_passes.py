"""IR example 3: optimization passes.

Demonstrates the pure IRPass pipeline: identity removal, adjacent-inverse
cancellation and same-axis rotation fusion (level 0/1/2).
"""

from microquantum import (
    CancelAdjacentInverse,
    CombineRotations,
    IRPassManager,
    QuantumCircuit,
    RemoveIdentityGates,
    optimize,
)


def main():
    qc = QuantumCircuit(2)
    qc.t(0)
    qc.tdg(0)          # cancels with t
    qc.s(1)
    qc.h(0)
    qc.h(0)            # self-inverse pair
    qc.cx(0, 1)
    qc.cx(0, 1)        # self-inverse pair
    qc.rx(0.3, 1)
    qc.rx(0.4, 1)      # fused into one gate at level 2

    source = qc.to_ir()
    print("=== Source IR ===")
    print(source)
    print("gates:", source.gate_names(), "depth:", source.depth)

    for level in (0, 1, 2):
        optimized = optimize(source, level=level)
        print(f"\n--- optimization level {level} ---")
        print("gates:", optimized.gate_names())
        print("depth:", optimized.depth)

    pm = IRPassManager(
        [RemoveIdentityGates(), CancelAdjacentInverse(), CombineRotations()]
    )
    manual = pm.run(source)
    print("\n--- manual pipeline (same as level 2) ---")
    print("gates:", manual.gate_names())


if __name__ == "__main__":
    main()