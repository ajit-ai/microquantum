"""IR example 5: dynamic circuit readiness.

Demonstrates mid-circuit measurement, qubit reset and classically
conditioned gates as first-class IR nodes, plus simulation of the
reset semantics.
"""

import numpy as np

from microquantum import DynamicCircuit, assert_valid, to_ir_dynamic


def main():
    dc = DynamicCircuit(2, 2)
    dc.h(0)
    dc.measure(0, 0)
    dc.h(1)
    dc.reset(0)               # re-prepare qubit 0 in |0>
    dc.classical_if(0, lambda d: d.x(1))

    ir = to_ir_dynamic(dc)
    print("=== Dynamic circuit -> IR ===")
    print(ir)
    assert_valid(ir)
    print("\nvalidation: clean")
    print("has_reset:", ir.has_reset())
    print("has_conditions:", ir.has_conditions())

    result = dc.run(seed=42)
    amp = np.asarray(result.final_state.amplitudes)
    print("\n=== Simulation (seed=42) ===")
    print("classical memory:", result.classical_memory)
    print("probabilities   :", np.abs(amp) ** 2)
    print("state normalized:", np.isclose(np.sum(np.abs(amp) ** 2), 1.0))


if __name__ == "__main__":
    main()