"""Execution example 6: a classical -> quantum -> classical loop.

:func:`microquantum.run_hybrid` drives a generic round-trip: classical state
is turned into a quantum measurement, and that measurement is folded back
into the classical state.  This example scans a rotation angle and reports
how the measured |1> population tracks it — deliberately not any specific
algorithm (no VQE/QAOA-style machinery here).
"""

from microquantum import ExecutionPlan, QuantumCircuit, run_hybrid


def build_workload(state, step):
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(state["angle"], 0)
    qc.h(0)  # h * rz(angle) * h rotates the |1> population by angle
    return ExecutionPlan.from_circuit(
        qc,
        shots=2048,
        seed=4,
        metadata={"angle": state["angle"]},
    )


def fold_back(state, step, result):
    return {
        "angle": state["angle"] + 0.5,
        "best": max(state["best"], result.probabilities.get("1", 0.0)),
    }


def main():
    print("=== Hybrid loop: 6 classical-quantum rounds ===")
    state = {"angle": 0.0, "best": 0.0}
    results = run_hybrid(6, build_workload, fold_back, state, seed=4)

    for result in results:
        idx = result.metadata["hybrid_step"]
        angle = result.metadata["angle"]
        prob = result.probabilities.get("1", 0.0)
        print(f"step {idx}: angle={angle:.1f} P(|1>)={prob:.3f}")

    best = max(r.probabilities.get("1", 0.0) for r in results)
    print(f"\nbest P(|1>) observed: {best:.3f}")


if __name__ == "__main__":
    main()