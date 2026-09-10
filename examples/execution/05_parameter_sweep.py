"""Execution example 5: parameter sweep.

:func:`microquantum.run_parameter_sweep` runs the same circuit over many
parameter bindings — the building block for scanning rotation angles and
building cost landscapes.
"""

from microquantum import Parameter, QuantumCircuit, run_parameter_sweep


def main():
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(theta, 0)
    qc.h(0)  # h * rz(theta) * h rotates the |1> population by theta

    angles = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print("=== Sweep over theta (raw floats) ===")
    results = run_parameter_sweep(qc, angles, shots=2048, seed=21)
    print(" theta | P(|1>)")
    print(" ------|------")
    for angle, result in zip(angles, results, strict=False):
        prob = result.probabilities.get("1", 0.0)
        print(f" {angle:5.2f} |  {prob:.3f}")

    print("\n=== Sweep with explicit binding dicts (multi-parameter) ===")
    two = QuantumCircuit(1)
    two.rx(Parameter("a"), 0)
    two.ry(Parameter("b"), 0)
    sweeps = run_parameter_sweep(
        two,
        [{"a": 0.2, "b": 0.3}, {"a": 0.4, "b": 0.5}],
        shots=1024,
        seed=2,
    )
    for result in sweeps:
        print(f"  {result.metadata['parameter_bindings']} -> "
              f"top {sorted(result.counts.items(), key=lambda kv: -kv[1])[0]}")


if __name__ == "__main__":
    main()