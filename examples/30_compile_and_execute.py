"""Example 30: Compile and execute through the runtime.

Demonstrates the three equivalent execution routes for a compiled program:

- Route A: compile explicitly, then ``backend.run(compiled)``.
- Route B: hand the :class:`CompilationResult` to the runtime via
  ``ExecutionPlan(compiled=...)`` (reuse instead of recompiling).
- Route C: let the runtime compile at run time via
  ``execute(circuit, optimization_level=...)``.

All three routes must produce the same seeded probability estimates.
"""

from __future__ import annotations

from microquantum import (
    Compiler,
    QuantumCircuit,
    StatevectorBackend,
    execute,
    to_ir,
)
from microquantum.runtime.plan import ExecutionPlan


def challenge() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(0)          # identity pair -> optimized away at level >= 1
    qc.cx(0, 1)
    return qc


def counts(backend, circuit, shots: int, seed: int):
    return sorted(backend.run(circuit, shots=shots, seed=seed).counts.items())


def normalize(counter) -> dict[str, float]:
    total = sum(counter.values())
    return {bits: count / total for bits, count in counter.items()}


def main() -> None:
    print("=== 30 Compile and execute through the runtime ===\n")

    qc = challenge()
    level = 1
    shots, seed = 2048, 7

    compiled = Compiler(optimization_level=level).compile(qc)
    rebuilt = compiled.circuit()
    print(f"source gates : {compiled.source.num_gates} -> "
          f"compiled gates: {compiled.result.num_gates}")
    print(f"passes       : {compiled.passes_applied}")
    print(f"IR round trip: gate names reconstruct to "
          f"{sorted(to_ir(rebuilt).gate_names())}\n")

    backend = StatevectorBackend()

    # Route A: explicit Compiler + backend.run(compiled)
    a = counts(backend, rebuilt, shots=shots, seed=seed)
    pa = normalize(dict(a))

    # Route B: runtime reuse of the CompilationResult (no recompiling)
    plan = ExecutionPlan(compiled=compiled, backend=backend, shots=shots, seed=seed)
    b = sorted(execute(plan=plan).counts.items())
    pb = normalize(dict(b))

    # Route C: runtime compiles at run time
    result = execute(qc, backend=backend, shots=shots, seed=seed,
                     optimization_level=level)
    c = sorted(result.counts.items())
    pc = normalize(dict(c))

    print("route A (backend.run)     :", a)
    print("route B (plan reuse)      :", b)
    print("route C (runtime compile) :", c)

    for label, probs in (("A", pa), ("B", pb), ("C", pc)):
        for outcome in set(pa) | set(pb) | set(pc):
            delta = abs(probs.get(outcome, 0.0) - pa.get(outcome, 0.0))
            assert delta <= 0.05, f"route {label} diverged on {outcome}"

    print("\nAll three routes agree within seeded sampling tolerance.")

    print("\nExample 30 completed!")


if __name__ == "__main__":
    main()