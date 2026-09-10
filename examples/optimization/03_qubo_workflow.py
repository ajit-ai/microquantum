"""QUBO -> OptimizationProblem -> Ising -> QUBO round-trip (MQ-05).

OptimizationProblem keeps both a binary-objective view and a spin-Ising
view of the same problem.  This example shows that energies agree exactly
across all three representations.
"""

import numpy as np

from microquantum import OptimizationProblem
from microquantum.optimization.qubo import QUBOBuilder


def main() -> None:
    print("=== QUBO / OptimizationProblem / Ising consistency ===\n")

    builder = QUBOBuilder(num_variables=3)
    # minimize: x0*x1*2 - x1*x2*3 + x0*(-1.5) + x1*(-0.25)
    builder.add_quadratic(0, 1, 2.0)
    builder.add_quadratic(1, 2, -3.0)
    builder.add_linear(0, -1.5)
    builder.add_linear(1, -0.25)
    qubo = builder.build("demo")

    problem = OptimizationProblem.from_qubo(qubo, name="demo")
    print(f"Ising Hamiltonian: {problem.cost_hamiltonian()}")

    qubo_back = problem.to_qubo()
    print("\n bitstring | QUBO energy |  Ising energy  | round-trip energy")
    for bits_int in range(2 ** problem.num_variables):
        bits = np.array(
            [(bits_int >> q) & 1 for q in range(problem.num_variables)], dtype=float
        )
        e_qubo = qubo.energy(bits)
        e_problem = problem.energy(bits)
        e_back = qubo_back.energy(bits)
        marker = "" if abs(e_qubo - e_back) < 1e-9 else "  <-- MISMATCH"
        bits_str = "".join("1" if b else "0" for b in bits)
        print(f"   {bits_str}      {e_qubo:8.3f}  {e_problem:13.3f}  {e_back:13.3f}{marker}")

    best_bits = min(
        range(2 ** problem.num_variables),
        key=lambda i: problem.energy(np.array([(i >> q) & 1 for q in range(problem.num_variables)], dtype=float)),
    )
    print(f"\nBrute-force optimum bitstring: "
          f"{best_bits:0{problem.num_variables}b} "
          f"(energy {problem.energy(np.array([(best_bits >> q) & 1 for q in range(problem.num_variables)], dtype=float)):.3f})")


if __name__ == "__main__":
    main()