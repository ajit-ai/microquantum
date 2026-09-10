"""MQ-07 example 03: parameter sweeps.

:class:`ParameterSweep` defines deterministic value grids per parameter:
explicit lists, ``range``-style steps, or ``linspace``-style point counts.
Sweeps combine over a base plan/circuit as an ordered Cartesian product.
"""

import microquantum as mq

# Three declaration styles, mixed freely.
sweep = mq.ParameterSweep(
    {
        "theta": [0.0, 1.0],                    # explicit values
        "phi": {"range": (0.0, 0.6, 0.3)},      # start, stop, step
        "psi": {"start": 0.0, "stop": 1.0, "num_points": 3},  # linspace
    }
)
print(f"parameters:  {sweep.parameters}")
print(f"combinations: {sweep.num_combinations} (deterministic order)")

for combination in sweep.combinations():
    print("  ", combination)

# Validation ties a sweep to the base work's actual parameters.
print(f"verify against full param set: "
      f"{sweep.verify(set(sweep.parameters)) or 'OK'}")
try:
    sweep.verify({"theta"})
except ValueError as exc:
    print(f"verify against partial set:   {str(exc).splitlines()[1].strip() if exc else ''}")