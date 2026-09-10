Phase Estimation
================

Quantum phase estimation (QPE) estimates the eigenphase ``theta`` of a
*unitary* operator ``U``: if ``U|psi> = exp(2*pi*i*theta)|psi>``, QPE returns
``theta``.

Usage
-----

.. code-block:: python

   from microquantum import HamiltonianProblem, Operator
   from microquantum.algorithms import PhaseEstimation

   # U = RZ(theta) is a unitary; its eigenphase encodes theta.
   unitary = Operator.Rz(0.3)
   problem = HamiltonianProblem(unitary, name="phase")

   qpe = PhaseEstimation(num_ancillae=4)
   print(qpe.validate(problem))             # []
   result = qpe.solve(problem, seed=0)

   print(result.phase)            # ~ 0.3  (normalized to [0.0, 1.0))
   print(result.eigenphase)       # 2*pi*phase in radians (0.3 * 2*pi)

Problem-driven use
------------------

:class:`~microquantum.PhaseEstimation` supports ``from_problem(problem)`` for
the :class:`~microquantum.HamiltonianProblem` / unconstrained
:class:`~microquantum.EigenvalueProblem` case.  **The Hamiltonian must be
unitary** — a Hermitian operator is generally *not* unitary, so for the
spectrum of a Hermitian operator use :doc:`vqe` instead of QPE.

Notes
-----

* ``num_ancillae`` controls the precision — more ancillae -> more accurate
  phase estimates (extra qubits).
* Works on any unitary :class:`~microquantum.Operator`.
* Result :class:`~microquantum.PhaseEstimationResult` is JSON-safe
  (``to_dict()`` / ``to_json()``).