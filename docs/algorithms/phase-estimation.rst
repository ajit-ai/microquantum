Phase Estimation
================

Quantum phase estimation (QPE) estimates the eigenphase ``theta`` of a
*unitary* operator ``U``: if ``U|psi> = exp(2*pi*i*theta)|psi>``, QPE returns
``theta``.  For example, ``RZ(0.3)`` has eigenphase ``0.3 / 2``, so the
resulting ``phase`` is ``0.3 / (2 * 2 * pi) ~ 0.024`` (normalized to
``[0.0, 1.0)``) and ``phase_radians`` is ``~ 0.15``.

Usage
-----

.. code-block:: python

   from microquantum import HamiltonianProblem, Operator
   from microquantum.algorithms import PhaseEstimation

   # U = RZ(theta) is a unitary; its eigenphase encodes theta.
   unitary = Operator.Rz(0.3)
   problem = HamiltonianProblem(unitary, name="phase")

   qpe = PhaseEstimation(unitary=unitary, num_counting_qubits=8)
   print(qpe.validate(problem))             # []
   result = qpe.solve(problem, seed=0)

   print(result.phase)            # ~ 0.024 (theta/2, normalized to [0.0, 1.0))
   print(result.phase_radians)    # ~ 0.15  (theta/2 in radians)

Problem-driven use
------------------

:class:`~microquantum.PhaseEstimation` supports ``from_problem(problem)`` for
the :class:`~microquantum.HamiltonianProblem` / unconstrained
:class:`~microquantum.EigenvalueProblem` case.  **The Hamiltonian must be
unitary** — a Hermitian operator is generally *not* unitary, so for the
spectrum of a Hermitian operator use :doc:`vqe` instead of QPE.

Notes
-----

* ``num_counting_qubits`` controls the precision — more counting qubits ->
  more accurate phase estimates (extra qubits).
* Works on any unitary :class:`~microquantum.Operator`.
* Result :class:`~microquantum.PhaseEstimationResult` is JSON-safe
  (``to_dict()`` / ``to_json()``).