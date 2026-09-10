Algorithms
==========

Algorithms consume problems through a uniform lifecycle:

.. code-block:: text

   problem = ...
   algorithm.validate(problem)     # list[str], empty when valid
   result  = algorithm.solve(problem, runtime=None)

The base contract is :class:`~microquantum.Algorithm`; results are typed
``*Result`` dataclasses that serialize via ``to_dict()`` / ``to_json()``.  A
:class:`~microquantum.runtime.ExecutionRuntime` may be passed to route every
circuit evaluation through the MQ-04 pipeline; without one, algorithms use the
internal NumPy state-vector engine.

Built-ins
---------

* :class:`~microquantum.VQE` — variational ground-state finder for an
  :class:`~microquantum.EigenvalueProblem` (parameter-shift or operator
  gradients).
* :class:`~microquantum.QAOA` — variational solver for an
  :class:`~microquantum.OptimizationProblem` (CNOT parity chains for ZZ terms).
* :class:`~microquantum.GroverSearch` — amplitude amplification for
  :class:`~microquantum.SearchProblem`.
* :class:`~microquantum.PhaseEstimation` — eigenphase of a *unitary*
  :class:`~microquantum.Operator`.
* :class:`~microquantum.QFT` / ``inverse_qft_circuit`` — quantum Fourier
  transform circuits.
* :class:`~microquantum.HamiltonianSimulation` (Trotter / qDRIFT /
  fourth-order), :class:`~microquantum.HHL`, :class:`~microquantum.VQD`,
  :class:`~microquantum.AdaptVQE`, :class:`~microquantum.ShorsAlgorithm`,
  :class:`~microquantum.BernsteinVazirani`, :class:`~microquantum.DeutschJozsa`,
  quantum walks (:class:`~microquantum.DiscreteQuantumWalk`,
  :class:`~microquantum.ContinuousQuantumWalk`).

Example
-------

.. code-block:: python

   from microquantum import EigenvalueProblem, Operator, Parameter, QuantumCircuit
   from microquantum.algorithms import VQE
   from microquantum.optimizers import COBYLA

   theta = Parameter("theta")
   ansatz = QuantumCircuit(1).ry(theta, 0)

   vqe = VQE(ansatz, Operator.Z(), COBYLA(max_iter=100))
   problem = EigenvalueProblem(Operator.Z(), k=1, name="z")

   print(vqe.validate(problem))            # []
   result = vqe.solve(problem, initial_params={theta: 0.5})
   print(result.eigenvalue)                # ~ -1.0
   print(result.to_dict()["algorithm"])    # "VQE"

Optimizers
----------

Classical optimizers in :mod:`microquantum.optimizers` share one interface
(:class:`~microquantum.Optimizer`): ``minimize(cost_fn, gradient_fn=None,
initial_params) -> OptimizerResult``.  Gradient-free (COBYLA, NelderMead),
gradient-based (GradientDescent, Adam, SPSA, QNSPSA) and quasi-Newton (BFGS,
L-BFGS-B) families are available.