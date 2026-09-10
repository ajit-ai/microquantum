Algorithms
==========

Quantum algorithms on top of the core engine.  Full API details live in the
generated reference; the links below jump straight to the relevant module.

Variational Algorithms
----------------------

* :class:`~microquantum.algorithms.vqe.VQE` — variational quantum eigensolver.
* :class:`~microquantum.algorithms.qaoa.QAOA` — quantum approximate optimization.
* :class:`~microquantum.algorithms.adapt_vqe.AdaptVQE` — adaptively grown ansatz.

Search, Estimation and Transforms
---------------------------------

* :class:`~microquantum.algorithms.grover.GroverSearch` — amplitude amplification.
* :class:`~microquantum.algorithms.amplitude_estimation.AmplitudeEstimation` — QAE.
* :class:`~microquantum.algorithms.phase_estimation.QuantumPhaseEstimation` — QPE.
* :class:`~microquantum.algorithms.qft.QFT` — quantum Fourier transform.

Foundational Algorithms
-----------------------

* :class:`~microquantum.algorithms.shor.Shor` — integer factorization.
* :class:`~microquantum.algorithms.bernstein_vazirani.BernsteinVazirani` — hidden
  string.
* :class:`~microquantum.algorithms.deutsch_jozsa.DeutschJozsa` — function balance.

Simulation
----------

* :mod:`microquantum.algorithms.hamiltonian_simulation` — evolution under a
  Hamiltonian.
* :class:`~microquantum.algorithms.quantum_walk.QuantumWalk` — discrete quantum
  walks.

Optimizers
----------

Used by the variational algorithms:

* :class:`microquantum.optimizers.base.Optimizer` — base class.
* :class:`microquantum.optimizers.gradient_descent.GradientDescentOptimizer`.
* :class:`microquantum.optimizers.spsa.SPSAOptimizer`.

Generated reference (by module)
-------------------------------

.. toctree::
   :hidden:

   /api/microquantum/algorithms/vqe/index
   /api/microquantum/algorithms/qaoa/index
   /api/microquantum/algorithms/adapt_vqe/index
   /api/microquantum/algorithms/grover/index
   /api/microquantum/algorithms/amplitude_estimation/index
   /api/microquantum/algorithms/phase_estimation/index
   /api/microquantum/algorithms/qft/index
   /api/microquantum/algorithms/hhl/index
   /api/microquantum/algorithms/shor/index
   /api/microquantum/algorithms/bernstein_vazirani/index
   /api/microquantum/algorithms/deutsch_jozsa/index
   /api/microquantum/algorithms/hamiltonian_simulation/index
   /api/microquantum/algorithms/quantum_walk/index
   /api/microquantum/algorithms/simulation_enhanced/index
   /api/microquantum/optimizers/base/index
   /api/microquantum/optimizers/gradient_descent/index
   /api/microquantum/optimizers/gradient_free/index
   /api/microquantum/optimizers/quasi_newton/index
   /api/microquantum/optimizers/spsa/index