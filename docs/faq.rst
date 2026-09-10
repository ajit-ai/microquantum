FAQ
===

What is MicroQuantum?
----------------------

A lightweight, NumPy-only quantum computing SDK.  You build circuits, execute
them locally, understand results, formulate problems, run algorithms, run
experiments and analyze outcomes — with a pluggable backend architecture ready
for future hardware providers.  It is implemented from scratch; it is **not**
a wrapper around Qiskit, Cirq or OpenQASM.

Who is this for?
----------------

SDK users (engineers, researchers, students) working on quantum circuits and
algorithms who want a small, dependency-light, fully local Python library.

Why NumPy-only?
---------------

Small install surface, no third-party runtime requirements, auditable
implementation, and no accidental dependency on a competing SDK.  GPU / NPU
acceleration is opt-in at the array layer; hardware execution plugs in through
the provider/backend boundary.

Do I need a GPU or a quantum computer?
--------------------------------------

No.  Every built-in backend is a local simulator; ``MockBackend`` is a
deterministic stub for pipelines and tests.  Hardware support is an out-of-box
boundary (``providers`` / ``HardwareProvider``) for optional enterprise
deployments.

Which qubit ordering does MicroQuantum use?
-------------------------------------------

**Big-endian**: qubit 0 is the most-significant bit and tensor axis 0 (the
mathematical convention).  ``bitstring[0]`` corresponds to qubit 0.

How do I bind circuit parameters?
---------------------------------

Create :class:`~microquantum.Parameter` s, use them as rotation angles, then
``qc.bind_parameters({theta: 0.5, ...})``.  Sweeps over many bindings are
expressed as :class:`~microquantum.ParameterSweep` in an
:class:`~microquantum.Experiment`.

How reproducible are results?
-----------------------------

With a seed, local simulators are fully deterministic.  Every execution also
records a configuration fingerprint (SHA-256 over the serialized
configuration) so you can prove two runs used identical parameters —
see :doc:`/experiments/reproducibility`.

Can I add my own backend / algorithm?
-------------------------------------

Yes — see :doc:`/developer-guide/extending`.  Subclass
:class:`~microquantum.Backend` (or :class:`~microquantum.BackendAdapter`),
:class:`~microquantum.Algorithm`, :class:`~microquantum.Problem`, or
:class:`~microquantum.Optimizer`; register backends for name-based selection.

Are provider/enterprise features in this SDK?
---------------------------------------------

No.  The public SDK is MIT and NumPy-only.  Vertical solvers and hardware
integrations live in separate, optional code outside this repository and are
never imported by the SDK.

How do I report a bug or ask a question?
----------------------------------------

Open an issue at https://github.com/ajit-ai/microquantum/issues.