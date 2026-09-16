Compilation
===========

MicroQuantum's compiler turns a program (a :class:`~microquantum.QuantumCircuit`
or an :class:`~microquantum.IRCircuit`) into an optimized, target-aware
intermediate representation that can be rebuilt into an executable circuit.

There is a single public entry point: the
:class:`~microquantum.Compiler` class.

.. code-block:: python

   from microquantum import Compiler, QuantumCircuit

   qc = QuantumCircuit(2).h(0).cx(0, 1)
   result = Compiler(optimization_level=1).compile(qc)
   compiled = result.circuit()      # executable QuantumCircuit

The pipeline is fixed and deterministic:

1. **Validate** — the IR is structurally validated; malformed IR fails loudly.
2. **Optimize** — safe, semantics-preserving transformations only.
3. **Decompose** — only when a ``Target`` is supplied, gates are expanded into
   the target's native basis (exact, standard identities).
4. **Diagnose** — target-compatibility problems are reported as a list; the
   compiler never silently drops an operation it cannot handle.

A :class:`~microquantum.CompilationResult` carries the source IR, the compiled
IR, the applied passes, the diagnostics and rich metadata; its ``circuit()``
method rebuilds a ``QuantumCircuit`` that runs on all existing backends.

Optimization levels
-------------------

``optimization_level`` selects how aggressive the standard pipeline is.
Levels outside ``0..2`` are rejected with ``ValueError``.

.. list-table::
   :widths: 10 50 40
   :header-rows: 1

   * - Level
     - Passes
     - Used for
   * - ``0``
     - none (validation only)
     - probing `"what does my program look like as IR?"`
   * - ``1``
     - ``remove-identity-gates``, ``cancel-adjacent-inverse``
     - general safe simplification
   * - ``2``
     - level 1 + ``combine-rotations``
     - reducing rotation count (never for symbolic angles)

.. code-block:: python

   from microquantum import Compiler, QuantumCircuit

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.h(0)              # identity pair -> removed
   qc.cx(0, 1)

   result = Compiler(optimization_level=1).compile(qc)
   print(result.passes_applied)
   # ['remove-identity-gates', 'cancel-adjacent-inverse']
   print(result.result.num_gates)          # 1  (the cnot)

The individual passes are also available directly
(:class:`~microquantum.RemoveIdentityGates`,
:class:`~microquantum.CancelAdjacentInverse`,
:class:`~microquantum.CombineRotations`) and can be composed with
:class:`~microquantum.IRPassManager` or passed to ``Compiler.compile(...,
passes=[...])``.

Safe transformations
--------------------

Every transformation preserves the state vector exactly (up to floating
point).  The implemented rules are structural and **numeric-only**:

* identity gates and zero-angle rotations ``R(0)`` are removed;
* adjacent inverse pairs cancel: self-inverse gates (``H X Y Z CNOT CZ
  SWAP``), the ``S``/``Sdg`` and ``T``/``Tdg`` pairs, and opposite rotations
  ``R(a) R(-a)`` (or a full ``2*pi`` turn);
* adjacent same-axis rotations on one qubit fuse: ``R(a) R(b) -> R(a+b)``.

Symbolic ``Parameter`` angles are never fused, cancelled or folded, because an
unbound ``RX(theta) RX(theta)`` is textually ``RX(2*theta)`` only after
binding — optimizing it away would change the program.

.. code-block:: python

   from microquantum import Compiler, Gate, IRCircuit, Parameter

   theta = Parameter("theta")
   symbolic = IRCircuit(num_qubits=1, operations=[
       Gate(name="rx", qubits=(0,), params=(theta,)),
       Gate(name="rx", qubits=(0,), params=(theta,)),
   ])
   out = Compiler(optimization_level=2).compile(symbolic)
   assert out.result.gate_names() == {"rx": 2}   # never simplified

Measurements
------------

Terminal measurements are preserved through compilation and rebuilding:
compile a measured circuit and ``result.circuit()`` keeps the same measured
qubits in the same order.  An unmeasured circuit is treated as a full
sampling program, so the compiler appends terminal measurements on every
qubit.  Measurement nodes act as barriers: gates on different sides never
cancel across a measurement.

.. code-block:: python

   from microquantum import Compiler, QuantumCircuit

   qc = QuantumCircuit(3).x(1).measure(2).measure(0)
   compiled = Compiler(optimization_level=1).compile(qc).circuit()
   assert compiled.measurements == [2, 0]

``QuantumCircuit -> IR -> QuantumCircuit`` is lossless: rotation signs are
recovered exactly, so ``from_ir(to_ir(qc))`` reproduces the same behavior
(with the sdk's exact-amplitude simulators the resulting state vectors agree
to machine precision).

Targets & capabilities
----------------------

Compiling toward a :class:`~microquantum.Target` adds basis decomposition and
compatibility diagnostics.  This is a **two-tier** contract:

* the compiler reports problems as a ``diagnostics`` list and flips
  ``is_compatible`` to ``False`` (soft, inspectable);
* the runtime raises ``ValueError`` when an execution plan or backend refuses
  the program (hard).

Unsupported gates are **never silently dropped** — they remain in the compiled
IR and are listed in ``diagnostics``.  The target's ``native_gates`` may spell
the controlled-NOT as ``"cx"``; the IR calls the same gate ``"cnot"`` and the
two are treated as identical.

.. code-block:: python

   from microquantum import Compiler, QuantumCircuit, Target

   target = Target(
       name="limited",
       num_qubits=2,
       native_gates=("h", "cx"),   # note the cx spelling
       supports_measurement=True,
   )
   result = Compiler().compile(QuantumCircuit(2).cz(0, 1), target=target)
   print(result.circuit().to_ir().gate_names())   # {'h': 2, 'cnot': 1} (cz lowered)
   assert result.is_compatible

Standard exact decompositions are used when they improve compatibility:

*: ``cz(c, t) -> h(t), cnot(c, t), h(t)``   (basis has ``h`` and ``cnot``)
*: ``swap(a, b) -> cnot(a,b), cnot(b,a), cnot(a,b)``   (basis has ``cnot``)

Metrics
-------

``CompilationResult.metadata`` (JSON-safe) reports the before/after shape:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Key
     - Meaning
   * - ``source_qubits`` / ``compiled_qubits``
     - qubit counts
   * - ``source_gates`` / ``compiled_gates``
     - gate counts
   * - ``source_depth`` / ``compiled_depth``
     - circuit depth (via the ``depth()`` method)
   * - ``source_gates_by_type`` / ``compiled_gates_by_type``
     - gate-name -> count dictionaries
   * - ``optimization_level``
     - the level used

Backends
--------

Compiled circuits are ordinary ``QuantumCircuit`` objects and run on every
local simulator.  The probability vectors are identical across the state
vector, density-matrix, MPS and TTN backends.

.. code-block:: python

   from microquantum import Compiler, MPSBackend, QuantumCircuit, StatevectorBackend

   qc = QuantumCircuit(3).h(0).h(0).cx(0, 1).cx(1, 2).measure_all()
   compiled = Compiler(optimization_level=1).compile(qc).circuit()
   sv = StatevectorBackend().run(compiled, shots=4000, seed=10).counts
   mps = MPSBackend().run(compiled, shots=4000, seed=10).counts
   print(sv, mps)   # statistically consistent

QASM
----

A compiled circuit exports standard OpenQASM 2.0 and re-imports losslessly:

.. code-block:: python

   from microquantum import Compiler, QuantumCircuit

   qc = QuantumCircuit(2).h(0).cx(0, 1)
   compiled = Compiler(optimization_level=1).compile(qc).circuit()
   restored = QuantumCircuit.from_qasm(compiled.qasm())

Determinism
-----------

Compilation is deterministic: compiling the same input twice yields an
identical ``CompilationResult`` (including its JSON serialization).

Limitations
-----------

The compiler deliberately stops before hardware.  Out of scope today:

* qubit routing / layout synthesis and coupling-map-aware scheduling;
* vendor device calibration, pulse or timing control;
* GPU/distributed/HPC orchestration, LLVM or external frameworks;
* optimal-circuit synthesis search (Qiskit/Cirq/pytket/SAT-SMT/ML are not used);
* noisy/error-mitigated compilation and QEC.

See the IR API reference (``api/microquantum/ir``, including
:class:`~microquantum.IRCircuit`, :class:`~microquantum.IRPass` and
:class:`~microquantum.IRPassManager`) for the internal representation and the
pass classes.  Run the whole flow end to end in the examples:
``examples/22_ir_roundtrip.py`` through
``examples/29_execute_compiled_circuit.py``.