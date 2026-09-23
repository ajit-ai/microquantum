"""MicroQuantum internal intermediate representation and compilation foundation.

The IR is MicroQuantum's own semantic representation: it is imported from
:class:`~microquantum.core.circuit.QuantumCircuit` via :func:`to_ir`,
inspected and validated, transformed by :class:`IRPass` pipelines, and
compiled against an MQ-02 :class:`~microquantum.core.device.Target` by
:class:`Compiler`.  OpenQASM remains an optional interchange format — the
IR is never an external framework's representation.

Node hierarchy::

    IRModule
        └── IRCircuit
              └── IRNode
                   ├── Gate
                   ├── Measurement
                   ├── Reset
                   ├── Barrier
                   ├── ConditionalBlock
                   ├── Loop
                   └── Switch
"""

from .builder import from_ir, to_ir, to_ir_dynamic
from .circuit_ir import IRCircuit, IRModule
from .compiler import CompilationResult, Compiler
from .control import Loop, Switch, loop_from_dict, switch_from_dict
from .cost import AliasAnalysis, CostModel
from .nodes import (
    Barrier,
    Condition,
    ConditionalBlock,
    Gate,
    IRNode,
    IRParam,
    Measurement,
    Reset,
)
from .passes import (
    BindParameters,
    CancelAdjacentInverse,
    CombineRotations,
    GateDecomposition,
    IRPass,
    IRPassManager,
    RemoveIdentityGates,
    optimize,
)
from .validation import assert_valid, validate

__all__ = [
    # Containers
    "IRCircuit",
    "IRModule",
    # Nodes
    "IRNode",
    "IRParam",
    "Gate",
    "Measurement",
    "Reset",
    "Barrier",
    "Condition",
    "ConditionalBlock",
    "Loop",
    "Switch",
    "loop_from_dict",
    "switch_from_dict",
    # Conversion
    "to_ir",
    "to_ir_dynamic",
    "from_ir",
    # Validation
    "validate",
    "assert_valid",
    # Passes
    "IRPass",
    "IRPassManager",
    "RemoveIdentityGates",
    "CancelAdjacentInverse",
    "CombineRotations",
    "BindParameters",
    "GateDecomposition",
    "AliasAnalysis",
    "optimize",
    # Cost model
    "CostModel",
    # Compilation
    "Compiler",
    "CompilationResult",
]