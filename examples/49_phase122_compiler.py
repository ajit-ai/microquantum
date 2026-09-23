"""Phase 122 W4 tour: hardware-aware transpiler passes."""

from __future__ import annotations

from microquantum.core.architecture import linear_architecture
from microquantum.core.circuit import QuantumCircuit
from microquantum.core.transpiler import (
    CommutationAwareCancellation,
    NoiseAwareLayout,
    PassContext,
    SwapRoutingPass,
)


def main() -> None:
    source = QuantumCircuit(3)
    source.cx(0, 2)
    context = PassContext()
    routed = SwapRoutingPass(architecture=linear_architecture(3)).transform(source, context)
    print("routed gates:", routed.num_gates, "swaps:", context.analysis["swaps_added"])
    print("final layout:", context.analysis["final_layout"])

    layout_context = PassContext()
    NoiseAwareLayout(linear_architecture(3), {"q0": 0.05, "q1": 0.01, "q2": 0.03}).transform(
        source, layout_context
    )
    print("noise layout:", layout_context.analysis["layout"])

    cancellable = QuantumCircuit(2)
    cancellable.x(1)
    cancellable.cx(0, 1)
    cancellable.x(1)
    print(
        "cancelled gates:",
        CommutationAwareCancellation().transform(cancellable, PassContext()).num_gates,
    )


if __name__ == "__main__":
    main()
