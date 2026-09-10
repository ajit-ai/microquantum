"""Backend example 2: inspecting and comparing backend capabilities.

:class:`BackendCapabilities` describe what a backend can do without running
anything: target class, execution modes, circuit features, qubit capacity
and native gates.  They serialize to JSON, so a plan / runbook / CI check
can reason about a backend before dispatch.
"""

from microquantum import (
    BackendRegistry,
    LocalProvider,
    StatevectorBackend,
)


def show(label: str, caps) -> None:
    print(f"--- {label} ---")
    print(f"  target_class   : {caps.target_class.value}")
    print(f"  max_qubits     : {caps.max_qubits}")
    print(f"  execution      : {sorted(caps.execution)}")
    print(f"  circuit_features: {sorted(caps.circuit_features)}")
    print(
        f"  modes          : "
        f"statevector={caps.supports_statevector} "
        f"shots={caps.supports_shots} "
        f"simulator={caps.is_simulator} hardware={caps.is_hardware}"
    )


def main():
    # 1) the reference local simulator
    show("LocalSimulatorBackend", StatevectorBackend().capabilities)

    # 2) everything discovered through the built-in provider
    print("=== LocalProvider backends ===")
    provider = LocalProvider()
    registry = BackendRegistry()
    provider.register_all(registry)
    for backend in registry.list():
        show(backend.name, backend.capabilities)
        print(f"  serialized: {backend.capabilities.to_dict()['target_class']}")

    # 3) capabilities are mergeable (capability intersection)
    merged = StatevectorBackend().capabilities.merge(
        LocalProvider().get_backend("mock").capabilities
    )
    print("=== merge(local_simulator, mock) ===")
    print(
        f"  shared execution: {sorted(merged.execution)} "
        f"| max_qubits={merged.max_qubits}"
    )


if __name__ == "__main__":
    main()