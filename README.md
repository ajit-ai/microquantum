# microquantum

**Independent Quantum Computing SDK for Business & Industry Solutions**

microquantum is a lightweight, open-source Python SDK that transforms quantum computing into practical business solutions. It is **not** a port or wrapper around Qiskit, Cirq, or OpenQASM — it is an independent, self-contained quantum computing library and framework with a standardized solver-result contract.

> **Core Rule**: microquantum is an independent language and library framework. It does not follow Qiskit/Cirq/OpenQASM patterns.

---

## Open-Source Model

This is the **public `microquantum` SDK** — published to PyPI (MIT).

> The proprietary vertical solvers (industry analytics, domain adapters) and
> the Quant's Mind platform live in a separate private repository
> (`Microquantum-company`) that depends on this SDK from PyPI.

---

## What the Open SDK Ships

| System | Description |
|---|---|
| **Core engine** | QuantumCircuit, operators, Pauli algebra, state/DensityMatrix, measurement, gradients, registers, serialization, transpiler |
| **Algorithms** | VQE, ADAPT-VQE, VQD, QAOA, Grover, Shor, QFT, Phase/Amplitude Estimation, HHL, Hamiltonian simulation, walks, BV, DJ |
| **Backends** | Statevector, DensityMatrix (noisy), MPS + tree tensor networks, pluggable NumPy/CuPy array backend, Executor |
| **Providers** | Raw REST clients for IBM Quantum and IonQ (no Qiskit/Cirq/OpenQASM), serializer + HardwareBackend adapter |
| **QML / QEC** | Encodings, quantum kernels, variational classifier; repetition/Shor/bit-flip/phase-flip codes |
| **Chemistry** | H2/LiH Hamiltonians, UCCSD, hardware-efficient ansätze |
| **Optimization** | QUBO toolchain: QUBOBuilder, IsingConverter, constraint penalties |
| **Result contract** | `microquantum.analytics.result.Result` — the standardized decision schema consumed by proprietary solvers |

---

## Installation

### Prerequisites
- **Python 3.10, 3.11, 3.12, or 3.13**
- **pip** (or **uv** for faster installs)
- **numpy ≥ 1.20** (installed automatically)

### Install from PyPI
```bash
pip install microquantum
```

### Install with uv (fastest)
```bash
uv pip install microquantum
```

### Install from source
```bash
git clone https://github.com/ajit-ai/microquantum.git
cd microquantum
pip install -e .
```

### Verify Installation
```bash
python -c "import microquantum; print(microquantum.__version__)"
# Output: 0.3.0
```

---

## Quick Start

### 1. Bell State (2 minutes)
```python
from microquantum import QuantumCircuit, StatevectorBackend, Executor

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

backend = StatevectorBackend()
result = backend.run_circuit(num_qubits=2, gates=[(op.matrix, targets) for op, targets in qc.gates], shots=1024)
print(result.counts)  # {'00': ~512, '11': ~512}
```

### 2. Standardized Result Contract
```python
from microquantum.analytics.result import Result

result = Result(
    problem="fraud_detection",
    decision={"flagged": True, "score": 0.87},
    confidence=0.87,
    qubit_count=4,
    runtime_ms=12.5,
    classical_baseline={"flagged": True, "score": 0.62},
)
print(result.to_json())
print(result.improved_over_classical)  # True
```

---

## Architecture Principles

### 1. Independent Implementation
microquantum implements all quantum operations from scratch using NumPy. No dependence on Qiskit, Cirq, OpenQASM, Strawberry Fields, or any other quantum SDK.

### 2. Layer Separation
Each layer only depends on layers below it. Analytics never touches raw matrices. Domain adapters never bypass the core engine.

```
Engine  →  Algorithms  →  Backends  →  Providers → NumPy
```

### 3. Big-Endian Qubit Ordering
Qubit 0 is the most significant bit. Tensor axis 0 = qubit 0. This matches mathematical convention.

### 4. Test-Driven Development
The SDK ships with a full test suite. Run with:
```bash
uv run pytest tests/ -v
```

---

## Modern Standard Qubit Count

| Hardware | Qubits | microquantum Simulated |
|---|---|---|
| Entanglement | 2 qubits | ✅ |
| Superposition | 2-4 qubits | ✅ |
| Quantum Teleportation | 3 qubits | ✅ |
| Grover's Search | 12+ qubits | ✅ |
| Shor's Algorithm | 10+ qubits | ✅ |
| HHL | 10+ qubits | ✅ |
| VQE Chemistry | 12+ qubits | ✅ |
| QAOA Optimization | 25 qubits | ✅ |
| **Practical CPU Limit** | **~25 qubits** | ✅ |

---

## SDK Features

### Core Engine
- ✅ QuantumCircuit: single/multi-qubit gates, parameterized, dynamic
- ✅ Operator algebra: tensor products, adjoints, Pauli algebra
- ✅ StateVector, DensityMatrix, measurement, expectation values
- ✅ Parameterized circuits, binding, gradient computation
- ✅ Circuit serialization (JSON/dict/QASM), transpilation, PASS Manager
- ✅ Coupling maps, routing, noise-aware placement
- ✅ Register-based circuits, classical registers, dynamic circuits
- ✅ Result contract (`analytics/result.py`) with JSON-safe serialization

### Algorithms (23+)
- ✅ VQE, ADAPT-VQE, VQD, QAOA
- ✅ Grover, Shor, QFT, Phase Estimation
- ✅ Amplitude Estimation, HHL, Hamiltonian simulation
- ✅ Quantum Walk (discrete + continuous), BV, DJ
- ✅ 4th-order Suzuki-Trotter, qDRIFT, Pauli commutation grouping

### QEC • QML • Benchmarks • Mitigation
- ✅ Repetition/Shor/bit-flip/phase-flip codes
- ✅ Quantum kernels, encoders, variational classifier
- ✅ Quantum Volume, RB, XEB, GST, cycle benchmarking
- ✅ ZNE, PEC, measurement error mitigation

### Real Hardware Providers (Phase 14)
- ✅ IBM Quantum provider (raw REST, token auth)
- ✅ IonQ provider (raw REST API v0.3, API token)
- ✅ CircuitSerializer → native gate JSON (no Qiskit/Cirq/OpenQASM)
- ✅ HardwareBackend adapter (drops into Executor/Backend API)
- ✅ HardwareJob lifecycle (submit → poll → result)

### Tensor-Network Simulation (Phase 15)
- ✅ Pluggable array backend: NumPy by default, CuPy/GPU opt-in (`set_array_backend("cupy")`)
- ✅ `MatrixProductState` + `MPSBackend`: exact SVD simulation with optional bond-dimension truncation
- ✅ `TreeTensorNetwork` + `TreeTensorNetworkBackend`: balanced binary tree with optional bond caps
- ✅ Truncation-error tracking on both simulators

### Optimization Toolchain
- ✅ QUBOBuilder, IsingConverter (QUBO ↔ Ising round-trip), constraint penalties
- ✅ Standardized result output for downstream solver integrations

---

## Project Structure

```
microquantum/
├── src/microquantum/
│   ├── core/                # Quantum engine
│   ├── algorithms/          # 23+ quantum algorithms
│   ├── backends/            # Simulators + noise + tensor networks
│   ├── providers/           # Real hardware clients (Phase 14)
│   ├── analytics/           # CSV loading, base analytics, result contract
│   ├── optimization/        # QUBO toolchain (Phase 13)
│   ├── qml/                 # Quantum machine learning
│   ├── qec/                 # Error correction codes
│   ├── benchmarks/          # Quantum benchmarking suite
│   ├── chemistry/           # Molecular Hamiltonians + ansätze
│   └── mitigation/          # Error mitigation techniques
├── examples/                # SDK-facing demos
├── docs/                    # Sphinx documentation
└── pyproject.toml           # Package configuration
```

---

## Running the Test Suite

```bash
# SDK tests
uv run pytest tests/ -v

# Type checking (0 errors)
uv run mypy src/microquantum/ --ignore-missing-imports

# Single module
uv run pytest tests/test_phase12_analytics.py -v
```

---

## Roadmap

- **Phase 12** ✅ Business analytics layer (moved to proprietary)
- **Phase 13** ✅ QUBO toolchain (open) + banking analytics (proprietary)
- **Phase 13.5** ✅ Extended analytics breadth (proprietary)
- **Phase 14** ✅ Real hardware integration (IBM Quantum, IonQ providers)
- **Phase 15** ✅ GPU acceleration and tensor-network simulation
- **Phase 16** ✅ Production hardening (proprietary platform)
- **Phase 17** ✅ Real-time fraud detection with Grover search (proprietary)
- **P-Open-Core** ✅ Public SDK repo (this one, PyPI) + private company repo (Microquantum-company)

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contact

- Author: Ajit Kumar
- Repository: https://github.com/ajit-ai/microquantum