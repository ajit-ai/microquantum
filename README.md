# Microquantum

**A lightweight, NumPy-only quantum computing SDK — MIT licensed and dependency-light.**

[![CI](https://img.shields.io/github/actions/workflow/status/ajit-ai/microquantum/ci.yml?branch=main&label=CI)](https://github.com/ajit-ai/microquantum/actions)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ajit-ai.github.io/microquantum/)
[![PyPI - Version](https://img.shields.io/pypi/v/microquantum)](https://pypi.org/project/microquantum/)
[![PyPI - Python Versions](https://img.shields.io/pypi/pyversions/microquantum)](https://pypi.org/project/microquantum/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/microquantum)](https://pypi.org/project/microquantum/)
[![License: MIT](https://img.shields.io/github/license/ajit-ai/microquantum)](LICENSE)

microquantum is an **independent quantum computing SDK**. It is **not** a port or
wrapper around Qiskit, Cirq, or OpenQASM — every circuit, operator, simulator and
algorithm is implemented from scratch with NumPy as the only hard dependency.

> **Positioning**: an *SDK* — a complete kit for building quantum applications.
> The core engine and algorithms form a *library* (you call it); the
> `DomainAdapter` pipeline and backend/provider abstractions form an embedded
> *framework* (it calls your code); a standardized result contract ties it all
> together for downstream solvers.

---

## What's Included

| System | Description |
|---|---|
| **Core engine** | `QuantumCircuit`, operators, Pauli algebra, `StateVector` / `DensityMatrix`, measurement, gradients, registers, serialization, transpiler + `PassManager` |
| **Algorithms** | VQE, ADAPT-VQE, VQD, QAOA, Grover, Shor, QFT, phase/amplitude estimation, HHL, Hamiltonian simulation (+ qDRIFT, 4th-order Trotter), quantum walks, BV, DJ |
| **Backends** | Statevector, noisy DensityMatrix, MPS and tree tensor networks, pluggable NumPy/CuPy array backend, high-level `Executor`, async `Job` |
| **Execution results & analytics** | structured `ExecutionRecord`s, `ParameterSweep`, `Experiment` / `ExperimentResult`, sampling / expectation / state analysis and `ResultAggregator` (MQ-07) |
| **Providers** | Raw REST clients for IBM Quantum and IonQ (no Qiskit/Cirq/OpenQASM), `CircuitSerializer`, `HardwareBackend` adapter |
| **QML / QEC** | Encodings, quantum kernels, variational classifier; repetition/Shor/bit-flip/phase-flip codes |
| **Chemistry** | H₂/LiH Hamiltonians, UCCSD and hardware-efficient ansätze |
| **Optimization** | QUBO/Ising toolchain: `QUBOBuilder`, `IsingConverter`, constraint penalties |
| **Benchmarks** | Quantum volume, randomized benchmarking, XEB, CLOPS, GST, cycle benchmarking, layer fidelity |
| **Mitigation** | Zero-noise extrapolation (ZNE), probabilistic error cancellation (PEC), measurement-error mitigation (MEM) |
| **Result contract** | `microquantum.analytics.result.Result` — a standardized, JSON-safe decision schema |
| **Domain framework** | `DomainAdapter` ABC: validate → encode → execute → decode, with `QuantumProblem` / `QuantumResult` / `ResultCache` |

---
---

## Installation

Requires **Python 3.10, 3.11, 3.12, or 3.13** and NumPy ≥ 1.20 (installed automatically).

```bash
pip install microquantum
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install microquantum
```

Or from source:

```bash
git clone https://github.com/ajit-ai/microquantum.git
cd microquantum
uv sync --group dev
```

Documentation for the Developer Preview is published at
<https://ajit-ai.github.io/microquantum/> (auto-deployed from the `main`
branch); the Sphinx sources live in `docs/`.

Optional GPU acceleration (NumPy stays the default; the CuPy backend is opt-in):

```bash
pip install "microquantum[gpu]"
```

Verify:

```bash
python -c "import microquantum; print(microquantum.__version__)"
```

---

## Quick Start

### 1. Bell state in two minutes

```python
from microquantum import Executor, QuantumCircuit, StatevectorBackend

qc = QuantumCircuit(2)
qc.h(0)        # Hadamard on qubit 0
qc.cx(0, 1)    # CNOT (control=0, target=1)

result = Executor(backend=StatevectorBackend()).run(qc, shots=1024)
print(result.counts)          # {'00': ~512, '11': ~512}
print(result.most_frequent()) # '00' or '11'
```

`Executor` is the single high-level entry point: pass any `Backend` (statevector,
noisy density matrix, MPS, tensor network, or a hardware provider) — or a
`NoiseModel` for built-in noisy simulation. Every `Backend` also exposes a direct
high-level `backend.run(circuit, shots=1024, seed=None)` call.

### 2. The standardized result contract

```python
from microquantum.analytics.result import Result

result = Result(
    problem="optimization",
    solution={"route": "A->C->B", "cost": 42.0},
    confidence=0.91,
    qubit_count=8,
    runtime_ms=48.3,
    baseline={"route": "A->B->C", "cost": 51.7},
)
print(result.to_json())
print(result.improved_over_baseline)  # True
```

### 3. Running on real hardware

```python
from microquantum import Executor, HardwareBackend, IBMQuantumCredentials, IBMQuantumProvider

provider = IBMQuantumProvider(IBMQuantumCredentials(api_token="..."))
backend = HardwareBackend(provider)

result = Executor(backend=backend).run(qc, shots=1024)
```

---

## Architecture Principles

1. **Independent implementation** — all quantum operations are implemented from
   scratch using NumPy. No dependence on Qiskit, Cirq, OpenQASM, Strawberry
   Fields, or any other quantum SDK.
2. **Layer separation** — each layer only depends on layers below it:
   Analytics never touches raw matrices; domain adapters never bypass the core engine.

   ```
   Engine  →  Algorithms  →  Backends  →  Providers  →  NumPy
   ```

3. **Big-endian qubit ordering** — qubit 0 is the most significant bit; tensor
   axis 0 = qubit 0. This matches the mathematical convention.
4. **Standardized results** — every successful run funnels into a typed result
   (`BackendResult`, `ExecutorResult`, `*Result`, `Result`) that can be
   serialized with `to_dict()` / `to_json()`.
5. **Test-driven** — a full test suite ships with the SDK and runs in CI.

---

## Project Structure

```
microquantum/
├── src/microquantum/
│   ├── core/          # Quantum engine, circuits, operators, transpiler
│   ├── algorithms/    # 23+ quantum algorithms
│   ├── experiments/   # Execution records, sweeps, experiments (MQ-07)
│   ├── analysis/      # Sampling / expectation / state analysis (MQ-07)
│   ├── backends/      # Simulators, noise, tensor networks, Executor
│   ├── providers/     # IBM Quantum & IonQ hardware clients (REST)
│   ├── analytics/     # CSV loading, result contract, analytics base
│   ├── optimization/  # QUBO / Ising toolchain
│   ├── qml/           # Quantum machine learning
│   ├── qec/           # Error correction codes
│   ├── benchmarks/    # Quantum benchmarking suite
│   ├── chemistry/     # Molecular Hamiltonians and ansätze
│   ├── mitigation/    # Error mitigation (ZNE, PEC, MEM)
│   └── optimizers/    # Classical optimizers
├── examples/          # Runnable demo scripts
├── docs/              # Sphinx documentation
└── pyproject.toml     # Package configuration
```

---

## Development

```bash
# Install dev tooling (pytest, coverage, mypy, ruff, sphinx)
uv sync --group dev

# Run the test suite (with coverage report)
uv run pytest tests/

# Type check (0 errors expected)
uv run mypy src/microquantum/ --ignore-missing-imports

# Lint / format
uv run ruff check src tests examples

# Build the docs
uv run sphinx-build docs docs/_build/html
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[CHANGELOG.md](CHANGELOG.md) for release history.

---

## Roadmap

- **v0.4.0** (current) — Developer Preview (see `docs/releases/developer-preview.rst`):
  complete warning-free documentation with auto-generated API reference,
  GitHub Pages deployment, and a consolidated CI/packaging pipeline.
- **v0.3.0** — public open-source release: unified `Backend.run()`,
  serializable results (`to_dict()`), SPDX/legacy metadata cleanup, coverage +
  ruff gates, SDK-only docs.
- **Next** — automated PyPI release workflow on version tags, stricter mypy
  coverage, more hardware providers and tutorials.

---

## License

MIT — see [LICENSE](LICENSE). Built as an independent, dependency-light quantum
computing SDK: use it, fork it, build on it.