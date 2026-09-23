"""Phase 121 W5 tour: experience and hardware."""

from __future__ import annotations

import tempfile
from pathlib import Path

from microquantum.analytics import ReportBuilder, Result
from microquantum.benchmarks import BenchmarkSuite, MirrorBenchmarking
from microquantum.experiments import AdaptiveSweep, Checkpoint, ParameterSweep
from microquantum.providers import ProviderCredentials, ProviderErrorMapper


def main() -> None:
    print("proxy:", ProviderCredentials(api_token="t", proxy="http://p:8080").proxy)
    print("mapper:", ProviderErrorMapper().to_status({"status": "running"}).value)

    result = Result(problem="demo", solution={"bits": "01"}, confidence=0.9)
    report = ReportBuilder(title="Demo").add_result("Outcome", result)
    print(report.to_markdown().splitlines()[0])

    print("polarization:", round(MirrorBenchmarking.polarization({"00": 90, "11": 10}, 2), 4))
    suite = BenchmarkSuite(seed=1)
    suite.add("m", MirrorBenchmarking(num_qubits=1, depths=(1,), num_circuits=1, num_shots=32, seed=0))
    print("suite:", suite.run().summary())

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.json"
        saved = Checkpoint("e1")
        saved.mark_done("a")
        saved.save(path)
        print("checkpoint:", Checkpoint.load(path).completed)

    adaptive = AdaptiveSweep(ParameterSweep({"theta": [0.0, 1.0, 2.0]}))
    refined = adaptive.refine([({"theta": 0.0}, 1.0), ({"theta": 1.0}, 0.0)])
    print("refined combos:", len(refined.combinations()))


if __name__ == "__main__":
    main()
