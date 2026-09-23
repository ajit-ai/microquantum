"""Phase 122 W2: version single-source, strict search typing, replay."""

from __future__ import annotations

import pytest

from microquantum import __version__ as package_version
from microquantum.problems import SearchProblem
from microquantum.providers import IonQCredentials, ProviderCredentials
from microquantum.providers.replay import ReplayTransport
from microquantum.runtime import runtime_info
from microquantum.runtime.info import _sdk_version


class TestVersionSingleSource:
    def test_sdk_version_matches_package(self) -> None:
        assert _sdk_version() == package_version
        assert package_version == "1.1.0"

    def test_runtime_info_matches_package(self) -> None:
        assert runtime_info().version == package_version

    def test_cli_info_matches_package(self, capsys: object) -> None:
        from microquantum._cli import main

        assert main(["info"]) == 0
        out = capsys.readouterr().out  # type: ignore[union-attr]
        assert package_version in out


class TestSearchProblemTyping:
    def test_bitstring_targets_normalize(self) -> None:
        problem = SearchProblem(num_qubits=3, target="101")
        assert problem.target == 5
        assert problem.target_indices() == [5]
        assert problem.is_marked("101")
        assert problem.to_dict()["target"] == 5

    def test_invalid_targets_rejected(self) -> None:
        with pytest.raises(TypeError):
            SearchProblem(num_qubits=2, target="012")
        with pytest.raises(TypeError):
            SearchProblem(num_qubits=2, target="")
        with pytest.raises(TypeError):
            SearchProblem(num_qubits=2, target=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            SearchProblem(num_qubits=2, target=1.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            SearchProblem(num_qubits=2, target=[0, "1"])  # type: ignore[list-item]
        with pytest.raises(ValueError):
            SearchProblem(num_qubits=2, target=9)

    def test_int_targets_still_work(self) -> None:
        problem = SearchProblem(num_qubits=2, target=[0, 3])
        assert problem.target_indices() == [0, 3]
        assert problem.num_solutions() == 2


class TestReplayTransport:
    def test_scripted_responses_in_order(self) -> None:
        transport = ReplayTransport(script=[(200, {"a": 1}), (404, {"error": "gone"})])
        assert transport(method="GET", url="https://x/1", headers={}, body=None) == (
            200,
            {"a": 1},
        )
        assert transport.remaining == 1
        assert transport(method="GET", url="https://x/2", headers=None, body=b"{}") == (
            404,
            {"error": "gone"},
        )
        assert len(transport.requests) == 2
        assert transport.requests[1]["body"] == "{}"
        with pytest.raises(LookupError):
            transport(method="GET", url="https://x/3", headers=None, body=None)

    def test_save_and_load_round_trip(self, tmp_path: object) -> None:
        from pathlib import Path

        transport = ReplayTransport(script=[(200, {"ok": True})])
        fixture = Path(str(tmp_path)) / "job.json"
        transport.save(fixture)
        rebuilt = ReplayTransport.load(fixture)
        assert rebuilt(method="POST", url="https://x/submit", headers={"H": "v"}, body=None) == (
            200,
            {"ok": True},
        )
        assert rebuilt.requests[0]["headers"] == {"H": "v"}
        with pytest.raises(ValueError):
            ReplayTransport.load(fixture.parent / "missing.json")

    def test_provider_flow_without_network(self) -> None:
        from microquantum.core.circuit import QuantumCircuit
        from microquantum.providers import IonQProvider

        script = [
            (200, {"id": "job-1", "status": "submitted"}),
            (200, {"id": "job-1", "status": "completed"}),
            (200, {"id": "job-1", "status": "completed", "data": {"histogram": {"00": 0.5, "11": 0.5}}}),
        ]
        provider = IonQProvider(
            credentials=IonQCredentials(api_token="test"),
            transport=ReplayTransport(script=script),
        )
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        job = provider.submit(circuit, shots=100)
        assert job.job_id == "job-1"
        assert len(provider._transport.requests) == 1
        assert provider._transport.requests[0]["method"] == "POST"
        assert repr(provider._transport).startswith("ReplayTransport(")

    def test_proxy_credentials_validate(self) -> None:
        assert ProviderCredentials(api_token="t").proxy is None
