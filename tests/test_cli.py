"""Phase 119 tests: the ``microquantum`` developer CLI."""

import json
import subprocess
import sys

import pytest

from microquantum._cli import main

BELL_STATE_QASM = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""


@pytest.fixture
def bell_qasm_file(tmp_path):
    path = tmp_path / "bell.qasm"
    path.write_text(BELL_STATE_QASM, encoding="utf-8")
    return path


class TestVersion:
    def test_version_command(self, capsys):
        assert main(["version"]) == 0
        out = capsys.readouterr().out.strip()
        assert out.startswith("microquantum ")

    def test_version_flag(self, capsys):
        assert main(["--version"]) == 0
        assert capsys.readouterr().out.strip().startswith("microquantum ")


class TestInfo:
    def test_info(self, capsys):
        assert main(["info"]) == 0
        out = capsys.readouterr().out
        assert "Runtime:   ExecutionRuntime" in out
        assert "Strategies:" in out
        assert "Backends:" in out

    def test_info_reflects_version(self, capsys):
        import microquantum

        main(["info"])
        assert microquantum.__version__ in capsys.readouterr().out


class TestBackends:
    def test_backends_text(self, capsys):
        assert main(["backends"]) == 0
        out = capsys.readouterr().out
        assert "local_simulator" in out

    def test_backends_json(self, capsys):
        assert main(["backends", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "backends" in data
        assert any(b["name"] == "local_simulator" for b in data["backends"])


class TestRun:
    def test_run_bell_state(self, capsys, bell_qasm_file):
        assert main(["run", str(bell_qasm_file), "--shots", "64", "--seed", "7"]) == 0
        out = capsys.readouterr().out
        assert "Backend:" in out
        assert "Shots:       64" in out
        assert "  |00>:" in out

    def test_run_json_flag_rejected_for_unknown_option(self, bell_qasm_file):
        with pytest.raises(SystemExit) as exc:
            main(["run", str(bell_qasm_file), "--optimization-level", "9"])
        assert exc.value.code == 2

    def test_run_missing_file_returns_one(self, capsys, tmp_path):
        missing = tmp_path / "nope.qasm"
        assert main(["run", str(missing)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("microquantum: error:")

    def test_run_missing_file_debug_reraises(self, tmp_path):
        missing = tmp_path / "nope.qasm"
        with pytest.raises(FileNotFoundError):
            main(["--debug", "run", str(missing)])

    def test_run_invalid_qasm_returns_one(self, capsys, tmp_path):
        bad = tmp_path / "bad.qasm"
        bad.write_text("this is not qasm", encoding="utf-8")
        assert main(["run", str(bad)]) == 1
        assert capsys.readouterr().err.startswith("microquantum: error:")


class TestArgumentHandling:
    def test_no_command_prints_help_and_returns_zero(self, capsys):
        assert main([]) == 0
        assert "usage: microquantum" in capsys.readouterr().out

    def test_unknown_command_exits_two(self):
        with pytest.raises(SystemExit) as exc:
            main(["frobnicate"])
        assert exc.value.code == 2

    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0
        assert "run" in capsys.readouterr().out


class TestModuleInvocation:
    @pytest.mark.parametrize("args", [["--version"], ["info"]])
    def test_python_dash_m_works(self, args):
        proc = subprocess.run(
            [sys.executable, "-m", "microquantum", *args],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        assert proc.stdout.strip()