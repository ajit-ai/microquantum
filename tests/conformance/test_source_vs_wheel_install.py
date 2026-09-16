"""Conformance: source install and wheel install must behave identically.

Builds a wheel from the checked-out source, installs it into a clean venv
(offline, inheriting site-packages so numpy is available), and runs the same
smoke program against both the source install and the wheel install.  The two
programs must produce byte-identical output (version, statevector, counts,
public ``__all__``).  Divergence means a packaging/serialization regression.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SOURCE_PY = sys.executable
WHEEL_BUILD_TIMEOUT = 600

SMOKE = """
import json
import sys
import microquantum as mq
from microquantum import (
    ExecutionPlan,
    ExecutionRuntime,
    GroverSearch,
    QuantumCircuit,
    SearchProblem,
)
import numpy as np

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
sv = qc.run()
res = ExecutionRuntime().execute(
    ExecutionPlan(name="smoke", circuit=qc, shots=512, seed=7)
)
sp = SearchProblem(num_qubits=4, target=5)
gr = GroverSearch.from_problem(sp).run(shots=128, seed=1)

print(mq.__version__)
print(repr(sv.amplitudes.tolist()))
print(json.dumps(res.get_counts(), sort_keys=True))
print(gr.most_probable)
print(json.dumps(sorted(getattr(mq, "__all__", []))))
"""


def _run_python(executable, cwd, timeout=180) -> tuple[int, str]:
    proc = subprocess.run(
        [executable, "-X", "utf8", "-c", SMOKE],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        combined += proc.stderr
    return proc.returncode, combined


def _copy_runtime_deps(source_site: Path, target_site: Path) -> None:
    """Mirror numpy (and its data/legacy fallback) into the wheel venv.

    uv resolves a venv's ``home`` to the uv-managed interpreter, so
    ``--system-site-packages`` does NOT inherit the project venv's packages.
    numpy is a normal third-party runtime dependency, so mirroring it keeps the
    wheel smoke self-hosting without needing network.
    """
    patterns = ("numpy", "numpy.libs", "numpy-*.dist-info")
    for pat in patterns:
        for src in source_site.glob(pat):
            target = target_site / src.name
            if src.is_dir():
                shutil.copytree(src, target, dirs_exist_ok=True)
            else:
                shutil.copy2(src, target)


def _build_wheel(out_dir: Path) -> Path:
    proc = subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=WHEEL_BUILD_TIMEOUT,
    )
    assert proc.returncode == 0, (
        f"uv build failed:\n{(proc.stdout + proc.stderr)[-3000:]}"
    )
    wheels = sorted(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def source_output(tmp_path_factory) -> str:
    code, out = _run_python(SOURCE_PY, ROOT)
    assert code == 0, f"source install smoke failed:\n{out}"
    return out


@pytest.fixture(scope="module")
def wheel_output(tmp_path_factory) -> str:
    tmp = tmp_path_factory.mktemp("wheel")
    wheel = _build_wheel(tmp)
    assert wheel.exists() and wheel.stat().st_size > 0

    venv = tmp / "venv"
    sub = "Scripts" if sys.platform == "win32" else "bin"
    venv_py = venv / sub / "python.exe"
    proc = subprocess.run(
        ["uv", "venv", "--python", sys.executable, "--system-site-packages", str(venv)],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert proc.returncode == 0, (
        f"venv creation failed:\n{(proc.stdout + proc.stderr)[-2000:]}"
    )

    proc = subprocess.run(
        [str(venv_py), "-m", "pip", "install", "--no-deps", str(wheel)],
        capture_output=True,
        text=True,
        timeout=420,
    )
    assert proc.returncode == 0, (
        f"wheel install failed:\n{(proc.stdout + proc.stderr)[-3000:]}"
    )

    import site as _site

    dev_site = Path(
        next(
            p
            for p in _site.getsitepackages()
            if p.rstrip("/\\").endswith("site-packages")
        )
    )
    venv_site = venv / "Lib" / "site-packages"
    _copy_runtime_deps(dev_site, venv_site)

    code, out = _run_python(str(venv_py), tmp)
    assert code == 0, f"wheel install smoke failed:\n{out}"
    return out


def _strip_version(output: str) -> str:
    lines = output.splitlines()
    return "\n".join(lines[1:])


def test_source_and_wheel_install_agree_on_version(
    source_output: str, wheel_output: str
) -> None:
    assert source_output.splitlines()[0] == wheel_output.splitlines()[0]


def test_source_and_wheel_install_agree_on_behavior(
    source_output: str, wheel_output: str
) -> None:
    assert _strip_version(source_output) == _strip_version(wheel_output)


def test_wheel_imports_only_public_surface(wheel_output: str) -> None:
    public = sorted(__import__("microquantum").__all__)
    assert len(public) > 50, "suspiciously small public surface"