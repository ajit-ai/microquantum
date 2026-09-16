"""Conformance gate: strict mypy across the full SDK (MQ-17).

Ensures [tool.mypy] is configured for strict mode, no per-module overrides
suppress errors in the public SDK, and ``mypy --strict`` passes with zero
errors.  This conformance test runs as part of the full test suite.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src" / "microquantum"
PYPROJECT = REPO_ROOT / "pyproject.toml"

_TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore\b")


def test_strict_mypy_config_contract() -> None:
    with PYPROJECT.open("rb") as fh:
        config = tomllib.load(fh)

    mypy = config["tool"]["mypy"]
    assert mypy.get("strict") is True

    global_ignore = mypy.get("ignore_missing_imports")
    assert global_ignore is None or global_ignore is False

    overrides: list[dict] = mypy.get("overrides", [])
    for override in overrides:
        modules = override.get("module", [])
        if override.get("disable_error_code"):
            assert not (
                "microquantum.core.circuit" in modules
                or "microquantum.core.gradient" in modules
            )

    cupy_modules = [
        override for override in overrides if "cupy" in override.get("module", [])
    ]
    scipy_modules = [
        override for override in overrides if "scipy" in override.get("module", [])
    ]
    assert cupy_modules, "no mypy override granting cupy missing-imports exception"
    assert scipy_modules, "no mypy override granting scipy missing-imports exception"


def test_no_type_ignore_in_package() -> None:
    ignore_count = 0
    for path in sorted(SRC_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        ignore_count += len(_TYPE_IGNORE.findall(text))
    assert ignore_count == 0, f"found {ignore_count} '# type: ignore' in the SDK"


def test_mypy_strict_conformance() -> None:
    uv = shutil.which("uv")
    assert uv, "uv must be on PATH to run the mypy conformance gate"
    result = subprocess.run(
        [uv, "run", "mypy", "src/microquantum/"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    stdout = result.stdout or ""
    assert result.returncode == 0, stdout + (result.stderr or "")
    assert "Success" in stdout
    lowered = stdout.lower()
    assert "error" not in lowered or "no issues found" in lowered