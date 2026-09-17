"""Minimal, dependency-light command-line interface for the MicroQuantum runtime.

The CLI is a thin developer-facing wrapper around the *same canonical public
Python APIs* used by normal programs — it is not an execution engine:

* ``info``      — runtime introspection (:func:`~microquantum.runtime.runtime_info`);
* ``backends``  — registered backends with capability summaries
  (:data:`~microquantum.backends.registry.default_registry`);
* ``run``       — execute an OpenQASM 2.0 file through
  :func:`~microquantum.runtime.execute`.
* ``version`` / ``--version`` — package version.

Only the Python standard library is used (``argparse``).  Ordinary user
errors print a one-line message and a non-zero exit status; ``--debug``
restores the full traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .backends.registry import default_registry
from .core.qasm import from_qasm
from .runtime import execute, runtime_info
from .runtime.info import _MQ_VERSION

_PROG = "microquantum"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="MicroQuantum quantum SDK developer tooling.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the MicroQuantum version and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show full tracebacks on failure",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "version",
        help="print the MicroQuantum version",
    )
    subparsers.add_parser(
        "info",
        help="show runtime implementation information",
    )
    backends = subparsers.add_parser(
        "backends",
        help="list registered backends with their capabilities",
    )
    backends.add_argument("--json", action="store_true", help="print JSON output")

    run = subparsers.add_parser(
        "run",
        help="execute an OpenQASM 2.0 circuit file and print measurement counts",
    )
    run.add_argument("file", help="path to an OpenQASM 2.0 circuit file")
    run.add_argument("--shots", type=int, default=1024, help="number of shots")
    run.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    run.add_argument(
        "--backend",
        default=None,
        help="registered backend name (defaults to the runtime default)",
    )
    run.add_argument(
        "--optimization-level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="compiler optimization level (0-2)",
    )
    return parser


def _version_string() -> str:
    return f"{_PROG} {_MQ_VERSION}"


def _print_version() -> None:
    print(_version_string())


def _cmd_info() -> int:
    info = runtime_info()
    print(f"MicroQuantum {info.version}")
    print(f"Runtime:   {info.runtime}")
    print(f"Python:    {info.python}")
    print(f"NumPy:     {info.numpy}")
    print(f"Strategies: {', '.join(info.strategies)}")
    print(f"Backends:  {', '.join(b['name'] for b in info.backends) or '(none)'}")
    print(f"Default:   {info.default_backend or '(none)'}")
    return 0


def _cmd_backends(json_output: bool) -> int:
    if json_output:
        print(json.dumps(default_registry.to_dict(), indent=2, sort_keys=True))
        return 0
    backends = default_registry.list()
    if not backends:
        print("No backends registered.")
        return 0
    for backend in backends:
        caps = backend.capabilities
        execution = ", ".join(sorted(caps.execution))
        max_qubits = caps.max_qubits if caps.max_qubits is not None else "unbounded"
        print(f"{backend.name}  ({caps.target_class.value}, max {max_qubits} qubits)")
        print(f"  execution: {execution}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    source = Path(args.file).read_text(encoding="utf-8")
    circuit = from_qasm(source)
    result = execute(
        circuit,
        shots=args.shots,
        seed=args.seed,
        backend=args.backend,
        optimization_level=args.optimization_level,
    )
    print(f"Backend:     {result.backend_name}")
    print(f"Shots:       {result.shots}")
    print(f"Distinct:    {len(result.counts)}")
    if result.counts:
        for bitstring in sorted(result.counts.keys()):
            print(f"  |{bitstring}>: {result.counts[bitstring]}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point returning the process exit status (0 on success)."""
    parser = _build_parser()
    args: Any = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.version or args.command == "version":
            _print_version()
            return 0
        if args.command is None:
            parser.print_help()
            return 0
        if args.command == "info":
            return _cmd_info()
        if args.command == "backends":
            return _cmd_backends(bool(args.json))
        if args.command == "run":
            return _cmd_run(args)
        parser.print_help()
        return 2
    except SystemExit:
        raise
    except Exception as exc:
        if args.debug:
            raise
        print(f"{_PROG}: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())