"""Check that the installed treekle runs the implementation its wheel should.

Run as ``python -P -W error scripts/check_wheel.py native|python`` in an
environment with the wheel installed. ``-P`` keeps the checkout off
``sys.path``; ``-W error`` turns the Rust core's fallback warning into a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import importlib.util
import sys

from treekle import deserialize, serialize


@dataclass(frozen=True)
class _Point:
    x: int
    y: float


@dataclass(frozen=True)
class _Shape:
    name: str
    corners: tuple[_Point, ...]


def main() -> None:
    """Exit non-zero unless both functions are the expected implementation."""
    if len(sys.argv) != 2 or sys.argv[1] not in {"native", "python"}:
        raise SystemExit("usage: check_wheel.py native|python")
    native = sys.argv[1] == "native"
    core = importlib.util.find_spec("treekle._rust")
    if native != (core is not None):
        raise SystemExit(f"treekle._rust is {'missing' if native else 'present'}")
    if native:
        from treekle import _rust  # noqa: PLC0415 -- Only a native wheel has it.

        expected = (_rust.to_plain, _rust.from_plain)
    else:
        expected = (serialize._python_to_plain, deserialize._python_from_plain)  # noqa: SLF001
    actual = (serialize.to_plain, deserialize.from_plain)
    if any(found is not wanted for found, wanted in zip(actual, expected, strict=True)):
        raise SystemExit(f"treekle does not run its {sys.argv[1]} implementation")
    shape = _Shape("square", (_Point(0, 0.0), _Point(1, 0.5)))
    if deserialize.from_plain(serialize.to_plain(shape), _Shape) != shape:
        raise SystemExit("a dataclass does not round-trip")
    print(f"{sys.argv[1]}: {core.origin if core else serialize.__file__}")


if __name__ == "__main__":
    main()
