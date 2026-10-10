"""Treekle: pickle semantics as plain text.

``to_plain`` turns any Python object graph -- dataclasses, functions, shared
references, cycles -- into a plain tree of dicts, lists, and scalars that any
JSON, TOML, or YAML writer can store. ``from_plain`` turns it back, building the
classes the tree names, and reads bad input best-effort: the error carries the
partial value and the location of every bad part. See :mod:`treekle.codec`.
"""

from __future__ import annotations

from treekle.codec import (
    Dialect,
    FieldPath,
    Hooks,
    Invalid,
    MutablePlainTree,
    Plain,
    PlainTree,
    ReadError,
    from_plain,
    immutable,
    loads,
    mutable,
    parse,
    to_plain,
)


__all__ = [
    "Dialect",
    "FieldPath",
    "Hooks",
    "Invalid",
    "MutablePlainTree",
    "Plain",
    "PlainTree",
    "ReadError",
    "from_plain",
    "immutable",
    "loads",
    "mutable",
    "parse",
    "to_plain",
]
