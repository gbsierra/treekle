"""Time msgspec, treekle, pickle and jsonpickle on every priml ``exp000`` config.

priml, msgspec and jsonpickle are not treekle dependencies, so run this from a
loop checkout's environment, which has all three:

  PYTHONPATH=. ~/loop/.venv/bin/python scripts/bench_configs.py --loop ~/loop

Encode times each library on the config itself; msgspec's ``to_builtins`` takes
``enc_hook=repr``, its only way past the callables the configs hold. msgspec
cannot decode the configs at all: its unions may not mix a custom type with
anything but None. So decode times the closest analogue msgspec accepts: each
config rebuilt as generated dataclasses holding only plain leaves (a callable,
type or dtype becomes its repr), and for msgspec a tagged ``msgspec.Struct``
twin. A decode that does not equal its input prints as ``≠``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import GenericAlias, ModuleType, UnionType
from typing import Protocol, cast, get_args, get_origin

import argparse
import dataclasses
import functools
import importlib
import operator
import pickle
import sys
import timeit

from treekle import deserialize, serialize


def main() -> int:
    """Print the encode and decode tables; return the process exit code."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    sys.path.insert(0, str(flags.loop))
    configs, skipped = _configs(flags.loop / "loop/priml/baselines", root=flags.loop)
    for label in skipped:
        print(f"skipped {label}: exp000 raised")
    module = ModuleType("treekle_bench_analogue")
    sys.modules[module.__name__] = module
    classes: dict[tuple[str, tuple[tuple[str, object], ...]], type] = {}
    codecs = [_CODECS[name]() for name in flags.columns]
    rows = {
        label: _row(
            config,
            _mirror(config, module=module, classes=classes),
            codecs=codecs,
            budget_sec=flags.budget_sec,
            repeat=flags.repeat,
        )
        for label, config in configs
    }
    print(_table("Encode", rows, names=flags.columns, index=0))
    print(_table("Decode (analogue)", rows, names=flags.columns, index=1))
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    loop: Path
    columns: list[str]
    budget_sec: float
    repeat: int


type _Times = dict[str, tuple[float | None, float | None]]
type _Prepared = tuple[Callable[[], object], Callable[[object], bool]]
type _Classes = dict[tuple[str, tuple[tuple[str, object], ...]], type]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Codec:
    """One library: how it encodes a config, and how it decodes an analogue.

    ``prepare`` takes an analogue and returns a zero-argument decode plus a
    check that the decode reproduced the analogue.
    """

    name: str
    encode: Callable[[object], object]
    prepare: Callable[[object], _Prepared]


class _Msgspec(Protocol):
    def to_builtins(
        self,
        obj: object,
        *,
        enc_hook: Callable[[object], object] | None = None,
    ) -> object: ...
    def convert(
        self,
        obj: object,
        type: object,
        *,
        from_attributes: bool = False,
    ) -> object: ...
    def defstruct(
        self,
        name: str,
        fields: Iterable[tuple[str, object]],
        *,
        tag: bool,
        tag_field: str,
        kw_only: bool,
    ) -> type: ...


class _FromPlain(Protocol):
    def __call__(
        self,
        data: object,
        target: object,
        *,
        allow_imports: bool,
    ) -> object: ...


class _Flattener(Protocol):
    def flatten(self, obj: object) -> object: ...


class _Restorer(Protocol):
    def restore(self, obj: object) -> object: ...


class _Jsonpickle(Protocol):
    Pickler: Callable[..., _Flattener]
    Unpickler: Callable[..., _Restorer]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--loop",
        type=Path,
        required=True,
        help="The loop checkout holding priml.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        choices=list(_CODECS),
        default=list(_CODECS),
        help="Libraries to time; percent columns are relative to msgspec.",
    )
    parser.add_argument(
        "--budget-sec",
        type=float,
        default=0.2,
        help="Least time one timing repeat runs.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=7,
        help="Repeats; the fastest counts.",
    )


def _configs(
    baselines: Path,
    *,
    root: Path,
) -> tuple[list[tuple[str, object]], list[str]]:
    """Build every ``exp000`` under ``baselines``; return them and those that raised."""
    configs: list[tuple[str, object]] = []
    skipped: list[str] = []
    for path in sorted(baselines.rglob("experiments*.py")):
        if path.name.endswith("_test.py") or "def exp000(" not in path.read_text():
            continue
        name = ".".join(path.relative_to(root).with_suffix("").parts)
        label = str(path.relative_to(baselines).with_suffix("")).removesuffix(
            "/experiments",
        )
        try:
            module = importlib.import_module(name)
            configs.append((label, cast("Callable[[], object]", module.exp000)()))
        except Exception:  # noqa: BLE001 -- A baseline that cannot build is skipped.
            skipped.append(label)
    return configs, skipped


def _mirror(value: object, *, module: ModuleType, classes: _Classes) -> object:
    """Rebuild ``value`` from generated dataclasses in ``module`` and plain leaves."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: _mirror(
                cast("object", getattr(value, field.name)),
                module=module,
                classes=classes,
            )
            for field in dataclasses.fields(value)
        }
        annotations = tuple((name, _annotation(item)) for name, item in fields.items())
        key = (type(value).__name__, annotations)
        if key not in classes:
            made = dataclasses.make_dataclass(
                f"{key[0].replace('.', '_')}_{len(classes)}",
                annotations,
                module=module.__name__,
                slots=True,
                kw_only=True,
            )
            setattr(module, made.__name__, made)
            classes[key] = made
        return cast("object", classes[key](**fields))
    if isinstance(value, list | tuple):
        sequence = cast("list[object] | tuple[object, ...]", value)
        items = [_mirror(item, module=module, classes=classes) for item in sequence]
        return items if isinstance(sequence, list) else tuple(items)
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {
            key if type(key) in {str, int} else repr(key): _mirror(
                item,
                module=module,
                classes=classes,
            )
            for key, item in mapping.items()
        }
    if value is None or type(value) in {bool, int, float, str}:
        return value
    return repr(value)


def _annotation(value: object) -> object:
    """Return the narrowest type of a mirrored value that msgspec can decode."""
    if value is None:
        return type(None)
    if isinstance(value, list):
        return GenericAlias(list, (_union(cast("list[object]", value)),))
    if isinstance(value, tuple):
        items = cast("tuple[object, ...]", value)
        return GenericAlias(tuple, tuple(map(_annotation, items)))
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return GenericAlias(dict, (_union(mapping), _union(mapping.values())))
    return type(value)


def _union(values: Iterable[object]) -> object:
    """Join the values' types; ``object`` where msgspec forbids the union."""
    members: list[object] = []
    for value in values:
        annotation = _annotation(value)
        if annotation not in members:
            members.append(annotation)
    origins = [get_origin(member) for member in members]
    arrays = sum(origin in {list, tuple} for origin in origins)
    mappings = origins.count(dict) + sum(
        isinstance(member, type) and dataclasses.is_dataclass(member)
        for member in members
    )
    if not members or arrays > 1 or (dict in origins and mappings > 1):
        return object
    return functools.reduce(_join, members)


def _join(left: object, right: object) -> object:
    """Return the union type ``left | right``."""
    return cast("Callable[[object, object], object]", operator.or_)(left, right)


def _row(
    config: object,
    analogue: object,
    *,
    codecs: list[_Codec],
    budget_sec: float,
    repeat: int,
) -> _Times:
    """Time each codec's encode of ``config`` and decode of ``analogue``."""
    times: _Times = {}
    for codec in codecs:
        decode, check = codec.prepare(analogue)
        encode = functools.partial(codec.encode, config)
        times[codec.name] = (
            _best_us(encode, budget_sec=budget_sec, repeat=repeat),
            _best_us(decode, budget_sec=budget_sec, repeat=repeat)
            if check(decode())
            else None,
        )
    return times


def _best_us(fn: Callable[[], object], *, budget_sec: float, repeat: int) -> float:
    """Return the fastest per-call microseconds over ``repeat`` timed batches."""
    timer = timeit.Timer(fn)
    number = 1
    while (elapsed := timer.timeit(number)) < budget_sec:
        number *= 2
    batches = [elapsed, *timer.repeat(repeat=repeat - 1, number=number)]
    return min(batches) / number * 1e6


def _table(title: str, rows: dict[str, _Times], *, names: list[str], index: int) -> str:
    """Render one markdown table: microseconds, then percent of msgspec's time."""
    relative = [n for n in names if n != "msgspec"] if "msgspec" in names else []
    header = [*names, *(f"{name} %" for name in relative)]
    lines = [
        f"**{title}** (µs per call)\n",
        "| config | " + " | ".join(header) + " |",
        "|---|" + "---:|" * len(header),
    ]
    cells = {
        label: {n: times[n][index] for n in names} for label, times in rows.items()
    }
    cells["**total**"] = {
        n: None
        if any(row[n] is None for row in cells.values())
        else sum(row[n] or 0.0 for row in cells.values())
        for n in names
    }
    for label, row in cells.items():
        values = [_cell(row[n], "{:.1f}") for n in names]
        percents = [
            _cell(_percent(row[n], row["msgspec"]), "{:.0f}%") for n in relative
        ]
        lines.append(f"| {label} | " + " | ".join([*values, *percents]) + " |")
    return "\n".join(lines) + "\n"


def _percent(value: float | None, base: float | None) -> float | None:
    return None if value is None or base is None or base == 0 else 100 * value / base


def _cell(value: float | None, spec: str) -> str:
    return "≠" if value is None else spec.format(value)


def _treekle_codec(
    name: str,
    *,
    to_plain: Callable[[object], object],
    from_plain: _FromPlain,
) -> _Codec:
    prepare = functools.partial(
        _prepare_treekle,
        to_plain=to_plain,
        from_plain=from_plain,
    )
    return _Codec(name=name, encode=to_plain, prepare=prepare)


def _prepare_treekle(
    analogue: object,
    *,
    to_plain: Callable[[object], object],
    from_plain: _FromPlain,
) -> _Prepared:
    tree, target = to_plain(analogue), type(analogue)
    return lambda: from_plain(tree, target, allow_imports=True), analogue.__eq__


def _pickle_codec() -> _Codec:
    return _Codec(name="pickle", encode=pickle.dumps, prepare=_prepare_pickle)


def _prepare_pickle(analogue: object) -> _Prepared:
    data = pickle.dumps(analogue)
    return lambda: cast("object", pickle.loads(data)), analogue.__eq__  # noqa: S301


def _jsonpickle_codec() -> _Codec:
    library = cast("_Jsonpickle", cast("object", importlib.import_module("jsonpickle")))
    return _Codec(
        name="jsonpickle",
        encode=lambda config: library.Pickler(keys=True).flatten(config),
        prepare=functools.partial(_prepare_jsonpickle, library=library),
    )


def _prepare_jsonpickle(analogue: object, *, library: _Jsonpickle) -> _Prepared:
    flat = library.Pickler(keys=True).flatten(analogue)
    return lambda: library.Unpickler(keys=True).restore(flat), analogue.__eq__


def _msgspec_codec() -> _Codec:
    library = cast("_Msgspec", cast("object", importlib.import_module("msgspec")))
    return _Codec(
        name="msgspec",
        encode=lambda config: library.to_builtins(config, enc_hook=repr),
        prepare=functools.partial(_prepare_msgspec, library=library, structs={}),
    )


def _prepare_msgspec(
    analogue: object,
    *,
    library: _Msgspec,
    structs: dict[object, type],
) -> _Prepared:
    target: object = _struct(type(analogue), library=library, structs=structs)
    built = _tagged(analogue)
    decode = functools.partial(library.convert, built, target)
    return decode, lambda struct: library.to_builtins(struct) == built


def _tagged(value: object) -> object:
    """Return ``value`` as builtins, each dataclass a dict tagged with its Struct name."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: _tagged(cast("object", getattr(value, field.name)))
            for field in dataclasses.fields(value)
        }
        return {"$type": type(value).__name__, **fields}
    if isinstance(value, list | tuple):
        sequence = cast("list[object] | tuple[object, ...]", value)
        items = [_tagged(item) for item in sequence]
        return items if isinstance(sequence, list) else tuple(items)
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {key: _tagged(item) for key, item in mapping.items()}
    return value


def _struct(
    annotation: object,
    *,
    library: _Msgspec,
    structs: dict[object, type],
) -> object:
    """Swap every generated dataclass in ``annotation`` for its tagged Struct twin."""
    recurse = functools.partial(_struct, library=library, structs=structs)
    if isinstance(annotation, UnionType):
        return functools.reduce(_join, map(recurse, get_args(annotation)))
    if isinstance(annotation, GenericAlias):
        arguments = tuple(map(recurse, get_args(annotation)))
        return GenericAlias(get_origin(annotation), arguments)
    if annotation in structs:
        return structs[annotation]
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        structs[annotation] = library.defstruct(
            annotation.__name__,
            [(f.name, recurse(f.type)) for f in dataclasses.fields(annotation)],
            tag=True,
            tag_field="$type",
            kw_only=True,
        )
        return structs[annotation]
    return cast("object", annotation)


_CODECS: dict[str, Callable[[], _Codec]] = {
    "msgspec": _msgspec_codec,
    "treekle-rust": functools.partial(
        _treekle_codec,
        "treekle-rust",
        to_plain=serialize.to_plain,
        from_plain=deserialize.from_plain,
    ),
    "treekle-python": functools.partial(
        _treekle_codec,
        "treekle-python",
        to_plain=serialize._python_to_plain,  # noqa: SLF001
        from_plain=deserialize._python_from_plain,  # noqa: SLF001
    ),
    "pickle": _pickle_codec,
    "jsonpickle": _jsonpickle_codec,
}


if __name__ == "__main__":
    raise SystemExit(main())
