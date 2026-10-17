"""The config benchmark builds analogues every library round-trips, and renders both tables.

msgspec and jsonpickle are not treekle dependencies, so stand-ins replace them:
jsonpickle by pickle, msgspec by a converter that builds the requested Struct.
"""

from __future__ import annotations

from pathlib import Path
from types import GenericAlias, SimpleNamespace, UnionType
from typing import TYPE_CHECKING, cast, get_args, get_origin

import dataclasses
import functools
import importlib.util
import pickle
import sys
import textwrap

import pytest


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


_TOY = """
from dataclasses import dataclass, field


@dataclass
class Leaf:
    x: int = 1


@dataclass
class Toy:
    name: str = "toy"
    rate: float = 0.5
    leaf: Leaf = field(default_factory=Leaf)
    leaves: list[object] = field(default_factory=lambda: [Leaf(), None])
    mixed: list[object] = field(default_factory=lambda: [[1], (2,)])
    maps: list[object] = field(default_factory=lambda: [{"a": 1}, Leaf()])
    empty: list[object] = field(default_factory=list)
    pair: tuple[object, ...] = (1, Leaf())
    unit: tuple[()] = ()
    keys: dict[object, int] = field(default_factory=lambda: {1: 2, (3, 4): 5})
    fn: object = len


def exp000():
    return Toy()
"""


@pytest.fixture
def bench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[..., None]]:
    """Return a runner of the benchmark over a fake loop checkout."""
    baselines = tmp_path / "loop/priml/baselines"
    for name, body in {
        "toy": _TOY,
        "nan": "def exp000():\n    return {'x': float('nan')}\n",
        "broken": "def exp000():\n    raise RuntimeError\n",
    }.items():
        (baselines / name).mkdir(parents=True)
        (baselines / name / "experiments.py").write_text(textwrap.dedent(body))
    (baselines / "toy/experiments_test.py").write_text("def exp000(): ...\n")
    monkeypatch.setattr(sys, "path", [*sys.path])
    monkeypatch.setitem(sys.modules, "msgspec", _fake_msgspec())
    monkeypatch.setitem(sys.modules, "jsonpickle", _fake_jsonpickle())
    spec = importlib.util.spec_from_file_location(
        "bench_configs",
        Path(__file__).with_name("bench_configs.py"),
    )
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    main = cast("Callable[[], int]", vars(module)["main"])
    before = set(sys.modules)

    def run(*columns: str) -> None:
        argv = [
            "bench",
            "--loop",
            str(tmp_path),
            "--budget-sec",
            "1e-4",
            "--repeat",
            "2",
        ]
        monkeypatch.setattr(sys, "argv", [*argv, "--columns", *columns])
        assert main() == 0

    yield run
    for name in set(sys.modules) - before:
        del sys.modules[name]


def test_every_library_round_trips_the_analogue(
    bench: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench("msgspec", "treekle-rust", "treekle-python", "pickle", "jsonpickle")
    out = capsys.readouterr().out
    assert "skipped broken: exp000 raised" in out
    encode, decode = out.split("**Decode (analogue)**")
    header = "| config | msgspec | treekle-rust | treekle-python | pickle | jsonpickle | treekle-rust % | treekle-python % | pickle % | jsonpickle % |"
    assert header in encode
    assert header in decode
    toy = next(line for line in decode.splitlines() if line.startswith("| toy |"))
    assert "≠" not in toy
    assert toy.endswith("% |")
    nan = next(line for line in decode.splitlines() if line.startswith("| nan |"))
    # The msgspec stand-in hands back the same float object, which equals itself.
    assert nan.count("≠") == 8
    total = next(line for line in decode.splitlines() if "**total**" in line)
    assert total.count("≠") == 8
    assert "≠" not in encode


def test_without_msgspec_the_tables_have_no_percent_columns(
    bench: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    bench("treekle-rust", "pickle")
    out = capsys.readouterr().out
    assert "| config | treekle-rust | pickle |\n" in out
    assert "%" not in out


def _fake_jsonpickle() -> SimpleNamespace:
    pickler = SimpleNamespace(flatten=pickle.dumps)
    unpickler = SimpleNamespace(restore=pickle.loads)
    return SimpleNamespace(
        Pickler=functools.partial(_keyed, pickler),
        Unpickler=functools.partial(_keyed, unpickler),
    )


def _keyed(made: SimpleNamespace, *, keys: bool) -> SimpleNamespace:
    assert keys
    return made


def _fake_msgspec() -> SimpleNamespace:
    return SimpleNamespace(
        to_builtins=_to_builtins,
        convert=_convert,
        defstruct=_defstruct,
    )


def _to_builtins(
    obj: object,
    *,
    enc_hook: Callable[[object], object] | None = None,
) -> object:
    if enc_hook is not None:
        return enc_hook(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {
            f.name: _to_builtins(cast("object", getattr(obj, f.name)))
            for f in dataclasses.fields(obj)
        }
        return {"$type": type(obj).__name__, **fields}
    if isinstance(obj, list | tuple):
        items = [_to_builtins(item) for item in cast("list[object]", obj)]
        return items if isinstance(obj, list) else tuple(items)
    if isinstance(obj, dict):
        return {
            k: _to_builtins(v) for k, v in cast("dict[object, object]", obj).items()
        }
    return obj


def _defstruct(
    name: str,
    fields: Iterable[tuple[str, object]],
    **options: object,
) -> type:
    assert options == {"tag": True, "tag_field": "$type", "kw_only": True}
    return dataclasses.make_dataclass(name, list(fields), kw_only=True)


def _convert(value: object, target: object) -> object:
    """Rebuild ``value`` as ``target``, which names Struct twins by class name."""
    if isinstance(target, type) and dataclasses.is_dataclass(target):
        fields = {
            field.name: _convert(
                cast("dict[str, object]", value)[field.name],
                cast("object", field.type),
            )
            for field in dataclasses.fields(target)
        }
        return cast("object", target(**fields))
    if isinstance(target, UnionType):
        members = cast("tuple[type, ...]", get_args(target))
        named: dict[str, object] = {member.__name__: member for member in members}
        return _convert(value, named.get(type(value).__name__, object))
    if isinstance(target, GenericAlias):
        args = cast("tuple[object, ...]", get_args(target))
        if isinstance(value, dict):
            mapping = cast("dict[object, object]", value)
            return {key: _convert(item, args[1]) for key, item in mapping.items()}
        items = cast("list[object] | tuple[object, ...]", value)
        if get_origin(target) is tuple:
            return tuple(
                _convert(item, arg) for item, arg in zip(items, args, strict=True)
            )
        return [_convert(item, args[0]) for item in items]
    return value
