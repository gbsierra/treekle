"""Tests that the Rust core gives what the pure-Python functions give.

``serialize`` keeps its Python version as ``_python_to_plain``; each case
here runs both and compares results, container types included, or the errors
they raise.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, IntEnum
from fractions import Fraction
from functools import partial
from pathlib import Path, PurePosixPath
from types import MappingProxyType, MemberDescriptorType
from typing import (
    TYPE_CHECKING,
    NamedTuple,
    cast,
    override,
)
from uuid import UUID
from zoneinfo import ZoneInfo

import datetime as dt
import gc
import inspect
import math
import os
import subprocess
import sys
import textwrap
import weakref

import pytest

from treekle import deserialize, serialize


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from treekle.types import Hooks


# Without the Rust core each case compares the Python versions with
# themselves, which still runs this file; only Rust-specific limits skip.
needs_rust = pytest.mark.skipif(
    serialize.to_plain is serialize._python_to_plain,
    reason="the Rust core is not built, or TREEKLE_PURE_PYTHON is set",
)


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Level(IntEnum):
    LOW = 1


class Pair(NamedTuple):
    a: int
    b: list[int]


@dataclass
class Point:
    x: int = 0
    y: float = 0.0


@dataclass
class Node:
    name: str
    children: list[Node] = field(default_factory=list["Node"])
    parent: Node | None = None


class Slotted:
    __slots__ = ("__dict__", "a")

    def __init__(self, a: int, b: int) -> None:
        self.a = a
        self.b = b


@dataclass(slots=True)
class AttributeOrder:
    z: int = 1
    a: int = 2


class Bare:
    def __init__(self, x: int) -> None:
        self.x = x


class Opaque:
    __slots__ = ()

    @override
    def __reduce_ex__(self, protocol: object) -> str | tuple[object, ...]:
        raise TypeError("no")


class MyDict(dict[str, int]):
    pass


class MyList(list[int]):
    pass


class Tensor:
    def __init__(self, data: list[float]) -> None:
        self.data = data


@dataclass(frozen=True)
class Frozen:
    x: int


@dataclass(slots=True)
class ChangingSlot:
    a: Tensor | int
    b: int


@dataclass
class ChangingDict:
    a: Tensor
    b: int


def module_function() -> None:
    pass


def _tensor_data(tensor: Tensor) -> list[float]:
    return tensor.data


def _tensor_tagged(tensor: Tensor) -> dict[str, list[float]]:
    return {"py/x": tensor.data}


@dataclass(slots=True)
class SlotPoint:
    x: int = 0
    y: list[int] = field(default_factory=list[int])


@dataclass(slots=True)
class SlotDynamic:
    a: int = 0
    b: int = 0

    def __getattr__(self, name: str) -> object:
        if name == "b":
            return "from getattr"
        raise AttributeError(name)


class SlotShadow(SlotPoint):
    """A property that shadows a slot: ``getattr`` reads the property."""

    __slots__ = ()

    @property
    @override
    def x(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]  # ty: ignore[invalid-property-type-override]
        return 99


class ForeignSlotOwner:
    __slots__ = ("x", "y")

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y


class ForeignSlotSource:
    __slots__ = ("a", "b", "c")

    def __init__(self) -> None:
        self.a = 0
        self.b = 0
        self.c = 0


def _unset_point() -> SlotPoint:
    point = SlotPoint()
    del point.x
    return point


def _unset_dynamic() -> SlotDynamic:
    dynamic = SlotDynamic()
    del dynamic.b
    return dynamic


def _shadow() -> SlotShadow:
    shadow = SlotShadow.__new__(SlotShadow)
    cast("MemberDescriptorType", vars(SlotPoint)["x"]).__set__(shadow, 1)
    cast("MemberDescriptorType", vars(SlotPoint)["y"]).__set__(shadow, [2])
    return shadow


def _cycle() -> list[object]:
    cycle: list[object] = []
    cycle.append(cycle)
    return cycle


def _dict_cycle() -> dict[str, object]:
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    return cycle


def _tree() -> Node:
    root = Node("root")
    root.children.append(Node("kid", parent=root))
    return root


class Key:
    """A dict key reduced to an instance of a class made afresh each time.

    A key's own encoding is dropped once its text is made, so each state
    object and its class die before the next key, whose class may then take
    the freed address.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    @override
    def __reduce__(self) -> tuple[object, ...]:
        gc.collect()
        name = f"slot{self.n}"
        state = type("Temp", (), {"__slots__": (name,)})()
        setattr(state, name, self.n)
        return (Key, (self.n,), state)


def _nested(depth: int) -> list[object]:
    root: list[object] = []
    current = root
    for _ in range(depth):
        child: list[object] = []
        current.append(child)
        current = child
    return root


SHARED = [1, 2]
PROXY = MappingProxyType({"a": SHARED, "b": SHARED})

# (id, value, dialect, hooks); every case runs with ``mutable`` on and off.
ENCODE_CASES: list[tuple[str, object, object, Hooks | None]] = [
    ("leaves", [None, True, 1, "s", 1.5, -0.0], "json", None),
    ("nonfinite json", [math.inf, -math.inf, math.nan], "json", None),
    ("nonfinite python", [math.inf, math.nan], "python", None),
    ("complex", [1 + 2j, complex(math.inf, math.nan)], "json", None),
    ("bytes", [b"abc", bytearray(b"xy")], "json", None),
    ("tuple", (1, (2, 3), []), "json", None),
    ("set ints", {3, 10, -1, 2}, "json", None),
    ("set mixed", {"b", "a", 1, (1, 2), None, 2.5}, "json", None),
    ("frozenset", frozenset({frozenset({1}), "x"}), "json", None),
    ("set and shared list", [SHARED, {(1, 2), (3,)}], "json", None),
    (
        "dict keys",
        {1: "a", None: "b", "py/x": 1, "json://y": 2, (1, 2): 3, 2.5: 5},
        "json",
        None,
    ),
    ("bool keys", {True: "a", False: "b"}, "json", None),
    ("int keys", {-12: "a", 2**100: "b", 0: "c"}, "json", None),
    ("shared", [SHARED, SHARED, {"k": SHARED}], "json", None),
    ("cycle", _cycle(), "json", None),
    ("dict cycle", _dict_cycle(), "json", None),
    ("dataclass tree", _tree(), "json", None),
    ("proxy", [PROXY, PROXY], "json", None),
    ("enum", [Color.RED, Color.RED, Level.LOW], "json", None),
    ("decimal fraction", [Decimal("1.5"), Fraction(1, 3)], "json", None),
    ("namedtuple", Pair(1, [2]), "json", None),
    ("partial", partial(int, base=2), "json", None),
    ("path uuid", [Path("/a/b"), PurePosixPath("x/y"), UUID(int=5)], "json", None),
    (
        "datetime",
        [
            dt.datetime(2026, 1, 2, tzinfo=dt.UTC),
            dt.datetime(2026, 1, 2, tzinfo=ZoneInfo("Europe/Paris")),
            dt.date(2026, 1, 2),
        ],
        "json",
        None,
    ),
    (
        "types and functions",
        [int, Point, module_function, max, len, math.floor],
        "json",
        None,
    ),
    ("bound method", [list[int]().append], "json", None),
    ("slotted with dict", Slotted(1, 2), "json", None),
    ("plain objects", [Bare(1), Bare(2)], "json", None),
    ("subclasses", [MyDict(a=1), MyList([1, 2]), OrderedDict(b=1, a=2)], "json", None),
    (
        "hook",
        [Tensor([1.0, math.inf]), Tensor([2.0])],
        "json",
        {Tensor: (_tensor_data, Tensor)},
    ),
    (
        "hook in proxy",
        Tensor([1.0]),
        "json",
        MappingProxyType({Tensor: (_tensor_tagged, Tensor)}),
    ),
    ("opaque", Opaque(), "json", None),
    ("local class", type("Local", (), {})(), "json", None),
    ("lambda", lambda: 0, "json", None),
    ("bad dialect", 1, "yaml", None),
    ("bad dialect type", 1, 3, None),
    ("object", object(), "json", None),
    ("range slice", [range(3), slice(1, 2)], "json", None),
    ("slots", [SlotPoint(1, [2]), _unset_point(), SlotPoint(3)], "json", None),
    ("slots and getattr", [SlotDynamic(1, 2), _unset_dynamic()], "json", None),
    ("slot shadowed by property", _shadow(), "json", None),
    ("lone surrogate tag keys", {"py/\ud800": 1, "json://\udfff": 2}, "json", None),
    (
        "keys reduced to short-lived classes",
        {Key(n): n for n in range(20)},
        "json",
        None,
    ),
]


def _shape(tree: object) -> object:
    """Return ``tree`` with each container's exact type made visible."""
    name = type(tree).__name__
    if isinstance(tree, (dict, MappingProxyType)):
        items = cast("dict[object, object]", tree).items()
        return (name, [(key, _shape(value)) for key, value in items])
    if isinstance(tree, (list, tuple)):
        return (
            name,
            [_shape(item) for item in cast("list[object]", tree)],
        )
    if isinstance(tree, float) and math.isnan(tree):
        return ("nan",)
    return (type(tree).__name__, tree)


# Each place links to its parent's, so placing a node costs one tuple.
type _Place = tuple[_Place, object] | None


def _path(place: _Place) -> list[object]:
    """Return the keys leading to ``place``."""
    keys: list[object] = []
    while place is not None:
        place, key = place
        keys.append(key)
    return keys[::-1]


def _assert_same_shape(actual: object, expected: object) -> None:
    """Assert ``_shape(actual) == _shape(expected)`` for trees of any depth."""
    pending: list[tuple[object, object, _Place]] = [(actual, expected, None)]
    while pending:
        left, right, at = pending.pop()
        assert type(left) is type(right), _path(at)
        if isinstance(left, (dict, MappingProxyType)):
            left_items = list(cast("Mapping[object, object]", left).items())
            right_items = list(cast("Mapping[object, object]", right).items())
            assert [key for key, _ in left_items] == [key for key, _ in right_items], (
                _path(at)
            )
            pending.extend(
                (one, other, (at, key))
                for (key, one), (_, other) in zip(left_items, right_items, strict=True)
            )
        elif isinstance(left, (list, tuple)):
            left_list = cast("Sequence[object]", left)
            right_list = cast("Sequence[object]", right)
            assert len(left_list) == len(right_list), _path(at)
            pending.extend(
                (one, other, (at, index))
                for index, (one, other) in enumerate(
                    zip(left_list, right_list, strict=True),
                )
            )
        elif isinstance(left, float) and math.isnan(left):
            assert isinstance(right, float), _path(at)
            assert math.isnan(right), _path(at)
        else:
            assert left == right, _path(at)


def _outcome(call: Callable[[], object]) -> object:
    """Return the call's result shape, or the type and text of what it raised."""
    try:
        return ("ok", _shape(call()))
    except Exception as error:  # noqa: BLE001
        return ("raise", type(error).__name__, str(error))


def _encoder(
    function: Callable[..., object],
    value: object,
    *,
    mutable: bool,
    dialect: object,
    hooks: Hooks | None,
) -> Callable[[], object]:
    options: dict[str, object] = {"mutable": mutable, "dialect": dialect}
    if hooks is not None:
        options["hooks"] = hooks
    return lambda: function(value, **options)


@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
@pytest.mark.parametrize(
    ("value", "dialect", "hooks"),
    [case[1:] for case in ENCODE_CASES],
    ids=[case[0] for case in ENCODE_CASES],
)
def test_to_plain_matches_python(
    value: object,
    dialect: object,
    hooks: Hooks | None,
    mutable: bool,
) -> None:
    python = _outcome(
        _encoder(
            serialize._python_to_plain,
            value,
            mutable=mutable,
            dialect=dialect,
            hooks=hooks,
        ),
    )
    rust = _outcome(
        _encoder(
            serialize.to_plain,
            value,
            mutable=mutable,
            dialect=dialect,
            hooks=hooks,
        ),
    )
    assert rust == python


@pytest.mark.parametrize(
    ("sort", "names"),
    [(True, ["py/object", "a", "z"]), (False, ["py/object", "z", "a"])],
)
def test_object_attribute_sort_matches_python(sort: bool, names: list[str]) -> None:
    value = AttributeOrder()
    expected = serialize._python_to_plain(value, sort=sort)

    assert isinstance(expected, dict)
    assert list(expected) == names
    assert serialize.to_plain(value, sort=sort) == expected


class _Scrambled(str):
    __slots__ = ()

    @override
    def __lt__(self, other: object) -> bool:
        return (sum(map(ord, self)) * 31 + sum(map(ord, str(other)))) % 3 == 0


class _Unordered(str):
    __slots__ = ()

    @override
    def __lt__(self, other: object) -> bool:
        raise ValueError("unordered")


@dataclass(frozen=True, repr=False)
class _ShownScrambled:
    n: int

    @override
    def __repr__(self) -> str:
        return _Scrambled(f"m{self.n}")


@dataclass(frozen=True, repr=False)
class _ShownUnordered:
    n: int

    @override
    def __repr__(self) -> str:
        return _Unordered(f"m{self.n}")


@pytest.mark.parametrize("kind", [_ShownScrambled, _ShownUnordered])
def test_set_members_with_inconsistent_repr_order_match_python(
    kind: Callable[[int], object],
) -> None:
    # A repr may be a str subclass whose ``<`` is no total order, or raises.
    value = {kind(n) for n in range(64)}

    assert _outcome(lambda: serialize.to_plain(value)) == _outcome(
        lambda: serialize._python_to_plain(value),
    )


@dataclass(slots=True)
class _SlottedPair:
    a: int
    b: int


@needs_rust
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("planted", ["import path", "stale import path", "slots"])
def test_planted_cache_entries_encode_as_python_does(
    monkeypatch: pytest.MonkeyPatch,
    planted: str,
    sort: bool,
) -> None:
    from treekle import reflect  # noqa: PLC0415 -- only this test reads its caches

    value = _SlottedPair(1, 2)
    serialize._python_to_plain(value, sort=sort)
    # The native walk keeps what it read from these caches across calls.
    assert serialize.to_plain(value, sort=sort) == serialize._python_to_plain(
        value,
        sort=sort,
    )
    if planted == "import path":
        monkeypatch.setitem(
            serialize._IMPORT_PATHS,
            id(_SlottedPair),
            (_SlottedPair, "a.b"),
        )
    elif planted == "stale import path":
        monkeypatch.setitem(
            serialize._IMPORT_PATHS,
            id(_SlottedPair),
            (object(), "a.b"),
        )
    else:
        monkeypatch.setitem(reflect._SLOT_NAMES, _SlottedPair, ("b",))
        monkeypatch.setitem(reflect._SORTED_SLOT_NAMES, _SlottedPair, ("b",))

    expected = serialize._python_to_plain(value, sort=sort)

    assert serialize.to_plain(value, sort=sort) == expected
    assert isinstance(expected, dict)
    assert (expected["py/object"] == "a.b") == (planted == "import path")
    assert ("a" in expected) == (planted != "slots")


@needs_rust
@pytest.mark.parametrize("source", ["cache", "mapping"])
def test_encoder_changes_during_walk_match_python(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    original = serialize._ENCODERS

    def replaced(_obj: object, _state: serialize._Encoding) -> str:
        return "replaced"

    def decoded(value: str) -> str:
        return value

    def run(function: Callable[..., object]) -> object:
        monkeypatch.setattr(serialize, "_ENCODERS", original)
        monkeypatch.setattr(
            serialize,
            "_ENCODER_CACHE",
            weakref.WeakKeyDictionary[type, serialize._Encoder](),
        )

        def replace(_value: Tensor) -> str:
            if source == "cache":
                serialize._ENCODER_CACHE[Point] = replaced
            else:
                monkeypatch.setattr(
                    serialize,
                    "_ENCODERS",
                    {**original, Point: replaced},
                )
            return "triggered"

        return function(
            [Point(1), Tensor([]), Point(2)],
            hooks={Tensor: (replace, decoded)},
        )

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("list[object]", expected)[2] == "replaced"


@needs_rust
@pytest.mark.parametrize("proxy", [False, True])
def test_hook_added_to_empty_mapping_during_walk_matches_python(
    proxy: bool,
) -> None:
    hooks: dict[type, tuple[object, object]] = {}

    def encode_tensor(value: Tensor) -> list[float]:
        return value.data

    class InstallHook:
        @override
        def __reduce_ex__(self, protocol: object) -> tuple[object, tuple[()]]:
            del protocol
            hooks[Tensor] = (encode_tensor, Tensor)
            return (list, ())

    def run(function: Callable[..., object]) -> object:
        hooks.clear()
        supplied = MappingProxyType(hooks) if proxy else hooks
        return function([InstallHook(), Tensor([7.0])], hooks=supplied)

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("list[dict[str, object]]", expected)[1]["py/hook"]


@needs_rust
def test_same_size_dict_key_mutation_matches_python() -> None:
    def run(function: Callable[..., object]) -> object:
        value: dict[str, object] = {}

        class ChangeKeys:
            @override
            def __reduce_ex__(self, protocol: object) -> tuple[object, tuple[()]]:
                del protocol
                del value["a"]
                value["c"] = 3
                return (list, ())

        value.update(a=ChangeKeys(), b=2)
        return _outcome(lambda: function(value))

    expected = run(serialize._python_to_plain)
    assert cast("tuple[object, ...]", expected)[0] == "ok"
    assert run(serialize.to_plain) == expected


@needs_rust
@pytest.mark.parametrize("change", ["slot", "dict_value"])
def test_attribute_changes_during_walk_match_python(
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    original_slot = cast("object", ChangingSlot.__dict__["b"])

    def replaced_b(_value: ChangingSlot) -> int:
        return 99

    def decoded(value: str) -> str:
        return value

    def run(function: Callable[..., object]) -> object:
        if change == "slot":
            monkeypatch.setattr(ChangingSlot, "b", original_slot)
            target: ChangingSlot | ChangingDict = ChangingSlot(Tensor([]), 2)
        else:
            target = ChangingDict(Tensor([]), 2)

        def replace(_value: Tensor) -> str:
            if change == "slot":
                monkeypatch.setattr(ChangingSlot, "b", property(replaced_b))
            else:
                target.b = 99
            return "triggered"

        return function(target, hooks={Tensor: (replace, decoded)})

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("dict[str, object]", expected)["b"] == 2


@needs_rust
def test_slot_changes_between_objects_match_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_slot = cast("object", ChangingSlot.__dict__["b"])

    def replaced_b(_value: ChangingSlot) -> int:
        return 99

    def replace(_value: Tensor) -> str:
        monkeypatch.setattr(ChangingSlot, "b", property(replaced_b))
        return "triggered"

    def decoded(value: str) -> str:
        return value

    def run(function: Callable[..., object]) -> object:
        monkeypatch.setattr(ChangingSlot, "b", original_slot)
        return function(
            [ChangingSlot(0, 2), Tensor([]), ChangingSlot(0, 3)],
            hooks={Tensor: (replace, decoded)},
        )

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("dict[str, object]", cast("list[object]", expected)[2])["b"] == 99


@needs_rust
def test_foreign_slot_descriptor_raises_like_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = ForeignSlotOwner(11, 22)
    monkeypatch.setattr(
        ForeignSlotOwner,
        "x",
        cast("MemberDescriptorType", vars(ForeignSlotSource)["b"]),
    )

    with pytest.raises(TypeError) as expected:
        serialize._python_to_plain(value)
    with pytest.raises(TypeError) as actual:
        serialize.to_plain(value)
    assert str(actual.value) == str(expected.value)


class _HidesHash(type):
    """Reports ``__hash__`` as ``None`` while the class's hash slot still hashes."""

    @override
    def __getattribute__(cls, name: str) -> object:
        if name == "__hash__":
            return None
        return cast("object", super().__getattribute__(name))


class _ReportsHash(type):
    """Reports a ``__hash__`` for a class whose hash slot refuses hashing."""

    @override
    def __getattribute__(cls, name: str) -> object:
        if name == "__hash__":
            return object.__hash__
        return cast("object", super().__getattribute__(name))


class HashHidden(metaclass=_HidesHash):
    def __init__(self, n: int) -> None:
        self.n = n

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (HashHidden, (self.n,))


class HashReported(metaclass=_ReportsHash):
    __hash__ = None  # pyright: ignore[reportAssignmentType]

    def __init__(self, n: int) -> None:
        self.n = n

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return (HashReported, (self.n,))


@pytest.mark.parametrize("kind", [HashHidden, HashReported])
def test_hashability_is_judged_by_the_class_attribute_as_python_does(
    kind: type[HashHidden | HashReported],
) -> None:
    # Python's ``is_mutable`` reads ``type(value).__hash__``, which these
    # metaclasses make disagree with the hash slot; it decides whether a
    # repeat is a ``py/ref`` or written again.
    value = kind(1)
    tree = [value, value]
    assert serialize.to_plain(tree) == serialize._python_to_plain(tree)


@needs_rust
def test_nesting_deeper_than_the_old_python_limit_matches_python() -> None:
    value = _nested(900)
    expected = serialize._python_to_plain(value)
    actual = serialize.to_plain(value)
    for tree in (expected, actual):
        cursor = tree
        for _ in range(900):
            assert isinstance(cursor, list)
            children = cast("list[object]", cursor)
            assert len(children) == 1
            cursor = children[0]
        assert cursor == []
    assert deserialize.from_plain(actual, object) is actual


def _deep(value: object, depth: int) -> object:
    """Return ``value`` nested ``depth`` lists down."""
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
def test_many_deep_subtrees_match_python(mutable: bool) -> None:
    shared = [0]
    value = _deep([[shared, [index]] for index in range(500)], 47)

    expected = serialize._python_to_plain(value, mutable=mutable)
    assert _shape(serialize.to_plain(value, mutable=mutable)) == _shape(expected)


class _Recipe:
    """Reduces to a recipe whose state holds whatever it was given."""

    def __init__(self, state: object) -> None:
        self.state = state

    @override
    def __reduce_ex__(self, protocol: object) -> tuple[object, ...]:
        del protocol
        return (_Recipe, (None,), {"state": self.state})


def _every_native_kind(depth: int) -> object:
    """Return a tree that nests each kind the native walk encodes ``depth`` deep."""
    shared = Point(1, 2.0)
    wrappers: list[Callable[[object, int], object]] = [
        lambda tree, level: {"plain": tree, "py/tagged": level, 3: shared},
        lambda tree, level: (tree, Color.RED, Level.LOW, Frozen(level)),
        lambda tree, level: [tree, {frozenset({level, "x"}), Frozen(level), "y"}],
        lambda tree, _: _Recipe(tree),
        lambda tree, level: Node(f"n{level}", [Node("leaf")], cast("Node", tree)),
        lambda tree, _: MappingProxyType({"m": tree, "f": math.inf, "s": shared}),
    ]
    tree: object = shared
    for level in range(depth):
        tree = wrappers[level % len(wrappers)](tree, level)
    return [shared, tree]


@needs_rust
@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
def test_deep_trees_of_every_kind_encode_natively_as_python_does(
    monkeypatch: pytest.MonkeyPatch,
    mutable: bool,
) -> None:
    # Python's walk is what the native one hands a subtree to when it cannot
    # encode it itself; none of this needs that, however deep.
    value = _every_native_kind(3_000)
    expected = serialize._python_to_plain(value, mutable=mutable)

    def refused(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the native walk ran Python's")

    monkeypatch.setattr(serialize, "_encode", refused)
    monkeypatch.setattr(serialize, "_Encoding", refused)
    _assert_same_shape(serialize.to_plain(value, mutable=mutable), expected)


@needs_rust
def test_each_python_encoder_shares_one_reference_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fresh map per Python encoder would copy every registration so far each
    # time, which is quadratic in how many values one encodes.
    received: list[object] = []

    class Holder:
        def __init__(self, held: object) -> None:
            self.held = held

    def encode_holder(value: object, state: serialize._Encoding) -> object:
        received.append(state.seen)
        serialize._register(value, state)
        node: dict[str, object] = {"held": cast("Holder", value).held}
        serialize._push_values(node, state)
        return node

    monkeypatch.setitem(serialize._ENCODER_CACHE, Holder, encode_holder)
    shared = [0]
    value = [[Holder(index), shared, Holder(shared)] for index in range(50)]
    expected = serialize._python_to_plain(value)
    received.clear()

    assert serialize.to_plain(value) == expected
    assert len(received) == 100
    assert all(seen is received[0] for seen in received)


@pytest.mark.parametrize("mutable", [True, False], ids=["mutable", "frozen"])
def test_rolled_back_registrations_deep_in_the_tree_match_python(
    mutable: bool,
) -> None:
    # A set member and an escaped key are encoded once for their text, and
    # what that registers is forgotten.
    inner = Frozen(1)
    deep: object = inner
    for _ in range(60):
        deep = (deep,)
    value = [{deep}, {deep: 1}, inner, [inner]]

    expected = serialize._python_to_plain(value, mutable=mutable)
    assert _shape(serialize.to_plain(value, mutable=mutable)) == _shape(expected)


@dataclass(eq=False)
class KeyBox:
    """A key hashed by identity, holding whatever it is given."""

    held: object


def _key_cases() -> dict[str, object]:
    point = Frozen(1)
    looped = KeyBox(held=None)
    looped.held = [looped]
    return {
        "repeat": {(point, point): 1},
        "earlier": {"first": [point], (point, point): 1},
        "nested": {KeyBox(held={(point, point): 1, "p": point}): point},
        "cycle": {looped: 1},
    }


@pytest.mark.parametrize("case", list(_key_cases()))
def test_repeats_inside_an_escaped_key_match_python(case: str) -> None:
    value = _key_cases()[case]

    expected = _outcome(lambda: serialize._python_to_plain(value))
    assert _outcome(lambda: serialize.to_plain(value)) == expected


@needs_rust
def test_a_python_encoder_inside_a_key_sees_it_is_inside_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Python walk the Rust one calls is placed under the same sentinel, so
    # a repeat on either side of the boundary is written in full.
    def encode_box(value: object, state: serialize._Encoding) -> object:
        serialize._register(value, state)
        node: dict[str, object] = {"held": cast("KeyBox", value).held}
        serialize._push_values(node, state)
        return node

    monkeypatch.setitem(serialize._ENCODER_CACHE, KeyBox, encode_box)
    point = Frozen(1)
    box = KeyBox(held=point)
    value = {(point, box, box): 1}

    expected = serialize._python_to_plain(value)
    assert serialize.to_plain(value) == expected
    assert isinstance(expected, dict)
    assert "py/ref" not in next(iter(expected))


@needs_rust
@pytest.mark.parametrize("change", ["class", "module"])
def test_enum_reduce_changed_between_calls_matches_python(
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    # What a walk learns about a class is kept for the next call; changing
    # the class or the module function it stands in for must be seen.
    value = [Color.RED, Color.BLUE]
    before = serialize._python_to_plain(value)
    assert serialize.to_plain(value) == before

    def custom(self: Color, protocol: object) -> tuple[object, ...]:
        del protocol
        return (Color, (self.value,), {"note": "custom"})

    if change == "class":
        monkeypatch.setattr(Color, "__reduce_ex__", custom, raising=False)
    else:
        original = serialize._canonical_reduce

        def canonical(value: object) -> object:
            if value is Color.RED:
                return (Color, ("red",), {"note": "module"})
            return original(value)

        monkeypatch.setattr(serialize, "_canonical_reduce", canonical)
    expected = serialize._python_to_plain(value)
    assert expected != before
    assert serialize.to_plain(value) == expected


@needs_rust
def test_list_changes_during_walk_match_python() -> None:
    def run(function: Callable[..., object]) -> object:
        value: list[object] = []

        def grow(_tensor: Tensor) -> str:
            value.append("added")
            return "triggered"

        value.extend([Tensor([]), "tail"])
        return function(value, hooks={Tensor: (grow, Tensor)})

    expected = run(serialize._python_to_plain)
    assert run(serialize.to_plain) == expected
    assert cast("list[object]", expected)[1:] == ["tail"]


def _run_python(script: str, **env: str) -> str:
    """Run ``script`` in a fresh interpreter; return what it printed."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        check=False,
        env={**os.environ, **env},
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


_WHICH = """
import treekle.serialize as serialize

print(serialize.to_plain is serialize._python_to_plain)
"""


def test_the_environment_variable_selects_python() -> None:
    assert _run_python(_WHICH, TREEKLE_PURE_PYTHON="1") == "True"


def test_a_missing_core_falls_back_to_python() -> None:
    hidden = "import sys\nsys.modules['treekle._rust'] = None\n" + _WHICH
    assert _run_python(hidden, TREEKLE_PURE_PYTHON="") == "True"


def test_a_core_that_cannot_set_up_falls_back_to_python_with_a_warning() -> None:
    # A core built from older sources imports, then refuses this module.
    mismatched = """
        import sys
        import types
        import warnings

        def refuse(module):
            raise AttributeError("treekle.serialize has no OLD_HELPER")

        core = types.ModuleType("treekle._rust")
        core.setup_encode = core.to_plain = refuse
        sys.modules["treekle._rust"] = core
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import treekle.serialize as serialize

        print(
            serialize.to_plain is serialize._python_to_plain,
            sum(
                warning.category is RuntimeWarning
                and "OLD_HELPER" in str(warning.message)
                for warning in caught
            ),
        )
        """
    assert _run_python(mismatched, TREEKLE_PURE_PYTHON="") == "True 1"


@needs_rust
@pytest.mark.parametrize("setting", ["", "0"])
def test_a_built_core_is_installed(setting: str) -> None:
    assert _run_python(_WHICH, TREEKLE_PURE_PYTHON=setting) == "False"


@needs_rust
@pytest.mark.parametrize(
    ("rust", "python"),
    [
        (serialize.to_plain, serialize._python_to_plain),
    ],
    ids=["to_plain"],
)
def test_the_rust_functions_document_themselves_as_python_does(
    rust: Callable[..., object],
    python: Callable[..., object],
) -> None:
    assert inspect.getdoc(rust) == inspect.getdoc(python)
    ours = inspect.signature(rust).parameters
    theirs = inspect.signature(python).parameters
    assert [(each.name, each.kind) for each in ours.values()] == [
        (each.name, each.kind) for each in theirs.values()
    ]
    # A text signature shows a default only when it is a literal.
    literal = {
        name: default
        for name, each in ours.items()
        if (default := cast("object", each.default)) is not Ellipsis
    }
    assert literal == {name: cast("object", theirs[name].default) for name in literal}
