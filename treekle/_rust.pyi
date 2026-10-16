from typing import Literal, overload

from treekle.types import Dialect, Hooks, MutablePlainTree, PlainTree

def setup_encode(module: dict[str, object]) -> None: ...
@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: Literal[True] = ...,
    hooks: Hooks = ...,
    sort: bool = ...,
) -> MutablePlainTree: ...
@overload
def to_plain(
    obj: object,
    *,
    dialect: Dialect = ...,
    mutable: bool,
    hooks: Hooks = ...,
    sort: bool = ...,
) -> PlainTree: ...
