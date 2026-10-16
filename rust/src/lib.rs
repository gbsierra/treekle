//! Treekle's Rust core: ``to_plain`` with the semantics of ``treekle.serialize``,
//! which installs it.

// GilCell and the per-interpreter Rc caches require a real GIL. Refuse
// free-threaded builds, including configurations that keep the GIL enabled
// today but could later disable it with PYTHON_GIL=0 or -Xgil=0. Those
// interpreters can install the universal Python wheel instead.
#[cfg(Py_GIL_DISABLED)]
compile_error!(
    "treekle's Rust core requires a GIL; use the pure-Python wheel on free-threaded Python"
);

mod common;
mod encode;
mod watch;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::common::{B, Maybe};

/// Read the encoder's helpers and sentinels from ``treekle.serialize``'s globals.
#[pyfunction]
fn setup_encode(module: &Bound<'_, PyDict>) -> PyResult<()> {
    encode::setup(module)
}

// The public function carries ``treekle.serialize``'s docstring verbatim,
// since it replaces that function; ``rust_parity_test`` checks that the two
// stay alike. A text signature can only show literal defaults, so the sentinel
// ones read ``...``.

/// Convert a Python value to plain data, tagging what ``dialect`` lacks.
///
/// Args:
///   obj: The value to convert; any object graph, shared references and cycles
///     included, except a cycle inside a dict key. An object repeated inside a
///     dict key reads back as an equal copy, not the same object.
///   dialect: The target format, which decides which values need a tag.
///   mutable: Build dicts and lists when true; ``MappingProxyType`` and tuples
///     when false.
///   hooks: Codecs for leaf types with no built-in conversion, such as tensors.
///   sort: Order each object's attributes by name; when false, keep slots in
///     MRO order then ``__dict__`` keys in insertion order. A stored form that
///     must stay byte-stable sets this false.
///
/// Returns:
///   tree: Fresh plain data that ``from_plain`` turns back into ``obj``.
///
/// Raises:
///   ValueError: ``dialect`` names no known format.
///   TypeError: A leaf has no conversion and no hook, or a dict key holds a
///     cycle.
#[pyfunction]
#[pyo3(
    signature = (obj, *, dialect = Maybe(None), mutable = Maybe(None), hooks = Maybe(None), sort = Maybe(None)),
    text_signature = "(obj, *, dialect='json', mutable=True, hooks=..., sort=True)"
)]
fn to_plain<'py>(
    py: Python<'py>,
    obj: &B<'py>,
    dialect: Maybe<'py>,
    mutable: Maybe<'py>,
    hooks: Maybe<'py>,
    sort: Maybe<'py>,
) -> PyResult<B<'py>> {
    encode::to_plain(py, obj, dialect.0, mutable.0, hooks.0, sort.0)
}

#[pymodule(gil_used = true)]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(setup_encode, module)?)?;
    module.add_function(wrap_pyfunction!(to_plain, module)?)?;
    Ok(())
}
