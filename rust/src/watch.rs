//! Dict watchers that tell a native cache when the Python state it copied
//! changes.
//!
//! A walk keeps facts it read from Python dicts (a module's globals, a cache
//! such as ``_ENCODER_CACHE``) across nodes and calls. Rereading them at every
//! node would cost a lookup or a Python call each time; instead each watched
//! dict advances a generation whenever it changes, and a fact is trusted only
//! while the generation it was read at is current. CPython calls the watcher
//! before every change to a watched dict (3.12+), including deletion by a
//! weak-reference callback and deallocation, so no change goes unseen.

use std::ffi::c_int;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type Callback = unsafe extern "C" fn(
    event: c_int,
    dict: *mut ffi::PyObject,
    key: *mut ffi::PyObject,
    new_value: *mut ffi::PyObject,
) -> c_int;

unsafe extern "C" {
    fn PyDict_AddWatcher(callback: Callback) -> c_int;
    fn PyDict_Watch(watcher_id: c_int, dict: *mut ffi::PyObject) -> c_int;
}

const UNTRIED: i32 = -2;
const UNAVAILABLE: i32 = -1;

/// One watcher and the generation it advances.
pub struct Watch {
    id: AtomicI32,
    generation: AtomicU64,
    callback: Callback,
}

impl Watch {
    const fn new(callback: Callback) -> Self {
        Self {
            id: AtomicI32::new(UNTRIED),
            generation: AtomicU64::new(0),
            callback,
        }
    }

    /// The current generation, or ``None`` when no watcher could be had, so
    /// nothing read from a watched dict may be trusted past its first use.
    #[inline]
    pub fn generation(&self) -> Option<u64> {
        if self.id.load(Ordering::Relaxed) < 0 {
            return None;
        }
        Some(self.generation.load(Ordering::Relaxed))
    }

    fn id(&self, _py: Python<'_>) -> Option<c_int> {
        let mut id = self.id.load(Ordering::Relaxed);
        if id == UNTRIED {
            // CPython allows a few watchers per interpreter; when none is
            // left, callers fall back to reading the dicts at each use.
            id = unsafe { PyDict_AddWatcher(self.callback) };
            if id < 0 {
                unsafe { ffi::PyErr_Clear() };
                id = UNAVAILABLE;
            }
            self.id.store(id, Ordering::Relaxed);
        }
        (id >= 0).then_some(id)
    }

    /// Watch ``dict``; ``false`` when it cannot be watched. Watching a dict
    /// twice is harmless.
    pub fn watch(&self, py: Python<'_>, dict: &Bound<'_, PyDict>) -> bool {
        let Some(id) = self.id(py) else {
            return false;
        };
        if unsafe { PyDict_Watch(id, dict.as_ptr()) } < 0 {
            unsafe { ffi::PyErr_Clear() };
            return false;
        }
        true
    }

    #[inline]
    fn changed(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }
}

unsafe extern "C" fn encode_changed(
    _event: c_int,
    _dict: *mut ffi::PyObject,
    _key: *mut ffi::PyObject,
    _new_value: *mut ffi::PyObject,
) -> c_int {
    ENCODE.changed();
    0
}

/// The dicts ``to_plain`` copies facts from: ``treekle.serialize`` and
/// ``treekle.reflect`` globals and the caches they hold.
pub static ENCODE: Watch = Watch::new(encode_changed);
