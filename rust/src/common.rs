//! Small helpers shared by the decoder and the encoder.

use std::cell::UnsafeCell;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList, PyString, PyTuple};

pub type B<'py> = Bound<'py, PyAny>;

/// A value only touched while the GIL is held. It is only ever swapped or
/// cloned, never borrowed, so Python code that runs mid-call and re-enters
/// (a hook calling ``to_plain``) can never alias a live reference into it.
pub struct GilCell<T>(UnsafeCell<T>);

// SAFETY: the module declares it needs the GIL, and every access takes a
// `Python` token, so no two threads reach the cell at once.
unsafe impl<T> Sync for GilCell<T> {}

impl<T> GilCell<T> {
    pub const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// Put ``value`` in the cell and return what it held. The old value is
    /// dropped by the caller, after the swap, so a ``__del__`` it triggers
    /// finds the cell consistent.
    pub fn replace(&self, _py: Python<'_>, value: T) -> T {
        // SAFETY: the GIL is held and no reference into the cell outlives a
        // call, so nothing else reads or writes it during the swap.
        unsafe { std::mem::replace(&mut *self.0.get(), value) }
    }
}

impl<T: Default> GilCell<T> {
    /// Take the cell's value, leaving the default.
    pub fn take(&self, py: Python<'_>) -> T {
        self.replace(py, T::default())
    }
}

impl<T: Clone> GilCell<T> {
    /// A copy of the cell's value.
    pub fn cloned(&self, _py: Python<'_>) -> T {
        // SAFETY: as in `replace`; `T::clone` here is an `Rc` or `Option`
        // clone, which runs no Python code.
        unsafe { (*self.0.get()).clone() }
    }
}

/// A ``to_plain`` or ``from_plain`` option, read as Python reads it: by truth
/// value at each use.
pub enum Flag<'py> {
    Known(bool),
    Object(B<'py>),
}

impl<'py> Flag<'py> {
    pub fn new(value: Option<B<'py>>, default: bool) -> Self {
        match value {
            None => Self::Known(default),
            Some(value) if tp(&value) == t_bool() => {
                Self::Known(value.as_ptr() == unsafe { ffi::Py_True() })
            }
            Some(value) => Self::Object(value),
        }
    }

    #[inline]
    pub fn get(&self) -> PyResult<bool> {
        match self {
            Self::Known(value) => Ok(*value),
            Self::Object(value) => value.is_truthy(),
        }
    }

    /// The option as Python holds it.
    pub fn object(&self, py: Python<'py>) -> B<'py> {
        match self {
            Self::Known(value) => pyo3::types::PyBool::new(py, *value).to_owned().into_any(),
            Self::Object(value) => value.clone(),
        }
    }
}

// Before 3.14, `Py_EnterRecursiveCall` only counts calls against a fixed limit
// (`Py_C_RECURSION_LIMIT`, 10000 on x86-64 Linux), and a walk's frames can
// exhaust the thread's stack long before that: an 8 MB stack near 7000 levels,
// a 512 KiB thread stack by 400. So those versions also check the real stack,
// as 3.14 does.
#[cfg(not(Py_3_14))]
mod stack {
    use std::cell::Cell;

    /// Stack left unused below the deepest level, for the Python code a walk
    /// calls there (hooks, helpers, reprs), which 3.12 and 3.13 run without
    /// stack checks of their own. A quarter of a small stack at most.
    const MARGIN: usize = 128 * 1024;

    #[derive(Clone, Copy)]
    struct Bounds {
        base: usize,
        limit: usize,
        top: usize,
    }

    thread_local! {
        /// `None` until read; `Some(None)` where the platform does not say.
        static BOUNDS: Cell<Option<Option<Bounds>>> = const { Cell::new(None) };
        static DEPTH: Cell<i32> = const { Cell::new(0) };
    }

    /// The thread's usable stack as `(lowest, highest)` address.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn read() -> Option<(usize, usize)> {
        unsafe {
            let mut attr: libc::pthread_attr_t = std::mem::zeroed();
            if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) != 0 {
                return None;
            }
            let mut addr: *mut libc::c_void = std::ptr::null_mut();
            let mut size: libc::size_t = 0;
            let mut guard: libc::size_t = 0;
            let ok = libc::pthread_attr_getstack(&attr, &mut addr, &mut size) == 0
                && libc::pthread_attr_getguardsize(&attr, &mut guard) == 0;
            libc::pthread_attr_destroy(&mut attr);
            (ok && size > guard).then(|| (addr as usize + guard, addr as usize + size))
        }
    }

    #[cfg(target_vendor = "apple")]
    fn read() -> Option<(usize, usize)> {
        unsafe {
            let thread = libc::pthread_self();
            let top = libc::pthread_get_stackaddr_np(thread) as usize;
            let size = libc::pthread_get_stacksize_np(thread);
            (size > 0 && top > size).then(|| (top - size, top))
        }
    }

    #[cfg(windows)]
    fn read() -> Option<(usize, usize)> {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
        }
        let (mut low, mut high) = (0, 0);
        unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
        (high > low).then_some((low, high))
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    )))]
    fn read() -> Option<(usize, usize)> {
        None
    }

    fn bounds() -> Option<Bounds> {
        if let Some(known) = BOUNDS.get() {
            return known;
        }
        let known = read().map(|(base, top)| Bounds {
            base,
            limit: base + MARGIN.min((top - base) / 4),
            top,
        });
        BOUNDS.set(Some(known));
        known
    }

    /// Whether one more level, the `depth`-th, would leave too little stack.
    /// Where the stack is unknown, or this code runs outside it (a user-space
    /// thread), `depth` is held to `sys.getrecursionlimit()` instead.
    #[inline]
    pub fn too_deep(depth: i32) -> bool {
        let marker = 0u8;
        let here = std::hint::black_box(&marker) as *const u8 as usize;
        match bounds() {
            Some(known) if here >= known.base && here < known.top => here < known.limit,
            _ => depth > unsafe { pyo3::ffi::Py_GetRecursionLimit() },
        }
    }

    pub fn depth() -> &'static std::thread::LocalKey<Cell<i32>> {
        &DEPTH
    }
}

/// One level of recursion, counted as CPython counts C recursion; before 3.14
/// the real stack is checked too, so a deep tree raises ``RecursionError``
/// instead of overflowing it.
pub struct Recursion;

impl Recursion {
    #[inline]
    pub fn enter(py: Python<'_>, context: &'static std::ffi::CStr) -> PyResult<Self> {
        #[cfg(not(Py_3_14))]
        let depth = stack::depth().get() + 1;
        #[cfg(not(Py_3_14))]
        if stack::too_deep(depth) {
            return Err(pyo3::exceptions::PyRecursionError::new_err(format!(
                "maximum recursion depth exceeded{}",
                context.to_string_lossy()
            )));
        }
        if unsafe { ffi::Py_EnterRecursiveCall(context.as_ptr()) } != 0 {
            return Err(PyErr::fetch(py));
        }
        #[cfg(not(Py_3_14))]
        stack::depth().set(depth);
        Ok(Self)
    }
}

impl Drop for Recursion {
    #[inline]
    fn drop(&mut self) {
        #[cfg(not(Py_3_14))]
        stack::depth().set(stack::depth().get() - 1);
        unsafe { ffi::Py_LeaveRecursiveCall() };
    }
}

unsafe extern "C" {
    fn PyUnstable_Type_AssignVersionTag(kind: *mut ffi::PyTypeObject) -> std::ffi::c_int;
}

/// What a fact learned from a class assumed about it. CPython replaces a
/// class's version tag whenever the class or a base changes, and never reuses
/// one; but 3.13 and later rename a class (``__qualname__``) without a new tag,
/// so the name object is held and compared too. Immutable types never change.
pub struct Stamp {
    tag: u32,
    qualname: Option<Py<PyAny>>,
}

impl Stamp {
    /// The class's current state, or ``None`` if it cannot change.
    pub fn of(class: &B<'_>) -> Option<Self> {
        let kind = class.as_ptr().cast::<ffi::PyTypeObject>();
        unsafe {
            if ffi::PyType_HasFeature(kind, ffi::Py_TPFLAGS_IMMUTABLETYPE) != 0 {
                return None;
            }
            if (*kind).tp_version_tag == 0 {
                // A class out of tags stays at 0, which never matches.
                PyUnstable_Type_AssignVersionTag(kind);
            }
            let qualname = heap_qualname(kind);
            Some(Self {
                tag: (*kind).tp_version_tag,
                qualname: (!qualname.is_null())
                    .then(|| Py::from_borrowed_ptr(class.py(), qualname)),
            })
        }
    }

    /// The class's current state, immutable classes included. A fact kept
    /// across calls by class address needs this: version tags are never
    /// reused, so a new class that takes a dead one's address never matches.
    pub fn always(class: &B<'_>) -> Self {
        let kind = class.as_ptr().cast::<ffi::PyTypeObject>();
        unsafe {
            if (*kind).tp_version_tag == 0 {
                PyUnstable_Type_AssignVersionTag(kind);
            }
            let qualname = heap_qualname(kind);
            Self {
                tag: (*kind).tp_version_tag,
                qualname: (!qualname.is_null())
                    .then(|| Py::from_borrowed_ptr(class.py(), qualname)),
            }
        }
    }

    /// Whether ``class`` is unchanged since this stamp was taken of it.
    #[inline]
    pub fn holds(&self, class: &B<'_>) -> bool {
        let kind = class.as_ptr().cast::<ffi::PyTypeObject>();
        let qualname = self
            .qualname
            .as_ref()
            .map_or(std::ptr::null_mut(), Py::as_ptr);
        self.tag != 0
            && unsafe { (*kind).tp_version_tag == self.tag && heap_qualname(kind) == qualname }
    }
}

#[inline]
unsafe fn heap_qualname(kind: *mut ffi::PyTypeObject) -> *mut ffi::PyObject {
    unsafe {
        if ffi::PyType_HasFeature(kind, ffi::Py_TPFLAGS_HEAPTYPE) == 0 {
            return std::ptr::null_mut();
        }
        (*kind.cast::<ffi::PyHeapTypeObject>()).ht_qualname
    }
}

#[inline(always)]
pub fn tp(obj: &B<'_>) -> *mut ffi::PyTypeObject {
    unsafe { ffi::Py_TYPE(obj.as_ptr()) }
}

macro_rules! type_ptr {
    ($name:ident, $sym:ident) => {
        #[inline(always)]
        pub fn $name() -> *mut ffi::PyTypeObject {
            &raw mut ffi::$sym
        }
    };
}

type_ptr!(t_dict, PyDict_Type);
type_ptr!(t_list, PyList_Type);
type_ptr!(t_tuple, PyTuple_Type);
type_ptr!(t_str, PyUnicode_Type);
type_ptr!(t_int, PyLong_Type);
type_ptr!(t_float, PyFloat_Type);
type_ptr!(t_bool, PyBool_Type);
type_ptr!(t_proxy, PyDictProxy_Type);
type_ptr!(t_object, PyBaseObject_Type);

#[inline(always)]
pub fn t_none() -> *mut ffi::PyTypeObject {
    unsafe { ffi::Py_TYPE(ffi::Py_None()) }
}

#[inline(always)]
pub fn is_str(obj: &B<'_>) -> bool {
    unsafe { ffi::PyUnicode_Check(obj.as_ptr()) != 0 }
}

#[inline(always)]
pub fn is_int(obj: &B<'_>) -> bool {
    unsafe { ffi::PyLong_Check(obj.as_ptr()) != 0 }
}

#[inline(always)]
pub fn is_float(obj: &B<'_>) -> bool {
    unsafe { ffi::PyFloat_Check(obj.as_ptr()) != 0 }
}

#[inline(always)]
pub fn is_tuple(obj: &B<'_>) -> bool {
    unsafe { ffi::PyTuple_Check(obj.as_ptr()) != 0 }
}

#[inline(always)]
pub fn is_list(obj: &B<'_>) -> bool {
    unsafe { ffi::PyList_Check(obj.as_ptr()) != 0 }
}

#[inline(always)]
pub fn is_dict(obj: &B<'_>) -> bool {
    unsafe { ffi::PyDict_Check(obj.as_ptr()) != 0 }
}

#[inline(always)]
pub fn is_class(obj: &B<'_>) -> bool {
    unsafe { ffi::PyType_Check(obj.as_ptr()) != 0 }
}

#[inline(always)]
pub fn is_callable(obj: &B<'_>) -> bool {
    unsafe { ffi::PyCallable_Check(obj.as_ptr()) != 0 }
}

/// ``type(obj).__hash__ is None``: whether instances of ``obj``'s class opt
/// out of hashing.
#[inline]
pub fn is_unhashable(obj: &B<'_>) -> PyResult<bool> {
    let kind = tp(obj);
    // Under a plain ``type`` metaclass the attribute is the class's own
    // ``__hash__``, which CPython keeps in step with ``tp_hash``. Any other
    // metaclass may answer the lookup itself (a ``__getattribute__``, a
    // ``__hash__`` property) while the slot still hashes.
    if unsafe { ffi::Py_TYPE(kind.cast()) } == &raw mut ffi::PyType_Type {
        // Slot functions are compared by address; libpython exports have one canonical address.
        return Ok(unsafe {
            (*kind).tp_hash.is_some_and(|f| {
                f as usize == ffi::PyObject_HashNotImplemented as *const () as usize
            })
        });
    }
    let py = obj.py();
    Ok(obj
        .get_type()
        .getattr(pyo3::intern!(py, "__hash__"))?
        .is_none())
}

/// The UTF-8 bytes of a str, or ``None`` when it holds lone surrogates.
#[inline]
pub fn utf8<'a>(obj: &'a B<'_>) -> Option<&'a [u8]> {
    unsafe {
        let mut size: ffi::Py_ssize_t = 0;
        let data = ffi::PyUnicode_AsUTF8AndSize(obj.as_ptr(), &mut size);
        if data.is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        Some(std::slice::from_raw_parts(data.cast::<u8>(), size as usize))
    }
}

/// ``isinstance(obj, str) and obj.startswith(prefixes)``. A subclass may
/// override ``startswith``, and a str with lone surrogates has no UTF-8 form,
/// so both ask Python.
#[inline]
pub fn str_starts_any(obj: &B<'_>, prefixes: &[&str]) -> PyResult<bool> {
    if !is_str(obj) {
        return Ok(false);
    }
    if tp(obj) == t_str()
        && let Some(text) = utf8(obj)
    {
        return Ok(prefixes
            .iter()
            .any(|prefix| text.starts_with(prefix.as_bytes())));
    }
    let py = obj.py();
    let argument = match prefixes {
        [prefix] => pystr(py, prefix),
        _ => PyTuple::new(py, prefixes)?.into_any(),
    };
    obj.call_method1(pyo3::intern!(py, "startswith"), (argument,))?
        .is_truthy()
}

/// ``isinstance(obj, str) and obj.startswith(prefix)``.
#[inline]
pub fn str_starts(obj: &B<'_>, prefix: &str) -> PyResult<bool> {
    str_starts_any(obj, &[prefix])
}

pub fn repr(obj: &B<'_>) -> PyResult<String> {
    Ok(obj.repr()?.to_string_lossy().into_owned())
}

pub fn text(obj: &B<'_>) -> PyResult<String> {
    Ok(obj.str()?.to_string_lossy().into_owned())
}

pub fn err_text(py: Python<'_>, err: &PyErr) -> PyResult<String> {
    text(err.value(py).as_any())
}

/// ``obj == other`` with Python's identity shortcut.
pub fn py_eq(obj: &B<'_>, other: &B<'_>) -> PyResult<bool> {
    if obj.is(other) {
        return Ok(true);
    }
    let result = unsafe { ffi::PyObject_RichCompareBool(obj.as_ptr(), other.as_ptr(), ffi::Py_EQ) };
    if result < 0 {
        return Err(PyErr::fetch(obj.py()));
    }
    Ok(result == 1)
}

/// ``obj != other`` as a condition.
pub fn py_ne(obj: &B<'_>, other: &B<'_>) -> PyResult<bool> {
    let result = unsafe { ffi::PyObject_RichCompareBool(obj.as_ptr(), other.as_ptr(), ffi::Py_NE) };
    if result < 0 {
        return Err(PyErr::fetch(obj.py()));
    }
    Ok(result == 1)
}

/// Whether ``obj`` is an exact dict whose keys are all exact ``str``.
pub fn is_str_keyed_dict(obj: &B<'_>) -> bool {
    if tp(obj) != t_dict() {
        return false;
    }
    let mut position: ffi::Py_ssize_t = 0;
    let mut key = std::ptr::null_mut();
    let mut value = std::ptr::null_mut();
    unsafe {
        while ffi::PyDict_Next(obj.as_ptr(), &mut position, &mut key, &mut value) != 0 {
            if ffi::Py_TYPE(key) != t_str() {
                return false;
            }
        }
    }
    true
}

/// Every entry of an exact dict, read at once. No Python code runs while it
/// reads, so nothing can change the dict mid-read.
pub fn dict_entries<'py>(dict: &Bound<'py, PyDict>) -> Vec<(B<'py>, B<'py>)> {
    let py = dict.py();
    let mut entries = Vec::with_capacity(dict.len());
    let mut position: ffi::Py_ssize_t = 0;
    let mut key: *mut ffi::PyObject = std::ptr::null_mut();
    let mut value: *mut ffi::PyObject = std::ptr::null_mut();
    unsafe {
        while ffi::PyDict_Next(dict.as_ptr(), &mut position, &mut key, &mut value) != 0 {
            entries.push((
                Bound::from_borrowed_ptr(py, key),
                Bound::from_borrowed_ptr(py, value),
            ));
        }
    }
    entries
}

/// An exact dict's items, read live as ``dict.items()`` would be.
///
/// User code can run between two items (a hook, an ``__init__``) and change
/// the dict. ``PyDict_Next`` walks entries in the order CPython's item
/// iterator does (3.12 to 3.14), so the iterator's two checks, made here the
/// same way, raise the same errors at the same item: the size it started
/// with, and no more items than that.
pub struct DictItems<'py> {
    dict: Bound<'py, PyDict>,
    position: ffi::Py_ssize_t,
    /// The size when iteration began; -1 once a size change was reported,
    /// so it is reported again, as the iterator's is.
    used: ffi::Py_ssize_t,
    left: ffi::Py_ssize_t,
    done: bool,
}

impl<'py> DictItems<'py> {
    pub fn new(dict: &Bound<'py, PyDict>) -> PyResult<Self> {
        let used = unsafe { ffi::PyDict_Size(dict.as_ptr()) };
        Ok(Self {
            dict: dict.clone(),
            position: 0,
            used,
            left: used,
            done: false,
        })
    }

    #[inline]
    pub fn next_item(&mut self) -> PyResult<Option<(B<'py>, B<'py>)>> {
        if self.done {
            return Ok(None);
        }
        let dict = self.dict.as_ptr();
        if unsafe { ffi::PyDict_Size(dict) } != self.used {
            self.used = -1;
            return Err(PyRuntimeError::new_err(
                "dictionary changed size during iteration",
            ));
        }
        let mut key = std::ptr::null_mut();
        let mut value = std::ptr::null_mut();
        if unsafe { ffi::PyDict_Next(dict, &mut self.position, &mut key, &mut value) } == 0 {
            self.done = true;
            return Ok(None);
        }
        if self.left == 0 {
            self.done = true;
            return Err(PyRuntimeError::new_err(
                "dictionary keys changed during iteration",
            ));
        }
        self.left -= 1;
        let py = self.dict.py();
        // SAFETY: both are borrowed from the live dict under the GIL and
        // owned here before any other code can run.
        unsafe {
            Ok(Some((
                Bound::from_borrowed_ptr(py, key),
                Bound::from_borrowed_ptr(py, value),
            )))
        }
    }
}

fn unpack_pair<'py>(pair: B<'py>) -> PyResult<(B<'py>, B<'py>)> {
    let kind = tp(&pair);
    if kind == t_tuple() {
        let pair = unsafe { pair.cast_unchecked::<PyTuple>() };
        if pair.len() == 2 {
            return Ok((pair.get_item(0)?, pair.get_item(1)?));
        }
    } else if kind == t_list() && pair.len()? == 2 {
        return Ok((pair.get_item(0)?, pair.get_item(1)?));
    }
    let mut iter = pair.try_iter()?;
    let first = iter
        .next()
        .transpose()?
        .ok_or_else(|| PyValueError::new_err("not enough values to unpack (expected 2, got 0)"))?;
    let second = iter
        .next()
        .transpose()?
        .ok_or_else(|| PyValueError::new_err("not enough values to unpack (expected 2, got 1)"))?;
    if iter.next().transpose()?.is_some() {
        return Err(PyValueError::new_err(
            "too many values to unpack (expected 2)",
        ));
    }
    Ok((first, second))
}

/// The items of any mapping: a dict directly, else through ``.items()``.
pub enum MapItems<'py> {
    Dict(DictItems<'py>),
    Iter(Bound<'py, PyIterator>),
}

impl<'py> MapItems<'py> {
    pub fn new(mapping: &B<'py>) -> PyResult<Self> {
        if tp(mapping) == t_dict() {
            return Ok(Self::Dict(DictItems::new(unsafe {
                mapping.cast_unchecked::<PyDict>()
            })?));
        }
        let items = mapping.call_method0(pyo3::intern!(mapping.py(), "items"))?;
        Ok(Self::Iter(items.try_iter()?))
    }

    #[inline]
    pub fn next_item(&mut self) -> PyResult<Option<(B<'py>, B<'py>)>> {
        match self {
            Self::Dict(items) => items.next_item(),
            Self::Iter(iter) => iter.next().transpose()?.map(unpack_pair).transpose(),
        }
    }
}

/// The items of any iterable, a list read live as its iterator would.
pub enum SeqItems<'py> {
    List(Bound<'py, PyList>, usize),
    Tuple(Bound<'py, PyTuple>, usize),
    Iter(Bound<'py, PyIterator>),
}

impl<'py> SeqItems<'py> {
    pub fn new(seq: &B<'py>) -> PyResult<Self> {
        let kind = tp(seq);
        if kind == t_list() {
            return Ok(Self::List(
                unsafe { seq.cast_unchecked::<PyList>() }.clone(),
                0,
            ));
        }
        if kind == t_tuple() {
            return Ok(Self::Tuple(
                unsafe { seq.cast_unchecked::<PyTuple>() }.clone(),
                0,
            ));
        }
        Ok(Self::Iter(seq.try_iter()?))
    }

    #[inline]
    pub fn next_item(&mut self) -> PyResult<Option<B<'py>>> {
        match self {
            Self::List(list, index) => {
                if *index >= list.len() {
                    return Ok(None);
                }
                let item = list.get_item(*index)?;
                *index += 1;
                Ok(Some(item))
            }
            Self::Tuple(tuple, index) => {
                if *index >= tuple.len() {
                    return Ok(None);
                }
                let item = tuple.get_item(*index)?;
                *index += 1;
                Ok(Some(item))
            }
            Self::Iter(iter) => iter.next().transpose(),
        }
    }
}

pub fn pystr<'py>(py: Python<'py>, value: &str) -> B<'py> {
    PyString::new(py, value).into_any()
}

/// An optional keyword argument whose absence differs from an explicit ``None``.
pub struct Maybe<'py>(pub Option<B<'py>>);

impl<'a, 'py> FromPyObject<'a, 'py> for Maybe<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(Some(obj.to_owned())))
    }
}
