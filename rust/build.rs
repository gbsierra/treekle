fn main() {
    // Exposes ``Py_3_13``-style cfgs, so code can fall back on older CPython.
    pyo3_build_config::use_pyo3_cfgs();
}
