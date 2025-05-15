use pyo3::{ffi::c_str, Python};
use qlit::python_module;

#[test]
fn python() {
    pyo3::append_to_inittab!(python_module);
    Python::with_gil(|py| Python::run(py, c_str!(include_str!("test.py")), None, None).unwrap());
}
