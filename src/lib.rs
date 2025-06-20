use circuit::{CliffordTCircuit, CliffordTGate};
use num_complex::Complex;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyString};

pub mod circuit;
mod generator;
#[cfg(feature = "gpu")]
mod gpu_generator;
mod simulate;
mod tableau;
mod utils;

use rayon::ThreadPoolBuilder;
pub use simulate::{simulate_circuit, simulate_circuit_parallel, simulate_circuit_parallel1};

#[cfg(feature = "gpu")]
pub use simulate::simulate_circuit_gpu;

use crate::simulate::simulate_circuit_parallel2;

fn parse_basis_state(w: &Bound<PyString>, n: usize) -> PyResult<Vec<bool>> {
    let w_str = w.to_str()?;
    let w_len = w_str.len();
    if w_len != n {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Basis state with length {w_len} does not match circuit with {n} qubits"
        )));
    }
    let res: Result<Vec<bool>, PyErr> = w_str
        .chars()
        .map(|c| match c {
            '0' => Ok(false),
            '1' => Ok(true),
            _ => Err(PyErr::new::<PyValueError, _>(format!(
                "Basis state must only contain '0' and '1', found '{c}'"
            ))),
        })
        .collect();
    Ok(res?)
}

#[pyfunction]
#[pyo3(name = "simulate_circuit")]
fn py_simulate_circuit(w: &Bound<PyString>, circuit: &CliffordTCircuit) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit(
        &parse_basis_state(w, circuit.qubits())?,
        circuit,
    ))
}

#[pyfunction]
#[pyo3(name = "simulate_circuit_parallel")]
fn py_simulate_circuit_parallel(
    w: &Bound<PyString>,
    circuit: &CliffordTCircuit,
) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit_parallel(
        &parse_basis_state(w, circuit.qubits())?,
        circuit,
    ))
}

#[pyfunction]
#[pyo3(name = "simulate_circuit_parallel1")]
fn py_simulate_circuit_parallel1(
    w: &Bound<PyString>,
    circuit: &CliffordTCircuit,
) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit_parallel1(
        &parse_basis_state(w, circuit.qubits())?,
        circuit,
    ))
}

#[pyfunction]
#[pyo3(name = "simulate_circuit_parallel2")]
fn py_simulate_circuit_parallel2(
    w: &Bound<PyString>,
    circuit: &CliffordTCircuit,
) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit_parallel2(
        &parse_basis_state(w, circuit.qubits())?,
        circuit,
    ))
}

#[pymodule]
#[pyo3(name = "qlit")]
pub fn python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ThreadPoolBuilder::new().build_global().unwrap();

    m.add_class::<CliffordTGate>()?;
    m.add_class::<CliffordTCircuit>()?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit_parallel1, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit_parallel2, m)?)?;
    Ok(())
}
