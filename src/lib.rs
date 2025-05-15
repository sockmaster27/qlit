use clifford_circuit::{CliffordTCircuit, CliffordTGate};
use num_complex::Complex;
use pyo3::{prelude::*, types::PyString};

pub mod clifford_circuit;
mod generator_col;
mod generator_row;
mod simulate;
mod utils;

pub use simulate::simulate_circuit;

fn parse_basis_state(w: &Bound<PyString>) -> PyResult<Vec<bool>> {
    let w_str = w.to_str()?;
    Ok(w_str
        .chars()
        .map(|c| match c {
            '0' => false,
            '1' => true,
            _ => panic!("Invalid basis state: {:?}", w_str),
        })
        .collect())
}

#[pyfunction]
#[pyo3(name = "simulate_circuit")]
fn py_simulate_circuit(w: &Bound<PyString>, circuit: &CliffordTCircuit) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit(&parse_basis_state(w)?, circuit))
}

#[pymodule]
#[pyo3(name = "qlit")]
pub fn python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CliffordTGate>()?;
    m.add_class::<CliffordTCircuit>()?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit, m)?)?;
    Ok(())
}
