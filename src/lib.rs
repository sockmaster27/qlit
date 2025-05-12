use clifford_circuit::{CliffordCircuit, CliffordGate};
use generator_col::GeneratorCol;
use generator_row::GeneratorRow;
use gpu_generator::{initialize_gpu, GpuGenerator};
use num_complex::Complex;
use pyo3::{prelude::*, types::PyString};

pub mod clifford_circuit;
mod generator_col;
mod generator_row;
mod gpu_generator;

/// Represents the 3 categories of probabilities for a basis state.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisStateProbability {
    /// This basis state is the only one that is possible to sample.
    One,
    /// This basis state is one of several possible states that can be sampled.
    InBetween,
    /// This basis state is not possible to sample.
    Zero,
}

/// Get the probability of reading the given basis state, `w`, for the given Clifford circuit.
pub fn simulate_clifford_circuit_gpu(
    w: &[bool],
    circuit: &CliffordCircuit,
) -> BasisStateProbability {
    let mut g = GpuGenerator::zero(circuit.qubits().try_into().unwrap());
    g.apply_gates(circuit.gates());
    g.probability(w)
}

pub fn coeff_ratio(w1: &[bool], w2: &[bool], circuit: &CliffordCircuit) -> Complex<i8> {
    let mut g = GeneratorCol::zero(circuit.qubits());
    for &gate in circuit.gates() {
        g.apply_gate(gate);
    }
    g.coeff_ratio(w1, w2)
}

pub fn coeff_ratio_row(w1: &[bool], w2: &[bool], circuit: &CliffordCircuit) -> Complex<i8> {
    let mut g = GeneratorRow::zero(circuit.qubits());
    for &gate in circuit.gates() {
        g.apply_gate(gate);
    }
    g.coeff_ratio(w1, w2)
}

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

/// Get the probability of reading the given basis state, `w`, for the given Clifford circuit.
#[pyfunction]
#[pyo3(name = "simulate_clifford_circuit_gpu")]
fn py_simulate_clifford_circuit_gpu(
    w: &Bound<PyString>,
    circuit: &CliffordCircuit,
) -> PyResult<BasisStateProbability> {
    Ok(simulate_clifford_circuit_gpu(
        &parse_basis_state(w)?,
        circuit,
    ))
}

#[pyfunction]
#[pyo3(name = "coeff_ratio")]
fn py_coeff_ratio(
    w1: &Bound<PyString>,
    w2: &Bound<PyString>,
    circuit: &CliffordCircuit,
) -> PyResult<Complex<f64>> {
    let r = coeff_ratio(&parse_basis_state(w1)?, &parse_basis_state(w2)?, circuit);
    Ok(Complex {
        re: r.re.into(),
        im: r.im.into(),
    })
}

#[pyfunction]
#[pyo3(name = "coeff_ratio_row")]
fn py_coeff_ratio_row(
    w1: &Bound<PyString>,
    w2: &Bound<PyString>,
    circuit: &CliffordCircuit,
) -> PyResult<Complex<f64>> {
    let r = coeff_ratio_row(&parse_basis_state(w1)?, &parse_basis_state(w2)?, circuit);
    Ok(Complex {
        re: r.re.into(),
        im: r.im.into(),
    })
}

#[pymodule]
#[pyo3(name = "qlit")]
pub fn python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    initialize_gpu();
    m.add_class::<CliffordGate>()?;
    m.add_class::<CliffordCircuit>()?;
    m.add_class::<BasisStateProbability>()?;
    m.add_function(wrap_pyfunction!(py_simulate_clifford_circuit_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(py_coeff_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(py_coeff_ratio_row, m)?)?;
    Ok(())
}
