use num_complex::Complex;
use pyo3::{PyAny, PyErr, exceptions::PyValueError, prelude::*, types::PyString, wrap_pyfunction};
use qlit::{
    BitString, Circuit, Gate, initialize_global, simulate_circuit, simulate_circuit_gpu,
    simulate_circuit_hybrid,
};

#[pyclass(from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[pyo3(name = "Gate")]
// For now, we unfortunately have to duplicate this
// See https://github.com/PyO3/pyo3/discussions/3368
enum PyGate {
    X(u32),
    Y(u32),
    Z(u32),
    S(u32),
    Sdg(u32),
    H(u32),
    Cnot(u32, u32),
    Cz(u32, u32),
    T(u32),
    Tdg(u32),
    Rz(f64, u32),
}
#[pymethods]
impl PyGate {
    fn __repr__(&self) -> String {
        use PyGate::*;
        let pre = "Gate.";
        match self {
            X(a) => format!("{pre}X({a})"),
            Y(a) => format!("{pre}Y({a})"),
            Z(a) => format!("{pre}Z({a})"),
            H(a) => format!("{pre}H({a})"),
            S(a) => format!("{pre}S({a})"),
            Sdg(a) => format!("{pre}Sdg({a})"),
            Cnot(a, b) => format!("{pre}Cnot({a}, {b})"),
            Cz(a, b) => format!("{pre}Cz({a}, {b})"),
            T(a) => format!("{pre}T({a})"),
            Tdg(a) => format!("{pre}Tdg({a})"),
            Rz(t, a) => format!("{pre}Rx({t}, {a})"),
        }
    }
}
impl From<PyGate> for Gate {
    fn from(gate: PyGate) -> Self {
        match gate {
            PyGate::X(a) => Gate::X(a),
            PyGate::Y(a) => Gate::Y(a),
            PyGate::Z(a) => Gate::Z(a),
            PyGate::S(a) => Gate::S(a),
            PyGate::Sdg(a) => Gate::Sdg(a),
            PyGate::H(a) => Gate::H(a),
            PyGate::Cnot(a, b) => Gate::Cnot(a, b),
            PyGate::Cz(a, b) => Gate::Cz(a, b),
            PyGate::T(a) => Gate::T(a),
            PyGate::Tdg(a) => Gate::Tdg(a),
            PyGate::Rz(t, a) => Gate::Rz(t, a),
        }
    }
}
impl From<Gate> for PyGate {
    fn from(gate: Gate) -> Self {
        match gate {
            Gate::X(a) => PyGate::X(a),
            Gate::Y(a) => PyGate::Y(a),
            Gate::Z(a) => PyGate::Z(a),
            Gate::S(a) => PyGate::S(a),
            Gate::Sdg(a) => PyGate::Sdg(a),
            Gate::H(a) => PyGate::H(a),
            Gate::Cnot(a, b) => PyGate::Cnot(a, b),
            Gate::Cz(a, b) => PyGate::Cz(a, b),
            Gate::T(a) => PyGate::T(a),
            Gate::Tdg(a) => PyGate::Tdg(a),
            Gate::Rz(t, a) => PyGate::Rz(t, a),
        }
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone, PartialEq)]
#[pyo3(name = "Circuit")]
struct PyCircuit(Circuit);
#[pymethods]
impl PyCircuit {
    #[new]
    fn new(qubits: u32, gates: Bound<'_, PyAny>) -> PyResult<Self> {
        let gates: PyResult<Vec<Gate>> = gates
            .try_iter()?
            .flat_map(|r| r.map(|obj| obj.extract::<PyGate>().map(Into::into).map_err(Into::into)))
            .collect();
        let res = Circuit::new(qubits, gates?);
        match res {
            Ok(circuit) => Ok(PyCircuit(circuit)),
            Err(e) => Err(PyErr::new::<PyValueError, _>(e.to_string())),
        }
    }

    /// Create a random circuit from the Clifford+T fragment, with the given number of `qubits` and `gates` of which `t_gates` are T gates.
    #[staticmethod]
    fn random_clifford_t(qubits: u32, gates: usize, t_gates: usize, seed: u64) -> Self {
        PyCircuit(Circuit::random_clifford_t(qubits, gates, t_gates, seed))
    }

    /// The number of qubits in the circuit.
    #[getter]
    fn qubits(&self) -> u32 {
        self.0.qubits()
    }

    /// The gates in the circuit, in the order that they are applied.
    #[getter]
    fn gates(&self) -> Vec<PyGate> {
        self.0.iter().map(PyGate::from).collect()
    }
}

fn parse_basis_state(w: &Bound<PyString>, n: u32) -> PyResult<BitString> {
    let w_str = w.to_cow()?;
    let w_len = w_str.len();
    if w_len.try_into() != Ok(n) {
        return Err(PyErr::new::<PyValueError, _>(format!(
            "Basis state with length {w_len} does not match circuit with {n} qubits"
        )));
    }
    let res: Result<BitString, PyErr> = w_str
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
fn py_simulate_circuit(w: &Bound<PyString>, circuit: &PyCircuit) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit(
        &parse_basis_state(w, circuit.qubits())?,
        &circuit.0,
    ))
}

#[pyfunction]
#[pyo3(name = "simulate_circuit_gpu")]
fn py_simulate_circuit_gpu(w: &Bound<PyString>, circuit: &PyCircuit) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit_gpu(
        &parse_basis_state(w, circuit.qubits())?,
        &circuit.0,
    ))
}

#[pyfunction]
#[pyo3(name = "simulate_circuit_hybrid")]
fn py_simulate_circuit_hybrid(w: &Bound<PyString>, circuit: &PyCircuit) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit_hybrid(
        &parse_basis_state(w, circuit.qubits())?,
        &circuit.0,
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    initialize_global();

    m.add_class::<PyGate>()?;
    m.add_class::<PyCircuit>()?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit_hybrid, m)?)?;
    Ok(())
}
