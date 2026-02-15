use num_complex::Complex;
use pyo3::{PyAny, PyErr, exceptions::PyValueError, prelude::*, types::PyString, wrap_pyfunction};
use qlit::{
    CliffordTCircuit, CliffordTGate, initialize_global, simulate_circuit, simulate_circuit_gpu,
    simulate_circuit_hybrid,
};
use rayon::ThreadPoolBuilder;

#[pyclass(from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyo3(name = "CliffordTGate")]
// For now, we unfortunately have to duplicate this
// See https://github.com/PyO3/pyo3/discussions/3368
enum PyCliffordTGate {
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
}
#[pymethods]
impl PyCliffordTGate {
    fn __repr__(&self) -> String {
        use PyCliffordTGate::*;
        let pre = "CliffordTGate.";
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
        }
    }
}
impl From<PyCliffordTGate> for CliffordTGate {
    fn from(gate: PyCliffordTGate) -> Self {
        match gate {
            PyCliffordTGate::X(a) => CliffordTGate::X(a),
            PyCliffordTGate::Y(a) => CliffordTGate::Y(a),
            PyCliffordTGate::Z(a) => CliffordTGate::Z(a),
            PyCliffordTGate::S(a) => CliffordTGate::S(a),
            PyCliffordTGate::Sdg(a) => CliffordTGate::Sdg(a),
            PyCliffordTGate::H(a) => CliffordTGate::H(a),
            PyCliffordTGate::Cnot(a, b) => CliffordTGate::Cnot(a, b),
            PyCliffordTGate::Cz(a, b) => CliffordTGate::Cz(a, b),
            PyCliffordTGate::T(a) => CliffordTGate::T(a),
            PyCliffordTGate::Tdg(a) => CliffordTGate::Tdg(a),
        }
    }
}
impl From<CliffordTGate> for PyCliffordTGate {
    fn from(gate: CliffordTGate) -> Self {
        match gate {
            CliffordTGate::X(a) => PyCliffordTGate::X(a),
            CliffordTGate::Y(a) => PyCliffordTGate::Y(a),
            CliffordTGate::Z(a) => PyCliffordTGate::Z(a),
            CliffordTGate::S(a) => PyCliffordTGate::S(a),
            CliffordTGate::Sdg(a) => PyCliffordTGate::Sdg(a),
            CliffordTGate::H(a) => PyCliffordTGate::H(a),
            CliffordTGate::Cnot(a, b) => PyCliffordTGate::Cnot(a, b),
            CliffordTGate::Cz(a, b) => PyCliffordTGate::Cz(a, b),
            CliffordTGate::T(a) => PyCliffordTGate::T(a),
            CliffordTGate::Tdg(a) => PyCliffordTGate::Tdg(a),
        }
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq)]
#[pyo3(name = "CliffordTCircuit")]
struct PyCliffordTCircuit(CliffordTCircuit);
#[pymethods]
impl PyCliffordTCircuit {
    #[new]
    fn new(qubits: u32, gates: Bound<'_, PyAny>) -> PyResult<Self> {
        let gates: PyResult<Vec<CliffordTGate>> = gates
            .try_iter()?
            .flat_map(|r| {
                r.map(|obj| {
                    obj.extract::<PyCliffordTGate>()
                        .map(Into::into)
                        .map_err(Into::into)
                })
            })
            .collect();
        let res = CliffordTCircuit::new(qubits, gates?);
        match res {
            Ok(circuit) => Ok(PyCliffordTCircuit(circuit)),
            Err(e) => Err(PyErr::new::<PyValueError, _>(e.to_string())),
        }
    }

    /// Create a random circuit with the given number of `qubits` and `gates` of which `t_gates` are T gates.
    #[staticmethod]
    fn random(qubits: u32, gates: usize, t_gates: usize, seed: u64) -> Self {
        PyCliffordTCircuit(CliffordTCircuit::random(qubits, gates, t_gates, seed))
    }

    /// The number of qubits in the circuit.
    #[getter]
    fn qubits(&self) -> u32 {
        self.0.qubits()
    }

    /// The number of T gates in the circuit.
    #[getter]
    fn t_gates(&self) -> usize {
        self.0.t_gates()
    }

    /// The gates in the circuit, in the order that they are applied.
    #[getter]
    fn gates(&self) -> Vec<PyCliffordTGate> {
        self.0
            .gates()
            .iter()
            .cloned()
            .map(PyCliffordTGate::from)
            .collect()
    }
}

fn parse_basis_state(w: &Bound<PyString>, n: u32) -> PyResult<Vec<bool>> {
    let w_str = w.to_cow()?;
    let w_len = w_str.len();
    if w_len.try_into() != Ok(n) {
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
fn py_simulate_circuit(
    w: &Bound<PyString>,
    circuit: &PyCliffordTCircuit,
) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit(
        &parse_basis_state(w, circuit.qubits())?,
        &circuit.0,
    ))
}

#[pyfunction]
#[pyo3(name = "simulate_circuit_gpu")]
fn py_simulate_circuit_gpu(
    w: &Bound<PyString>,
    circuit: &PyCliffordTCircuit,
) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit_gpu(
        &parse_basis_state(w, circuit.qubits())?,
        &circuit.0,
    ))
}

#[pyfunction]
#[pyo3(name = "simulate_circuit_hybrid")]
fn py_simulate_circuit_hybrid(
    w: &Bound<PyString>,
    circuit: &PyCliffordTCircuit,
) -> PyResult<Complex<f64>> {
    Ok(simulate_circuit_hybrid(
        &parse_basis_state(w, circuit.qubits())?,
        &circuit.0,
    ))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ThreadPoolBuilder::new().build_global().unwrap();
    initialize_global();

    m.add_class::<PyCliffordTGate>()?;
    m.add_class::<PyCliffordTCircuit>()?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_circuit_hybrid, m)?)?;
    Ok(())
}
