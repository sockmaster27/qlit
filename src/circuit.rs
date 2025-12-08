use std::{error::Error, fmt::Display};

use pyo3::{conversion::FromPyObject, prelude::*};
use rand::{rngs::SmallRng, Rng, SeedableRng};

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordTGate {
    X(usize),
    Y(usize),
    Z(usize),

    S(usize),
    /// The inverse of the S gate.
    Sdg(usize),

    H(usize),

    Cnot(usize, usize),
    Cz(usize, usize),

    T(usize),
    /// The inverse of the T gate.
    Tdg(usize),
}
#[pymethods]
impl CliffordTGate {
    fn __repr__(&self) -> String {
        match self {
            CliffordTGate::X(a) => format!("CliffordTGate.X({a})"),
            CliffordTGate::Y(a) => format!("CliffordTGate.Y({a})"),
            CliffordTGate::Z(a) => format!("CliffordTGate.Z({a})"),

            CliffordTGate::H(a) => format!("CliffordTGate.H({a})"),

            CliffordTGate::S(a) => format!("CliffordTGate.S({a})"),
            CliffordTGate::Sdg(a) => format!("CliffordTGate.Sdg({a})"),

            CliffordTGate::Cnot(a, b) => format!("CliffordTGate.Cnot({a}, {b})"),

            CliffordTGate::Cz(a, b) => format!("CliffordTGate.Cz({a}, {b})"),

            CliffordTGate::T(a) => format!("CliffordTGate.T({a})"),
            CliffordTGate::Tdg(a) => format!("CliffordTGate.Tdg({a})"),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliffordTCircuit {
    /// The number of qubits in the circuit.
    qubits: usize,
    /// The number of `T` and `Tdg` gates in the circuit.
    t_gates: usize,
    /// The ordered list of gates in the circuit.
    gates: Vec<CliffordTGate>,
}
impl CliffordTCircuit {
    pub fn new(
        qubits: usize,
        gates: impl IntoIterator<Item = CliffordTGate>,
    ) -> Result<Self, CircuitCreationError> {
        let gates: Vec<CliffordTGate> = gates.into_iter().collect();
        let mut t_gates = 0;
        let check_index = |a| {
            if a >= qubits {
                return Err(CircuitCreationError::InvalidQubitIndex { index: a, qubits });
            }
            Ok(())
        };
        for &gate in &gates {
            match gate {
                CliffordTGate::X(a) => check_index(a)?,
                CliffordTGate::Y(a) => check_index(a)?,
                CliffordTGate::Z(a) => check_index(a)?,
                CliffordTGate::H(a) => check_index(a)?,
                CliffordTGate::S(a) => check_index(a)?,
                CliffordTGate::Sdg(a) => check_index(a)?,
                CliffordTGate::Cnot(a, b) => {
                    check_index(a)?;
                    check_index(b)?;
                }
                CliffordTGate::Cz(a, b) => {
                    check_index(a)?;
                    check_index(b)?;
                }
                CliffordTGate::T(a) => {
                    check_index(a)?;
                    t_gates += 1;
                }
                CliffordTGate::Tdg(a) => {
                    check_index(a)?;
                    t_gates += 1;
                }
            }
        }
        Ok(CliffordTCircuit {
            qubits,
            t_gates,
            gates,
        })
    }

    /// The number of qubits in the circuit.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// The number of `T` and `Tdg` gates in the circuit.
    pub fn t_gates(&self) -> usize {
        self.t_gates
    }

    /// The gates in the circuit, in the order that they are applied.
    pub fn gates(&self) -> &[CliffordTGate] {
        &self.gates
    }
}
#[pymethods]
impl CliffordTCircuit {
    #[new]
    fn py_new(qubits: usize, gates: Bound<'_, PyAny>) -> PyResult<Self> {
        let gates: PyResult<Vec<CliffordTGate>> = gates
            .try_iter()?
            .flat_map(|r| r.map(|obj| CliffordTGate::extract((&obj).into()).map_err(Into::into)))
            .collect();
        Ok(CliffordTCircuit::new(qubits, gates?)?)
    }

    /// Create a random circuit with the given number of `qubits` and `gates` of which `t_gates` are T gates.
    #[staticmethod]
    pub fn random(qubits: usize, gates: usize, t_gates: usize, seed: u64) -> Self {
        assert!(
            t_gates <= gates,
            "t_gates must be less than or equal to gates"
        );

        let mut rng = SmallRng::seed_from_u64(seed);

        let mut t_gate_positions = Vec::new();
        for _ in 0..t_gates {
            let mut pos;
            loop {
                pos = rng.random_range(0..gates);
                if !t_gate_positions.contains(&pos) {
                    t_gate_positions.push(pos);
                    break;
                }
            }
        }

        CliffordTCircuit {
            qubits,
            t_gates,
            gates: (0..gates)
                .map(|i| {
                    let a = rng.random_range(0..qubits);
                    let mut b = a;
                    while b == a {
                        b = rng.random_range(0..qubits);
                    }
                    if t_gate_positions.contains(&i) {
                        match rng.random_range(0..=1) {
                            0 => CliffordTGate::T(a),
                            1 => CliffordTGate::Tdg(a),
                            _ => unreachable!(),
                        }
                    } else {
                        match rng.random_range(0..=7) {
                            0 => CliffordTGate::X(a),
                            1 => CliffordTGate::Y(a),
                            2 => CliffordTGate::Z(a),
                            3 => CliffordTGate::H(a),
                            4 => CliffordTGate::S(a),
                            5 => CliffordTGate::Sdg(a),
                            6 => CliffordTGate::Cnot(a, b),
                            7 => CliffordTGate::Cz(a, b),
                            _ => unreachable!(),
                        }
                    }
                })
                .collect(),
        }
    }

    /// The number of qubits in the circuit.
    #[getter]
    #[pyo3(name = "qubits")]
    fn py_qubits(&self) -> usize {
        self.qubits
    }

    /// The number of T gates in the circuit.
    #[getter]
    #[pyo3(name = "t_gates")]
    fn py_t_gates(&self) -> usize {
        self.t_gates
    }

    /// The gates in the circuit, in the order that they are applied.
    #[getter]
    #[pyo3(name = "gates")]
    fn py_gates(&self) -> Vec<CliffordTGate> {
        self.gates.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitCreationError {
    InvalidQubitIndex { index: usize, qubits: usize },
}
impl Display for CircuitCreationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitCreationError::InvalidQubitIndex { index, qubits } => {
                write!(
                    f,
                    "Invalid qubit index {index} for circuit of {qubits} qubits"
                )
            }
        }
    }
}
impl Error for CircuitCreationError {}
impl From<CircuitCreationError> for PyErr {
    fn from(value: CircuitCreationError) -> Self {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(value.to_string())
    }
}
