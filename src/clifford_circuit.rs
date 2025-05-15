use pyo3::{conversion::FromPyObjectBound, prelude::*};
use rand::{rngs::SmallRng, Rng, RngCore, SeedableRng};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordGate {
    S(usize),
    H(usize),
    Cnot(usize, usize),
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordTGate {
    S(usize),
    H(usize),
    Cnot(usize, usize),
    T(usize),
}
#[pymethods]
impl CliffordTGate {
    fn __repr__(&self) -> String {
        match self {
            CliffordTGate::S(a) => format!("CliffordTGate.S({a})"),
            CliffordTGate::H(a) => format!("CliffordTGate.H({a})"),
            CliffordTGate::Cnot(a, b) => format!("CliffordTGate.Cnot({a}, {b})"),
            CliffordTGate::T(a) => format!("CliffordTGate.T({a})"),
        }
    }
}
impl From<CliffordGate> for CliffordTGate {
    fn from(gate: CliffordGate) -> Self {
        match gate {
            CliffordGate::S(a) => CliffordTGate::S(a),
            CliffordGate::H(a) => CliffordTGate::H(a),
            CliffordGate::Cnot(a, b) => CliffordTGate::Cnot(a, b),
        }
    }
}

pub struct CliffordCircuit {
    /// The number of qubits in the circuit.
    qubits: usize,
    /// The ordered list of gates in the circuit.
    gates: Vec<CliffordGate>,
}
impl CliffordCircuit {
    pub fn new(qubits: usize, gates: impl IntoIterator<Item = CliffordGate>) -> Self {
        // TODO: Validate gate indices less than n
        let gates: Vec<CliffordGate> = gates.into_iter().collect();
        CliffordCircuit { qubits, gates }
    }

    /// The number of qubits in the circuit.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// The gates in the circuit, in the order that they are applied.
    pub fn gates(&self) -> &[CliffordGate] {
        &self.gates
    }

    /// Create a random circuit with the given number of `qubits` and `gates`.
    pub fn random(qubits: usize, gates: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        CliffordCircuit {
            qubits,
            gates: (0..gates)
                .map(|_| {
                    let a = rng.random_range(0..qubits);
                    match rng.next_u32() % 3 {
                        0 => CliffordGate::H(a),
                        1 => CliffordGate::S(a),
                        2 => {
                            let mut b = a;
                            while b == a {
                                b = rng.random_range(0..qubits);
                            }
                            CliffordGate::Cnot(a, b)
                        }
                        _ => unreachable!(),
                    }
                })
                .collect(),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct CliffordTCircuit {
    /// The number of qubits in the circuit.
    qubits: usize,
    /// The number of T gates in the circuit.
    t_gates: usize,
    /// The ordered list of gates in the circuit.
    gates: Vec<CliffordTGate>,
}
impl CliffordTCircuit {
    pub fn new(qubits: usize, gates: impl IntoIterator<Item = CliffordTGate>) -> Self {
        // TODO: Validate gate indices less than n
        let gates: Vec<CliffordTGate> = gates.into_iter().collect();
        let t_gates = gates
            .iter()
            .filter(|g| matches!(g, CliffordTGate::T(_)))
            .count();
        CliffordTCircuit {
            qubits,
            t_gates,
            gates,
        }
    }

    /// The number of qubits in the circuit.
    pub fn qubits(&self) -> usize {
        self.qubits
    }

    /// The number of T gates in the circuit.
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
            .flat_map(|r| r.map(|obj| CliffordTGate::from_py_object_bound((&obj).into())))
            .collect();
        Ok(CliffordTCircuit::new(qubits, gates?))
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
                    if t_gate_positions.contains(&i) {
                        CliffordTGate::T(a)
                    } else {
                        match rng.random_range(0..=2) {
                            0 => CliffordTGate::H(a),
                            1 => CliffordTGate::S(a),
                            2 => {
                                let mut b = a;
                                while b == a {
                                    b = rng.random_range(0..qubits);
                                }
                                CliffordTGate::Cnot(a, b)
                            }
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
