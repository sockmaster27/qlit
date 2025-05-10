use std::collections::HashMap;

use pyo3::{conversion::FromPyObjectBound, prelude::*};
use rand::{rngs::SmallRng, RngCore, SeedableRng};

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliffordGate {
    S(usize),
    H(usize),
    Cnot(usize, usize),
}
#[pymethods]
impl CliffordGate {
    fn __repr__(&self) -> String {
        match self {
            CliffordGate::S(a) => format!("CliffordGate.S({a})"),
            CliffordGate::H(a) => format!("CliffordGate.H({a})"),
            CliffordGate::Cnot(a, b) => format!("CliffordGate.Cnot({a}, {b})"),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct CliffordCircuit {
    /// The number of qubits in the circuit.
    qubits: usize,
    /// The ordered list of gates in the circuit.
    gates: Vec<CliffordGate>,
}
impl CliffordCircuit {
    pub fn new(qubits: usize, gates: impl IntoIterator<Item = CliffordGate>) -> Self {
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

    /// Get a circuit from a QASM AST.
    ///
    /// # Panics
    /// - If the AST contains non-Clifford gates.
    /// - If the AST is invalid, e.g. contains registers with negative size.
    /// - If the AST includes unsupported features.
    pub fn from_ast(ast: &Vec<qasm::AstNode>) -> Self {
        let mut qubits = 0;
        let mut register_starts = HashMap::new();
        let mut gates = Vec::new();
        for node in ast.iter() {
            match node {
                qasm::AstNode::QReg(name, size) => {
                    let size: usize = (*size).try_into().expect("Register with negative size");
                    if register_starts.contains_key(name) {
                        panic!("Duplicate register definition: {:?}", name);
                    }
                    register_starts.insert(name, qubits);
                    qubits += size;
                }
                qasm::AstNode::ApplyGate(name, qubits, params) => {
                    if !params.is_empty() {
                        panic!("Parameters not supported: {:?}", params);
                    }
                    for arg in qubits.iter() {
                        match arg {
                            qasm::Argument::Register(_) => {
                                panic!("Register arguments not supported")
                            }
                            _ => {}
                        }
                    }
                    match name.as_str() {
                        "s" => {
                            if qubits.len() != 1 {
                                panic!("S gate must have exactly one argument");
                            }
                            let qasm::Argument::Qubit(register, index) = &qubits[0] else {
                                panic!("S gate must have qubit argument");
                            };
                            let index: usize =
                                (*index).try_into().expect("Qubit with negative index");
                            gates.push(CliffordGate::S(register_starts[&register] + index));
                        }
                        "h" => {
                            if qubits.len() != 1 {
                                panic!("H gate must have exactly one argument");
                            }
                            let qasm::Argument::Qubit(register, index) = &qubits[0] else {
                                panic!("H gate must have qubit argument");
                            };
                            let index: usize =
                                (*index).try_into().expect("Qubit with negative index");
                            gates.push(CliffordGate::H(register_starts[&register] + index));
                        }
                        "cx" => {
                            if qubits.len() != 2 {
                                panic!("CNOT gate must have exactly two arguments");
                            }
                            let (
                                qasm::Argument::Qubit(a_register, a_index),
                                qasm::Argument::Qubit(b_register, b_index),
                            ) = (&qubits[0], &qubits[1])
                            else {
                                panic!("CNOT gate must have qubit arguments");
                            };
                            let a_index: usize =
                                (*a_index).try_into().expect("Qubit with negative index");
                            let b_index: usize =
                                (*b_index).try_into().expect("Qubit with negative index");
                            gates.push(CliffordGate::Cnot(
                                register_starts[&a_register] + a_index,
                                register_starts[&b_register] + b_index,
                            ));
                        }
                        _ => panic!("Unsupported gate: {:?}", name),
                    }
                }
                _ => panic!("Unsupported node: {:?}", node),
            }
        }
        CliffordCircuit { qubits, gates }
    }
}
#[pymethods]
impl CliffordCircuit {
    #[new]
    fn py_new(qubits: usize, gates: Bound<'_, PyAny>) -> PyResult<Self> {
        let gates: PyResult<Vec<CliffordGate>> = gates
            .try_iter()?
            .flat_map(|r| r.map(|obj| CliffordGate::from_py_object_bound((&obj).into())))
            .collect();
        Ok(CliffordCircuit::new(qubits, gates?))
    }

    /// Create a random circuit with the given number of `qubits` and `gates`.
    #[staticmethod]
    pub fn random(qubits: usize, gates: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        CliffordCircuit {
            qubits,
            gates: (0..gates)
                .map(|_| {
                    let a = rng.next_u64() as usize % qubits;
                    match rng.next_u32() % 3 {
                        0 => CliffordGate::H(a),
                        1 => CliffordGate::S(a),
                        2 => {
                            let mut b = a;
                            while b == a {
                                b = rng.next_u64() as usize % qubits;
                            }
                            CliffordGate::Cnot(a, b)
                        }
                        _ => unreachable!(),
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

    /// The gates in the circuit, in the order that they are applied.
    #[getter]
    #[pyo3(name = "gates")]
    fn py_gates(&self) -> Vec<CliffordGate> {
        self.gates.clone()
    }
}
