use num_complex::Complex;

use crate::{Circuit, Gate};
use std::f64::consts::FRAC_1_SQRT_2;

/// The Clifford+P gate set contains all Clifford/stabilizer gates,
/// plus P (Pauli sum decompositions).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CliffordPGate {
    X(u32),
    Y(u32),
    Z(u32),
    S(u32),
    Sdg(u32),
    H(u32),
    Cnot(u32, u32),
    Cz(u32, u32),

    /// Piz(ci, cz) = ci*I + cz*Z
    Piz(Complex<f64>, Complex<f64>, u32),
}
#[derive(Debug, Clone, PartialEq)]
pub struct CliffordPCircuit {
    /// The number of qubits in the circuit.
    qubits: u32,
    /// The number of Pauli decomposition gates in the circuit.
    p_gates: usize,
    /// The ordered list of gates in the circuit.
    gates: Vec<CliffordPGate>,
}
impl CliffordPCircuit {
    pub fn qubits(&self) -> u32 {
        self.qubits
    }

    pub fn p_gates(&self) -> usize {
        self.p_gates
    }

    pub fn gates(&self) -> &[CliffordPGate] {
        &self.gates
    }
}

/// Decomposes the given circuit to the Clifford+P gate set.
pub fn decompose(circuit: &Circuit) -> CliffordPCircuit {
    use CliffordPGate::*;

    let qubits = circuit.qubits();
    let mut gates = Vec::with_capacity(circuit.gate_count());
    for gate in circuit {
        let decomp: &[CliffordPGate] = match gate {
            Gate::X(a) => &[X(a)],
            Gate::Y(a) => &[Y(a)],
            Gate::Z(a) => &[Z(a)],
            Gate::S(a) => &[S(a)],
            Gate::Sdg(a) => &[Sdg(a)],
            Gate::H(a) => &[H(a)],
            Gate::Cnot(a, b) => &[Cnot(a, b)],
            Gate::Cz(a, b) => &[Cz(a, b)],

            Gate::T(a) => &[Piz(
                (1.0 + FRAC_1_SQRT_2 * (1.0 + Complex::I)) / 2.0,
                (1.0 - FRAC_1_SQRT_2 * (1.0 + Complex::I)) / 2.0,
                a,
            )],
            Gate::Tdg(a) => &[Piz(
                (1.0 + FRAC_1_SQRT_2 * (1.0 - Complex::I)) / 2.0,
                (1.0 - FRAC_1_SQRT_2 * (1.0 - Complex::I)) / 2.0,
                a,
            )],

            Gate::Rx(t, a) => &[H(a), rz(t, a), H(a)],
            Gate::Ry(t, a) => &[Sdg(a), H(a), rz(t, a), H(a), S(a)],
            Gate::Rz(t, a) => &[rz(t, a)],
        };
        gates.extend(decomp);
    }

    let p_gates = gates
        .iter()
        .filter(|gate| match gate {
            Piz(_, _, _) => true,
            _ => false,
        })
        .count();
    CliffordPCircuit {
        qubits,
        p_gates,
        gates,
    }
}

fn rz(t: f64, a: u32) -> CliffordPGate {
    CliffordPGate::Piz(
        Complex::new((t / 2.0).cos(), 0.0),
        Complex::new(0.0, -(t / 2.0).sin()),
        a,
    )
}
