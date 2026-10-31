use num_complex::Complex;

use crate::{Circuit, Gate};
use std::f64::consts::FRAC_1_SQRT_2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateDecomposed {
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
pub struct CircuitDecomposed {
    qubits: u32,
    gates: Vec<GateDecomposed>,
    /// The number of Pauli decomposition gates in the circuit.
    p_gates: usize,
}
impl CircuitDecomposed {
    pub fn qubits(&self) -> u32 {
        self.qubits
    }

    pub fn gates(self) -> impl Iterator<Item = GateDecomposed> {
        self.gates.into_iter()
    }

    pub fn p_gates(&self) -> usize {
        self.p_gates
    }
}

/// Decomposes the given circuit to the Clifford+P gate set.
pub fn decompose(circuit: &Circuit) -> CircuitDecomposed {
    use GateDecomposed::*;

    let qubits = circuit.qubits();
    let mut gates = Vec::with_capacity(circuit.gate_count());
    for gate in circuit {
        let decomp: &[GateDecomposed] = match gate {
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

            Gate::U(t, p, l, a) => &[phase(l, a), Sdg(a), H(a), rz(t, a), H(a), S(a), phase(p, a)],
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
    CircuitDecomposed {
        qubits,
        p_gates,
        gates,
    }
}

fn phase(t: f64, a: u32) -> GateDecomposed {
    let e = Complex::from_polar(1.0, t);
    GateDecomposed::Piz((1.0 + e) / 2.0, (1.0 - e) / 2.0, a)
}

fn rz(t: f64, a: u32) -> GateDecomposed {
    GateDecomposed::Piz(
        Complex::new((t / 2.0).cos(), 0.0),
        Complex::new(0.0, -(t / 2.0).sin()),
        a,
    )
}
