use std::f64::consts::SQRT_2;

use num_complex::Complex;
use num_traits::One;

use crate::{
    clifford_circuit::{CliffordCircuit, CliffordGate},
    generator_col::GeneratorCol,
};

pub fn clifford_phase(w: &[bool], circuit: &CliffordCircuit) -> Complex<f64> {
    let n = circuit.qubits();
    assert_eq!(w.len(), n, "Basis state must have length {n}");

    let mut x = vec![false; n];
    let mut k = Complex::one();
    let mut g = GeneratorCol::zero(n);
    for &gate in circuit.gates() {
        match gate {
            CliffordGate::S(a) => {
                if x[a] == true {
                    k *= Complex::i();
                }
            }
            CliffordGate::Cnot(a, b) => {
                x[b] ^= x[a];
            }
            CliffordGate::H(a) => {
                let r = g.coeff_ratio_flipped_bit(&x, a);
                if r != -Complex::one() {
                    k *= (r + 1.0) / SQRT_2;
                    x[a] = false;
                } else {
                    if x[a] == false {
                        k *= 2.0 / SQRT_2;
                    } else {
                        k *= -2.0 / SQRT_2;
                    }
                    x[a] = true;
                }
            }
        }
        g.apply_gate(gate);
    }

    k * g.coeff_ratio(&x, w)
}
