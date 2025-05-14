use std::{f64::consts::SQRT_2, vec};

use num_complex::Complex;
use num_traits::{One, Zero};

use crate::{
    clifford_circuit::{CliffordCircuit, CliffordGate, CliffordTCircuit, CliffordTGate},
    generator_col::GeneratorCol,
};

const C_I: Complex<f64> = Complex {
    re: (SQRT_2 + 1.0) / (2.0 * SQRT_2),
    im: 1.0 / (2.0 * SQRT_2),
};
const C_Z: Complex<f64> = Complex {
    re: (SQRT_2 - 1.0) / (2.0 * SQRT_2),
    im: -1.0 / (2.0 * SQRT_2),
};

pub fn clifford_phase(w: &[bool], circuit: &CliffordCircuit) -> Complex<f64> {
    let n = circuit.qubits();
    assert_eq!(
        w.len(),
        n,
        "Basis state with length {n} does not match circuit with {n} qubits"
    );

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

pub fn simulate_circuit(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let n = circuit.qubits();
    let t = circuit.t_gates();
    assert_eq!(
        w.len(),
        n,
        "Basis state with length {n} does not match circuit with {n} qubits"
    );

    let mut path = vec![false; t];
    let mut coeff = Complex::zero();
    let mut done = false;

    while !done {
        let mut x = vec![false; n];
        let mut k = Complex::one();
        let mut g = GeneratorCol::zero(n);
        let mut r = 0;
        for &gate in circuit.gates() {
            match gate {
                CliffordTGate::S(a) => {
                    if x[a] == true {
                        k *= Complex::i();
                    }
                    g.apply_s_gate(a);
                }
                CliffordTGate::Cnot(a, b) => {
                    x[b] ^= x[a];
                    g.apply_cnot_gate(a, b);
                }
                CliffordTGate::H(a) => {
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
                    g.apply_h_gate(a);
                }
                CliffordTGate::T(a) => {
                    if path[r] == false {
                        k *= C_I;
                    } else {
                        if x[a] == true {
                            k *= -1.0;
                        }
                        g.apply_z_gate(a);
                        k *= C_Z;
                    }
                    r += 1;
                }
            }
        }

        coeff += k * g.coeff_ratio(&x, w);

        done = next_path(&mut path);
    }

    coeff
}

/// Mutate the given path to the next one.
/// Returns true if the given path is the all-true path.
fn next_path(path: &mut Vec<bool>) -> bool {
    for i in 0..path.len() {
        path[i] = !path[i];
        if path[i] == true {
            return false;
        }
    }
    true
}
