use std::{f64::consts::SQRT_2, vec};

use num_complex::Complex;

use crate::{
    clifford_circuit::{CliffordTCircuit, CliffordTGate},
    generator::Generator,
    gpu_generator::GpuGenerator,
};

const C_I: Complex<f64> = Complex {
    re: (SQRT_2 + 1.0) / (2.0 * SQRT_2),
    im: 1.0 / (2.0 * SQRT_2),
};
const C_Z: Complex<f64> = Complex {
    re: (SQRT_2 - 1.0) / (2.0 * SQRT_2),
    im: -1.0 / (2.0 * SQRT_2),
};

/// Compute the coefficient of the given basis state, `w`,
/// after `circuit` has been applied to the zero state.
///
/// # Panics
/// If `w` has a length different from `circuit.qubits()`.
pub fn simulate_circuit(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let w_len = w.len();
    let n = circuit.qubits();
    let t = circuit.t_gates();
    assert_eq!(
        w_len, n,
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    let mut path = vec![false; t];
    let mut coeff = Complex::ZERO;
    let mut done = false;

    while !done {
        let mut x = vec![false; n];
        let mut x_coeff = Complex::ONE;
        let mut g = Generator::zero(n);
        let mut seen_t_gates = 0;
        for &gate in circuit.gates() {
            match gate {
                CliffordTGate::S(a) => {
                    if x[a] == true {
                        x_coeff *= Complex::I;
                    }
                    g.apply_s_gate(a);
                }
                CliffordTGate::Cnot(a, b) => {
                    x[b] ^= x[a];
                    g.apply_cnot_gate(a, b);
                }
                CliffordTGate::H(a) => {
                    let r = g.coeff_ratio_flipped_bit(&x, a);
                    if r != -Complex::ONE {
                        x_coeff *= (r + 1.0) / SQRT_2;
                        x[a] = false;
                    } else {
                        if x[a] == false {
                            x_coeff *= 2.0 / SQRT_2;
                        } else {
                            x_coeff *= -2.0 / SQRT_2;
                        }
                        x[a] = true;
                    }
                    g.apply_h_gate(a);
                }
                CliffordTGate::T(a) => {
                    if path[seen_t_gates] == false {
                        x_coeff *= C_I;
                    } else {
                        if x[a] == true {
                            x_coeff *= -1.0;
                        }
                        g.apply_z_gate(a);
                        x_coeff *= C_Z;
                    }
                    seen_t_gates += 1;
                }
            }
        }

        coeff += x_coeff * g.coeff_ratio(&x, w);

        done = next_path(&mut path);
    }

    coeff
}

/// Compute the coefficient of the given basis state, `w`,
/// after `circuit` has been applied to the zero state.
///
/// # Panics
/// If `w` has a length different from `circuit.qubits()`.
pub fn simulate_circuit_gpu(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let w_len = w.len();
    let n = circuit.qubits();
    let t = circuit.t_gates();
    assert_eq!(
        w_len, n,
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    let mut path = vec![false; t];
    let mut coeff = Complex::ZERO;
    let mut done = false;

    while !done {
        let mut x = vec![false; n];
        let mut x_coeff = Complex::ONE;
        let mut g = GpuGenerator::zero(n);
        let mut seen_t_gates = 0;
        for &gate in circuit.gates() {
            match gate {
                CliffordTGate::S(a) => {
                    if x[a] == true {
                        x_coeff *= Complex::I;
                    }
                    g.apply_s_gate(a);
                }
                CliffordTGate::Cnot(a, b) => {
                    x[b] ^= x[a];
                    g.apply_cnot_gate(a, b);
                }
                CliffordTGate::H(a) => {
                    let r = g.coeff_ratio_flipped_bit(&x, a);
                    if r != -Complex::ONE {
                        x_coeff *= (r + 1.0) / SQRT_2;
                        x[a] = false;
                    } else {
                        if x[a] == false {
                            x_coeff *= 2.0 / SQRT_2;
                        } else {
                            x_coeff *= -2.0 / SQRT_2;
                        }
                        x[a] = true;
                    }
                    g.apply_h_gate(a);
                }
                CliffordTGate::T(a) => {
                    if path[seen_t_gates] == false {
                        x_coeff *= C_I;
                    } else {
                        if x[a] == true {
                            x_coeff *= -1.0;
                        }
                        g.apply_z_gate(a);
                        x_coeff *= C_Z;
                    }
                    seen_t_gates += 1;
                }
            }
        }

        coeff += x_coeff * g.coeff_ratio(&x, w);

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
