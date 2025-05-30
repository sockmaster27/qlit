use std::{
    f64::consts::{FRAC_1_SQRT_2, SQRT_2},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    thread, vec,
};

use num_complex::Complex;

use crate::{
    circuit::{CliffordTCircuit, CliffordTGate},
    generator::Generator,
};

#[cfg(feature = "gpu")]
use crate::gpu_generator::GpuGenerator;

// T = C_I*I + C_Z*Z
const C_I: Complex<f64> = Complex {
    re: 0.5 + 0.5 * FRAC_1_SQRT_2,
    im: 0.5 * FRAC_1_SQRT_2,
};
const C_Z: Complex<f64> = Complex {
    re: 0.5 - 0.5 * FRAC_1_SQRT_2,
    im: -0.5 * FRAC_1_SQRT_2,
};

// Tdg = C_I_DG*I + C_Z_DG*Z
const C_I_DG: Complex<f64> = Complex {
    re: 0.5 + 0.5 * FRAC_1_SQRT_2,
    im: -0.5 * FRAC_1_SQRT_2,
};
const C_Z_DG: Complex<f64> = Complex {
    re: 0.5 - 0.5 * FRAC_1_SQRT_2,
    im: 0.5 * FRAC_1_SQRT_2,
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
    let mut w_coeff = Complex::ZERO;
    let mut done = false;

    while !done {
        let mut x = vec![false; n];
        let mut x_coeff = Complex::ONE;
        let mut g = Generator::zero(n);
        let mut seen_t_gates = 0;
        for &gate in circuit.gates() {
            match gate {
                CliffordTGate::X(a) => {
                    x[a] = !x[a];
                    g.apply_x_gate(a);
                }
                CliffordTGate::Y(a) => {
                    if x[a] == true {
                        x_coeff *= -Complex::I;
                    } else {
                        x_coeff *= Complex::I;
                    }
                    x[a] = !x[a];
                    g.apply_y_gate(a);
                }
                CliffordTGate::Z(a) => {
                    if x[a] == true {
                        x_coeff *= -Complex::ONE;
                    }
                    g.apply_z_gate(a);
                }
                CliffordTGate::S(a) => {
                    if x[a] == true {
                        x_coeff *= Complex::I;
                    }
                    g.apply_s_gate(a);
                }
                CliffordTGate::Sdg(a) => {
                    if x[a] == true {
                        x_coeff *= -Complex::I;
                    }
                    g.apply_sdg_gate(a);
                }
                CliffordTGate::Cnot(a, b) => {
                    x[b] ^= x[a];
                    g.apply_cnot_gate(a, b);
                }
                CliffordTGate::Cz(a, b) => {
                    if x[a] == true && x[b] == true {
                        x_coeff *= -Complex::ONE;
                    }
                    g.apply_cz_gate(a, b);
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
                CliffordTGate::Tdg(a) => {
                    if path[seen_t_gates] == false {
                        x_coeff *= C_I_DG;
                    } else {
                        if x[a] == true {
                            x_coeff *= -1.0;
                        }
                        g.apply_z_gate(a);
                        x_coeff *= C_Z_DG;
                    }
                    seen_t_gates += 1;
                }
            }
        }

        w_coeff += x_coeff * g.coeff_ratio(&x, w);

        done = increment_path(&mut path);
    }

    w_coeff
}

/// Compute the coefficient of the given basis state, `w`,
/// after `circuit` has been applied to the zero state.
///
/// # Panics
/// If `w` has a length different from `circuit.qubits()`.
pub fn simulate_circuit_parallel(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let w_len = w.len();
    let n = circuit.qubits();
    let t = circuit.t_gates();
    assert_eq!(
        w_len, n,
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    let next_path = Mutex::new(vec![false; t]);
    let w_coeff = Mutex::new(Complex::ZERO);
    let done = AtomicBool::new(false);

    thread::scope(|s| {
        for _ in 0..num_cpus::get_physical() {
            s.spawn(|| loop {
                let mut next_path_locked = next_path.lock().unwrap();
                if done.load(Ordering::SeqCst) {
                    return;
                }
                let path = next_path_locked.clone();
                done.store(increment_path(&mut *next_path_locked), Ordering::SeqCst);
                drop(next_path_locked);

                let mut x = vec![false; n];
                let mut x_coeff = Complex::ONE;
                let mut g = Generator::zero(n);
                let mut seen_t_gates = 0;
                for &gate in circuit.gates() {
                    match gate {
                        CliffordTGate::X(a) => {
                            x[a] = !x[a];
                            g.apply_x_gate(a);
                        }
                        CliffordTGate::Y(a) => {
                            if x[a] == true {
                                x_coeff *= -Complex::I;
                            } else {
                                x_coeff *= Complex::I;
                            }
                            x[a] = !x[a];
                            g.apply_y_gate(a);
                        }
                        CliffordTGate::Z(a) => {
                            if x[a] == true {
                                x_coeff *= -Complex::ONE;
                            }
                            g.apply_z_gate(a);
                        }
                        CliffordTGate::S(a) => {
                            if x[a] == true {
                                x_coeff *= Complex::I;
                            }
                            g.apply_s_gate(a);
                        }
                        CliffordTGate::Sdg(a) => {
                            if x[a] == true {
                                x_coeff *= -Complex::I;
                            }
                            g.apply_sdg_gate(a);
                        }
                        CliffordTGate::Cnot(a, b) => {
                            x[b] ^= x[a];
                            g.apply_cnot_gate(a, b);
                        }
                        CliffordTGate::Cz(a, b) => {
                            if x[a] == true && x[b] == true {
                                x_coeff *= -Complex::ONE;
                            }
                            g.apply_cz_gate(a, b);
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
                        CliffordTGate::Tdg(a) => {
                            if path[seen_t_gates] == false {
                                x_coeff *= C_I_DG;
                            } else {
                                if x[a] == true {
                                    x_coeff *= -1.0;
                                }
                                g.apply_z_gate(a);
                                x_coeff *= C_Z_DG;
                            }
                            seen_t_gates += 1;
                        }
                    }
                }
                *w_coeff.lock().unwrap() += x_coeff * g.coeff_ratio(&x, w);
            });
        }
    });

    let res = *w_coeff.lock().unwrap();
    res
}

/// Compute the coefficient of the given basis state, `w`,
/// after `circuit` has been applied to the zero state.
///
/// # Panics
/// If `w` has a length different from `circuit.qubits()`.
#[cfg(feature = "gpu")]
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
                _ => todo!(),
            }
        }

        coeff += x_coeff * g.coeff_ratio(&x, w);

        done = increment_path(&mut path);
    }

    coeff
}

/// Mutate the given path to the next one.
/// Returns true if the given path is the all-true path.
fn increment_path(path: &mut Vec<bool>) -> bool {
    for i in 0..path.len() {
        path[i] = !path[i];
        if path[i] == true {
            return false;
        }
    }
    true
}
