use std::{
    cmp::min,
    f64::consts::{FRAC_1_SQRT_2, SQRT_2},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    vec,
};

use num_complex::Complex;
use rayon::Scope;

use crate::{
    circuit::{CliffordTCircuit, CliffordTGate},
    generator::Generator,
    generator1,
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

/// Apply the gates in `gates` to `x`, `x_coeff` and `g`.
///
/// For each `T`/`Tdg` gate in the circuit, `path` determines whether it is applied as a `Z` gate or not.
///
/// This is done in such a way to maintain the invariant that
/// `x_coeff = <x|psi> != 0`.
fn apply_gates_for_path(
    x: &mut [bool],
    x_coeff: &mut Complex<f64>,
    g: &mut Generator,
    path: &[bool],
    gates: &[CliffordTGate],
) {
    let mut seen_t_gates = 0;
    for &gate in gates {
        match gate {
            CliffordTGate::X(a) => {
                x[a] = !x[a];
                g.apply_x_gate(a);
            }
            CliffordTGate::Y(a) => {
                if x[a] == true {
                    *x_coeff *= -Complex::I;
                } else {
                    *x_coeff *= Complex::I;
                }
                x[a] = !x[a];
                g.apply_y_gate(a);
            }
            CliffordTGate::Z(a) => {
                if x[a] == true {
                    *x_coeff *= -Complex::ONE;
                }
                g.apply_z_gate(a);
            }
            CliffordTGate::S(a) => {
                if x[a] == true {
                    *x_coeff *= Complex::I;
                }
                g.apply_s_gate(a);
            }
            CliffordTGate::Sdg(a) => {
                if x[a] == true {
                    *x_coeff *= -Complex::I;
                }
                g.apply_sdg_gate(a);
            }
            CliffordTGate::Cnot(a, b) => {
                x[b] ^= x[a];
                g.apply_cnot_gate(a, b);
            }
            CliffordTGate::Cz(a, b) => {
                if x[a] == true && x[b] == true {
                    *x_coeff *= -Complex::ONE;
                }
                g.apply_cz_gate(a, b);
            }
            CliffordTGate::H(a) => {
                let r = g.coeff_ratio_flipped_bit(&x, a);
                if r != -Complex::ONE {
                    *x_coeff *= (r + 1.0) / SQRT_2;
                    x[a] = false;
                } else {
                    if x[a] == false {
                        *x_coeff *= 2.0 / SQRT_2;
                    } else {
                        *x_coeff *= -2.0 / SQRT_2;
                    }
                    x[a] = true;
                }
                g.apply_h_gate(a);
            }

            CliffordTGate::T(a) => {
                if path[seen_t_gates] == false {
                    *x_coeff *= C_I;
                } else {
                    if x[a] == true {
                        *x_coeff *= -Complex::ONE;
                    }
                    g.apply_z_gate(a);
                    *x_coeff *= C_Z;
                }
                seen_t_gates += 1;
            }
            CliffordTGate::Tdg(a) => {
                if path[seen_t_gates] == false {
                    *x_coeff *= C_I_DG;
                } else {
                    if x[a] == true {
                        *x_coeff *= -Complex::ONE;
                    }
                    g.apply_z_gate(a);
                    *x_coeff *= C_Z_DG;
                }
                seen_t_gates += 1;
            }
        }
    }
}

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
        apply_gates_for_path(&mut x, &mut x_coeff, &mut g, &path, &circuit.gates());
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

    rayon::in_place_scope(|s| {
        let threads = min(
            num_cpus::get(),
            2usize.saturating_pow(t.try_into().unwrap_or(u32::MAX)),
        );
        for _ in 0..threads {
            s.spawn(|_| {
                let mut w_coeff_local = Complex::ZERO;
                loop {
                    let mut next_path_locked = next_path.lock().unwrap();
                    if done.load(Ordering::SeqCst) {
                        break;
                    }
                    let path = next_path_locked.clone();
                    done.store(increment_path(&mut *next_path_locked), Ordering::SeqCst);
                    drop(next_path_locked);

                    let mut x = vec![false; n];
                    let mut x_coeff = Complex::ONE;
                    let mut g = Generator::zero(n);
                    apply_gates_for_path(&mut x, &mut x_coeff, &mut g, &path, &circuit.gates());

                    w_coeff_local += x_coeff * g.coeff_ratio(&x, w);
                }
                *w_coeff.lock().unwrap() += w_coeff_local;
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
pub fn simulate_circuit_parallel1(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let w_len = w.len();
    let n = circuit.qubits();
    assert_eq!(
        w_len, n,
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    fn apply_gates<'a>(
        s: &Scope<'a>,
        mut x: Vec<bool>,
        mut x_coeff: Complex<f64>,
        mut g: Generator,
        gates: &'a [CliffordTGate],
        w: &'a [bool],
        w_coeff: &'a Mutex<Complex<f64>>,
    ) {
        for (i, &gate) in gates.iter().enumerate() {
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
                    let xp = x.clone();
                    let mut xp_coeff = x_coeff.clone();
                    let mut gp = g.clone();
                    if xp[a] == true {
                        xp_coeff *= -Complex::ONE;
                    }
                    gp.apply_z_gate(a);
                    xp_coeff *= C_Z;
                    s.spawn(move |s| {
                        apply_gates(s, xp, xp_coeff, gp, &gates[i + 1..], w, w_coeff);
                    });

                    x_coeff *= C_I;
                }
                CliffordTGate::Tdg(a) => {
                    let xp = x.clone();
                    let mut xp_coeff = x_coeff.clone();
                    let mut gp = g.clone();
                    if xp[a] == true {
                        xp_coeff *= -Complex::ONE;
                    }
                    gp.apply_z_gate(a);
                    xp_coeff *= C_Z_DG;
                    s.spawn(move |s| {
                        apply_gates(s, xp, xp_coeff, gp, &gates[i + 1..], w, w_coeff);
                    });

                    x_coeff *= C_I_DG;
                }
            }
        }

        *w_coeff.lock().unwrap() += x_coeff * g.coeff_ratio(&x, w);
    }

    let w_coeff = Mutex::new(Complex::ZERO);
    let x = vec![false; n];
    let x_coeff = Complex::ONE;
    let g = Generator::zero(n);
    rayon::in_place_scope(|s| {
        apply_gates(s, x, x_coeff, g, &circuit.gates(), w, &w_coeff);
    });

    let res = *w_coeff.lock().unwrap();
    res
}

/// TODO: Find an optimal value for this.
const MAX_BATCH_SIZE_LOG2: usize = 16;

/// Compute the coefficient of the given basis state, `w`,
/// after `circuit` has been applied to the zero state.
///
/// # Panics
/// If `w` has a length different from `circuit.qubits()`.
pub fn simulate_circuit_parallel2(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let w_len = w.len();
    let n = circuit.qubits();
    let t = circuit.t_gates();
    assert_eq!(
        w_len, n,
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    let threads = min(
        num_cpus::get(),
        2usize.saturating_pow(t.try_into().unwrap_or(u32::MAX)),
    );
    let threads_log2 = threads
        .next_power_of_two()
        .ilog2()
        .try_into()
        .unwrap_or(usize::MAX);

    // Ensure that there is at least one batch per thread.
    let batch_size_log2 = min(MAX_BATCH_SIZE_LOG2, t - threads_log2);

    let next_path = Mutex::new(vec![false; t - batch_size_log2]);
    let w_coeff = Mutex::new(Complex::ZERO);
    let done = AtomicBool::new(false);

    rayon::in_place_scope(|s| {
        for _ in 0..threads {
            s.spawn(|_| {
                let mut w_coeff_local = Complex::ZERO;
                loop {
                    let mut next_path_locked = next_path.lock().unwrap();
                    if done.load(Ordering::SeqCst) {
                        break;
                    }
                    let path = next_path_locked.clone();
                    done.store(increment_path(&mut *next_path_locked), Ordering::SeqCst);
                    drop(next_path_locked);

                    let mut r_cols = 1;
                    let mut xs = vec![vec![false; n]];
                    let mut x_coeffs = vec![Complex::ONE];
                    let mut g = generator1::Generator::zero(n, batch_size_log2);
                    let mut seen_t_gates = 0;
                    for &gate in circuit.gates() {
                        match gate {
                            CliffordTGate::X(a) => {
                                for i in 0..r_cols {
                                    assert!(i < xs.len(), "{seen_t_gates} --- {xs:?}");
                                    xs[i][a] = !xs[i][a];
                                }
                                g.apply_x_gate(a);
                            }
                            CliffordTGate::Y(a) => {
                                for i in 0..r_cols {
                                    assert!(i < xs.len(), "{seen_t_gates} --- {xs:?}");
                                    if xs[i][a] == true {
                                        x_coeffs[i] *= -Complex::I;
                                    } else {
                                        x_coeffs[i] *= Complex::I;
                                    }
                                    xs[i][a] = !xs[i][a];
                                }
                                g.apply_y_gate(a);
                            }
                            CliffordTGate::Z(a) => {
                                for i in 0..r_cols {
                                    assert!(i < xs.len(), "{seen_t_gates} --- {xs:?}");
                                    if xs[i][a] == true {
                                        x_coeffs[i] *= -Complex::ONE;
                                    }
                                }
                                g.apply_z_gate(a);
                            }
                            CliffordTGate::S(a) => {
                                for i in 0..r_cols {
                                    assert!(i < xs.len(), "{seen_t_gates} --- {xs:?}");
                                    if xs[i][a] == true {
                                        x_coeffs[i] *= Complex::I;
                                    }
                                }
                                g.apply_s_gate(a);
                            }
                            CliffordTGate::Sdg(a) => {
                                for i in 0..r_cols {
                                    assert!(i < xs.len(), "{seen_t_gates} --- {xs:?}");
                                    if xs[i][a] == true {
                                        x_coeffs[i] *= -Complex::I;
                                    }
                                }
                                g.apply_sdg_gate(a);
                            }
                            CliffordTGate::Cnot(a, b) => {
                                for i in 0..r_cols {
                                    assert!(i < xs.len(), "{seen_t_gates} --- {xs:?}");
                                    xs[i][b] ^= xs[i][a];
                                }
                                g.apply_cnot_gate(a, b);
                            }
                            CliffordTGate::Cz(a, b) => {
                                for i in 0..r_cols {
                                    assert!(i < xs.len(), "{seen_t_gates} --- {xs:?}");
                                    if xs[i][a] == true && xs[i][b] == true {
                                        x_coeffs[i] *= -Complex::ONE;
                                    }
                                }
                                g.apply_cz_gate(a, b);
                            }
                            CliffordTGate::H(a) => {
                                for i in 0..r_cols {
                                    assert!(i < xs.len(), "{seen_t_gates} --- {xs:?}");
                                    let r = g.coeff_ratio_flipped_bit(i, &xs[i], a);
                                    if r != -Complex::ONE {
                                        x_coeffs[i] *= (r + 1.0) / SQRT_2;
                                        xs[i][a] = false;
                                    } else {
                                        if xs[i][a] == false {
                                            x_coeffs[i] *= 2.0 / SQRT_2;
                                        } else {
                                            x_coeffs[i] *= -2.0 / SQRT_2;
                                        }
                                        xs[i][a] = true;
                                    }
                                }
                                g.apply_h_gate(a);
                            }

                            CliffordTGate::T(a) => {
                                if seen_t_gates < path.len() {
                                    if path[seen_t_gates] == false {
                                        for i in 0..r_cols {
                                            x_coeffs[i] *= C_I;
                                        }
                                    } else {
                                        for i in 0..r_cols {
                                            if xs[i][a] == true {
                                                x_coeffs[i] *= -Complex::ONE;
                                            }
                                            x_coeffs[i] *= C_Z;
                                        }
                                        g.apply_z_gate(a);
                                    }
                                } else {
                                    for i in 0..r_cols {
                                        let index_i = i;
                                        let index_z = i + r_cols;
                                        xs.push(xs[index_i].clone());
                                        x_coeffs.push(x_coeffs[index_i]);

                                        x_coeffs[index_i] *= C_I;

                                        if xs[index_z][a] == true {
                                            x_coeffs[index_z] *= -Complex::ONE;
                                        }
                                        x_coeffs[index_z] *= C_Z;
                                    }
                                    g.apply_t_gate(a);
                                    r_cols *= 2;
                                }

                                seen_t_gates += 1;
                            }
                            CliffordTGate::Tdg(a) => {
                                if seen_t_gates < path.len() {
                                    if path[seen_t_gates] == false {
                                        for i in 0..r_cols {
                                            x_coeffs[i] *= C_I_DG;
                                        }
                                    } else {
                                        for i in 0..r_cols {
                                            if xs[i][a] == true {
                                                x_coeffs[i] *= -Complex::ONE;
                                            }
                                            x_coeffs[i] *= C_Z_DG;
                                        }
                                        g.apply_z_gate(a);
                                    }
                                } else {
                                    for i in 0..r_cols {
                                        let index_i = i;
                                        let index_z = i + r_cols;
                                        xs.push(xs[index_i].clone());
                                        x_coeffs.push(x_coeffs[index_i]);

                                        x_coeffs[index_i] *= C_I_DG;

                                        if xs[index_z][a] == true {
                                            x_coeffs[index_z] *= -Complex::ONE;
                                        }
                                        x_coeffs[index_z] *= C_Z_DG;
                                    }
                                    g.apply_t_gate(a);
                                    r_cols *= 2;
                                }

                                seen_t_gates += 1;
                            }
                        }
                    }
                    for i in 0..r_cols {
                        w_coeff_local += x_coeffs[i] * g.coeff_ratio(i, &xs[i], w);
                    }
                }
                *w_coeff.lock().unwrap() += w_coeff_local;
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
/// Returns true if this results in the all-false path.
fn increment_path(path: &mut Vec<bool>) -> bool {
    increment_path_by(path, 0)
}
/// Mutate the given path to the `n`th next one, (`n = 2^n_log2`).
/// Returns true if this results in going through the all-false path.
fn increment_path_by(path: &mut Vec<bool>, n_log2: usize) -> bool {
    for i in n_log2..path.len() {
        path[i] = !path[i];
        if path[i] == true {
            return false;
        }
    }
    true
}
