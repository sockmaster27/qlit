use std::{
    cmp::min,
    f64::consts::{FRAC_1_SQRT_2, SQRT_2},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
    },
    vec,
};

use num_complex::Complex;

#[cfg(feature = "gpu")]
use crate::simulate_gpu::GpuSimulator;

use crate::{
    circuit::{CliffordTCircuit, CliffordTGate},
    tableau::ExtendedTableau,
};

const N_TOO_LARGE: &str = "Number of qubits too large";
const INDEX_TOO_LARGE: &str = "Qubit index too large";

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

/// TODO: Find an optimal value for this.
const MAX_BATCH_SIZE_LOG2: usize = 20;

/// Compute the coefficient of the given basis state, `w`,
/// after `circuit` has been applied to the zero state.
///
/// This implementation runs exclusively on the CPU making no use of GPU acceleration.
///
/// # Panics
/// If `w` has a length different from `circuit.qubits()`.
pub fn simulate_circuit(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let w_len = w.len();
    let n = circuit.qubits();
    let t = circuit.t_gates();
    assert_eq!(
        n.try_into(),
        Ok(w_len),
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

                    w_coeff_local += run_cpu(w, circuit, &path, batch_size_log2);
                }

                *w_coeff.lock().unwrap() += w_coeff_local;
            });
        }
    });

    w_coeff.into_inner().unwrap()
}

/// Compute the coefficient of the given basis state, `w`,
/// after `circuit` has been applied to the zero state.
///
/// This implementation runs primarily on the GPU.
///
/// # Panics
/// If `w` has a length different from `circuit.qubits()`.
#[cfg(feature = "gpu")]
pub fn simulate_circuit_gpu(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let w_len = w.len();
    let n = circuit.qubits();
    let t = circuit.t_gates();
    assert_eq!(
        n.try_into(),
        Ok(w_len),
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    let batch_size_log2 = min(MAX_BATCH_SIZE_LOG2, t);
    let mut path = vec![false; t - batch_size_log2];
    let mut w_coeff = Complex::ZERO;
    let mut g = GpuSimulator::new(circuit, w, batch_size_log2);
    let mut done = false;
    while !done {
        w_coeff += g.run(&path);
        done = increment_path(&mut path);
    }
    w_coeff
}

/// Compute the coefficient of the given basis state, `w`,
/// after `circuit` has been applied to the zero state.
///
/// This implementation utilizes both the CPU and GPU in parallel.
///
/// # Panics
/// If `w` has a length different from `circuit.qubits()`.
#[cfg(feature = "gpu")]
pub fn simulate_circuit_hybrid(w: &[bool], circuit: &CliffordTCircuit) -> Complex<f64> {
    let w_len = w.len();
    let n = circuit.qubits();
    let t = circuit.t_gates();
    assert_eq!(
        n.try_into(),
        Ok(w_len),
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

    let mut gpu_sim = GpuSimulator::new(circuit, w, batch_size_log2);

    let mut w_coeff_local = Complex::<f64>::ZERO;
    let next_path = Mutex::new(vec![false; t - batch_size_log2]);
    let w_coeff = Mutex::new(Complex::<f64>::ZERO);
    let done = AtomicBool::new(false);

    rayon::in_place_scope(|s| {
        for _ in 0..(threads - 1) {
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

                    w_coeff_local += run_cpu(w, circuit, &path, batch_size_log2);
                }

                *w_coeff.lock().unwrap() += w_coeff_local;
            });
        }

        loop {
            let mut next_path_locked = next_path.lock().unwrap();
            if done.load(Ordering::SeqCst) {
                break;
            }
            let path = next_path_locked.clone();
            done.store(increment_path(&mut *next_path_locked), Ordering::SeqCst);
            drop(next_path_locked);

            w_coeff_local += gpu_sim.run(&path);
        }
    });

    w_coeff_local + w_coeff.into_inner().unwrap()
}

fn run_cpu(
    w: &[bool],
    circuit: &CliffordTCircuit,
    path: &[bool],
    batch_size_log2: usize,
) -> Complex<f64> {
    let n: usize = circuit.qubits().try_into().expect(N_TOO_LARGE);
    let mut r_cols = 1;
    let mut xs = vec![vec![false; n]];
    let mut x_coeffs = vec![Complex::ONE];
    let mut g = ExtendedTableau::zero(n, batch_size_log2);
    let mut seen_t_gates = 0;
    for &gate in circuit.gates() {
        match gate {
            CliffordTGate::X(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..r_cols {
                    xs[i][a] = !xs[i][a];
                }
                g.apply_x_gate(a);
            }
            CliffordTGate::Y(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..r_cols {
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
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..r_cols {
                    if xs[i][a] == true {
                        x_coeffs[i] *= -Complex::ONE;
                    }
                }
                g.apply_z_gate(a);
            }
            CliffordTGate::S(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..r_cols {
                    if xs[i][a] == true {
                        x_coeffs[i] *= Complex::I;
                    }
                }
                g.apply_s_gate(a);
            }
            CliffordTGate::Sdg(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..r_cols {
                    if xs[i][a] == true {
                        x_coeffs[i] *= -Complex::I;
                    }
                }
                g.apply_sdg_gate(a);
            }
            CliffordTGate::Cnot(a, b) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let b: usize = b.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..r_cols {
                    xs[i][b] ^= xs[i][a];
                }
                g.apply_cnot_gate(a, b);
            }
            CliffordTGate::Cz(a, b) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let b: usize = b.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..r_cols {
                    if xs[i][a] == true && xs[i][b] == true {
                        x_coeffs[i] *= -Complex::ONE;
                    }
                }
                g.apply_cz_gate(a, b);
            }
            CliffordTGate::H(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let rs = g.coeff_ratios_flipped_bit(xs.iter().map(Vec::as_slice), a);
                for i in 0..r_cols {
                    let r = rs[i];
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
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
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
                    g.split_r_columns(a);
                    r_cols *= 2;
                }

                seen_t_gates += 1;
            }
            CliffordTGate::Tdg(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
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
                    g.split_r_columns(a);
                    r_cols *= 2;
                }

                seen_t_gates += 1;
            }
        }
    }

    let mut w_coeff = Complex::ZERO;
    let rs = g.coeff_ratios(xs.iter().map(Vec::as_slice), w);
    for i in 0..r_cols {
        w_coeff += x_coeffs[i] * rs[i];
    }
    w_coeff
}

/// Mutate the given path to the next one.
/// Returns true if this results in the all-false path.
fn increment_path(path: &mut Vec<bool>) -> bool {
    for e in path {
        *e = !*e;
        if *e == true {
            return false;
        }
    }
    true
}
