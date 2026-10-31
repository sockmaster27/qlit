use std::{
    cmp::min,
    f64::consts::SQRT_2,
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
    Circuit,
    bitstring::{BitString, BitStringArray},
    decompose::decompose,
    plan::{CircuitPlanned, GatePlanned, plan},
    tableau::ExtendedTableau,
    utils::{INDEX_TOO_LARGE, N_TOO_LARGE},
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
pub fn simulate_circuit(w: &BitString, circuit: &Circuit) -> Complex<f64> {
    let circuit_decomposed = decompose(circuit);
    let w_len = w.len();
    let n = circuit_decomposed.qubits();
    let t = circuit_decomposed.p_gates();
    assert_eq!(
        n.try_into(),
        Ok(w_len),
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    let threads = min(
        rayon::current_num_threads(),
        2usize.saturating_pow(t.try_into().unwrap_or(u32::MAX)),
    );
    let threads_log2 = threads
        .next_power_of_two()
        .ilog2()
        .try_into()
        .unwrap_or(usize::MAX);
    // Ensure that there is at least one batch per thread.
    let batch_size_log2 = min(MAX_BATCH_SIZE_LOG2, t - threads_log2);
    let circuit_planned = plan(circuit_decomposed, batch_size_log2);

    let next_path = Mutex::new(vec![false; t - batch_size_log2]);
    let w_coeff = Mutex::new(Complex::ZERO);
    let done = AtomicBool::new(false);

    rayon::in_place_scope(|s| {
        s.spawn_broadcast(|_, _| {
            let mut w_coeff_local = Complex::ZERO;
            loop {
                let mut next_path_locked = next_path.lock().unwrap();
                if done.load(Ordering::SeqCst) {
                    break;
                }
                let path = next_path_locked.clone();
                done.store(increment_path(&mut *next_path_locked), Ordering::SeqCst);
                drop(next_path_locked);

                w_coeff_local += run_cpu(w, &circuit_planned, &path, batch_size_log2);
            }

            *w_coeff.lock().unwrap() += w_coeff_local;
        });
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
pub fn simulate_circuit_gpu(w: &BitString, circuit: &Circuit) -> Complex<f64> {
    let circuit_decomposed = decompose(circuit);
    let w_len = w.len();
    let n = circuit_decomposed.qubits();
    let t = circuit_decomposed.p_gates();
    assert_eq!(
        n.try_into(),
        Ok(w_len),
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    let batch_size_log2 = min(MAX_BATCH_SIZE_LOG2, t);
    let circuit_planned = plan(circuit_decomposed, batch_size_log2);

    let mut path = vec![false; t - batch_size_log2];
    let mut w_coeff = Complex::ZERO;
    let mut g = GpuSimulator::new(&circuit_planned, w, batch_size_log2);
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
pub fn simulate_circuit_hybrid(w: &BitString, circuit: &Circuit) -> Complex<f64> {
    let circuit_decomposed = decompose(circuit);
    let w_len = w.len();
    let n = circuit_decomposed.qubits();
    let t = circuit_decomposed.p_gates();
    assert_eq!(
        n.try_into(),
        Ok(w_len),
        "Basis state with length {w_len} does not match circuit with {n} qubits"
    );

    let threads = min(
        rayon::current_num_threads(),
        2usize.saturating_pow(t.try_into().unwrap_or(u32::MAX)),
    );
    let threads_log2 = threads
        .next_power_of_two()
        .ilog2()
        .try_into()
        .unwrap_or(usize::MAX);

    // Ensure that there is at least one batch per thread.
    let batch_size_log2 = min(MAX_BATCH_SIZE_LOG2, t - threads_log2);
    let circuit_planned = plan(circuit_decomposed, batch_size_log2);

    let mut gpu_sim = GpuSimulator::new(&circuit_planned, w, batch_size_log2);

    let mut w_coeff_local = Complex::<f64>::ZERO;
    let next_path = Mutex::new(vec![false; t - batch_size_log2]);
    let w_coeff = Mutex::new(Complex::<f64>::ZERO);
    let done = AtomicBool::new(false);

    rayon::in_place_scope(|s| {
        s.spawn_broadcast(|_, _| {
            let mut w_coeff_local = Complex::ZERO;
            loop {
                let mut next_path_locked = next_path.lock().unwrap();
                if done.load(Ordering::SeqCst) {
                    break;
                }
                let path = next_path_locked.clone();
                done.store(increment_path(&mut *next_path_locked), Ordering::SeqCst);
                drop(next_path_locked);

                w_coeff_local += run_cpu(w, &circuit_planned, &path, batch_size_log2);
            }

            *w_coeff.lock().unwrap() += w_coeff_local;
        });

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
    w: &BitString,
    circuit: &CircuitPlanned,
    path: &[bool],
    batch_size_log2: usize,
) -> Complex<f64> {
    use GatePlanned::*;

    let n: usize = circuit.qubits().try_into().expect(N_TOO_LARGE);
    let mut xs = BitStringArray::new(n, 1 << batch_size_log2);
    let mut x_coeffs = vec![Complex::ONE];
    let mut g = ExtendedTableau::zero(n, batch_size_log2);
    let mut seen_p_gates = 0;
    for &gate in circuit.gates() {
        match gate {
            X(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..g.contained_states() {
                    xs.flip(i, a);
                }
                g.apply_x_gate(a);
            }
            Y(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..g.contained_states() {
                    if xs.get(i, a) {
                        x_coeffs[i] *= -Complex::I;
                    } else {
                        x_coeffs[i] *= Complex::I;
                    }
                    xs.flip(i, a);
                }
                g.apply_y_gate(a);
            }
            Z(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..g.contained_states() {
                    if xs.get(i, a) {
                        x_coeffs[i] *= -Complex::ONE;
                    }
                }
                g.apply_z_gate(a);
            }
            S(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..g.contained_states() {
                    if xs.get(i, a) {
                        x_coeffs[i] *= Complex::I;
                    }
                }
                g.apply_s_gate(a);
            }
            Sdg(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..g.contained_states() {
                    if xs.get(i, a) {
                        x_coeffs[i] *= -Complex::I;
                    }
                }
                g.apply_sdg_gate(a);
            }
            Cnot(a, b) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let b: usize = b.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..g.contained_states() {
                    if xs.get(i, a) {
                        xs.flip(i, b);
                    }
                }
                g.apply_cnot_gate(a, b);
            }
            Cz(a, b) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let b: usize = b.try_into().expect(INDEX_TOO_LARGE);
                for i in 0..g.contained_states() {
                    if xs.get(i, a) && xs.get(i, b) {
                        x_coeffs[i] *= -Complex::ONE;
                    }
                }
                g.apply_cz_gate(a, b);
            }
            H(a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let contained_states = g.contained_states();
                let (r, signs) = g.coeff_ratios_flipped_bit(&xs, a);
                for i in 0..contained_states {
                    let r = if r != Complex::ZERO && signs[i] {
                        -r
                    } else {
                        r
                    };
                    if r != -Complex::ONE {
                        x_coeffs[i] *= (r + 1.0) / SQRT_2;
                        xs.unset(i, a);
                    } else {
                        if xs.get(i, a) {
                            x_coeffs[i] *= -2.0 / SQRT_2;
                        } else {
                            x_coeffs[i] *= 2.0 / SQRT_2;
                        }
                        xs.set(i, a);
                    }
                }
                g.apply_h_gate(a);
            }

            PizSequential(ci, cz, a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                if path[seen_p_gates] {
                    for i in 0..g.contained_states() {
                        if xs.get(i, a) {
                            x_coeffs[i] *= -Complex::ONE;
                        }
                        x_coeffs[i] *= cz;
                    }
                    g.apply_z_gate(a);
                } else {
                    for i in 0..g.contained_states() {
                        x_coeffs[i] *= ci;
                    }
                }
                seen_p_gates += 1;
            }
            PizFork(ci, cz, a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                let contained_states = g.contained_states();
                g.fork_apply_z_gate(a);
                for i in 0..contained_states {
                    let index_i = i;
                    let index_z = i + contained_states;
                    xs.copy_within(index_i, index_z);
                    x_coeffs.push(x_coeffs[index_i]);

                    x_coeffs[index_i] *= ci;

                    if xs.get(index_z, a) {
                        x_coeffs[index_z] *= -Complex::ONE;
                    }
                    x_coeffs[index_z] *= cz;
                }
                seen_p_gates += 1;
            }
            PizDeduplicate(c, ci, cz, a) => {
                let a: usize = a.try_into().expect(INDEX_TOO_LARGE);
                for (i1, i2) in xor_pairs(g.contained_states(), c) {
                    let mut z_coeff1 = x_coeffs[i2];
                    let mut z_coeff2 = x_coeffs[i1];
                    if xs.get(i2, a) {
                        z_coeff1 *= -Complex::ONE;
                    }
                    if xs.get(i1, a) {
                        z_coeff2 *= -Complex::ONE;
                    }
                    if !xs.equal_within(i1, i2) {
                        z_coeff1 *= g.coeff_ratio(&xs.slice_string(i2), &xs.slice_string(i1), i1);
                        z_coeff2 *= g.coeff_ratio(&xs.slice_string(i1), &xs.slice_string(i2), i2);
                    }
                    let new_coeff1 = ci * x_coeffs[i1] + cz * z_coeff1;
                    let new_coeff2 = ci * x_coeffs[i2] + cz * z_coeff2;
                    x_coeffs[i1] = new_coeff1;
                    x_coeffs[i2] = new_coeff2;
                }
                seen_p_gates += 1;
            }
        }
    }

    let mut w_coeff = Complex::ZERO;
    let contained_states = g.contained_states();
    let rs = g.coeff_ratios(&xs, &w.slice());
    for i in 0..contained_states {
        w_coeff += x_coeffs[i] * rs[i];
    }
    w_coeff
}

/// Mutate the given path to the next one.
/// Returns true if this results in the all-false path.
fn increment_path(path: &mut Vec<bool>) -> bool {
    for e in path {
        *e = !*e;
        if *e {
            return false;
        }
    }
    true
}

/// Iterate over all pairs `(i, i ^ c)` of indices into an array of length `len` (a power of two),
/// such that every index occurs in exactly one pair.
fn xor_pairs(len: usize, c: usize) -> impl Iterator<Item = (usize, usize)> {
    let (pair_count, low_mask) = if c == 0 {
        (len, usize::MAX)
    } else {
        (len / 2, c.isolate_lowest_one() - 1)
    };
    (0..pair_count).map(move |m| {
        let high_bits = (m & !low_mask) << 1;
        let low_bits = m & low_mask;
        let i = high_bits | low_bits;
        (i, i ^ c)
    })
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_xor_pairs1() {
        let res: Vec<_> = xor_pairs(0, 0).collect();
        assert_eq!(res, vec![]);
    }
    #[test]
    fn test_xor_pairs2() {
        let len = 1 << 7;
        let mut seen = vec![false; len];
        let c = 0b0110100;
        for (i1, i2) in xor_pairs(len, c) {
            assert_eq!(i1, i2 ^ c);
            assert!(!seen[i1]);
            assert!(!seen[i2]);
            seen[i1] = true;
            seen[i2] = true;
        }
        assert_eq!(seen, vec![true; len]);
    }
    #[test]
    fn test_xor_pairs3() {
        let len = 1 << 4;
        let mut seen = vec![false; len];
        let c = 0b1011;
        for (i1, i2) in xor_pairs(len, c) {
            assert_eq!(i1, i2 ^ c);
            assert!(!seen[i1]);
            assert!(!seen[i2]);
            seen[i1] = true;
            seen[i2] = true;
        }
        assert_eq!(seen, vec![true; len]);
    }
}
