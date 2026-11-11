use criterion::Criterion;
use num_complex::Complex;
use qlit::{BitString, Circuit, Gate, initialize_global};
use rand::{Rng, RngExt, SeedableRng, rngs::Xoshiro128PlusPlus, seq::index::sample};
use std::{hint::black_box, time::Duration};

fn init() {
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    initialize_global();
}

fn setup(qubits: u32, gates: usize, non_clifford: usize) -> (BitString, Circuit) {
    init();

    // Xoshiro128++ (unlike SmallRng) is portable across platforms, so a given
    // seed always produces the same circuit regardless of pointer width.
    // Useful for CI testing and benchmarking.
    let rng = &mut Xoshiro128PlusPlus::seed_from_u64(123);
    let w = BitString::random(qubits.try_into().unwrap(), rng);
    let circuit = Circuit::random(qubits, gates, non_clifford, rng);
    (w, circuit)
}

/// Append a CCZ gate, compiled to Clifford+T using the standard 7 T gate phase-polynomial decomposition:
/// `4abc = a + b + c - (a^b) - (a^c) - (b^c) + (a^b^c)`.
fn ccz(gates: &mut Vec<Gate>, a: u32, b: u32, c: u32) {
    use Gate::*;
    gates.extend([
        T(a),
        T(b),
        T(c),
        Cnot(a, b),
        Tdg(b),
        Cnot(c, b),
        T(b),
        Cnot(a, b),
        Tdg(b),
        Cnot(c, b),
        Cnot(c, a),
        Tdg(a),
        Cnot(c, a),
    ]);
}

/// The hidden shift problem for Maiorana-McFarland bent functions, as used to benchmark
/// Clifford+T simulators in Bravyi & Gosset, "Improved classical simulation of quantum circuits dominated by Clifford gates" (2016).
///
/// On `2 * half_qubits` qubits (registers `x` and `y`), let `f(x, y) = x.y + h(y)`,
/// where `h` is a sum of `ccz_per_oracle` distinct random cubic terms in `y`.
/// This is bent, with dual `f~(x, y) = x.y + h(x)`.
/// Given oracle access to the shifted function `g(z) = f(z + s)`, the circuit
/// `H^n O_f~ H^n O_g H^n |0>` outputs exactly the hidden shift, `|s>`.
///
/// Each cubic term is a CCZ gate costing 7 T gates, and both oracles contain `ccz_per_oracle` of them,
/// for a total T count of `14 * ccz_per_oracle`.
///
/// Returns the hidden shift and the circuit, so the amplitude of the shift in the final state is exactly 1.
fn hidden_shift(
    half_qubits: u32,
    ccz_per_oracle: usize,
    rng: &mut impl Rng,
) -> (BitString, Circuit) {
    use Gate::*;

    let m = half_qubits;
    let n = 2 * m;
    let m_us = m as usize;
    assert!(
        ccz_per_oracle <= m_us * (m_us - 1) * (m_us - 2) / 6,
        "not enough distinct triples of qubits"
    );

    let shift: Vec<bool> = (0..n).map(|_| rng.random()).collect();

    let mut triples: Vec<[u32; 3]> = Vec::new();
    while triples.len() < ccz_per_oracle {
        let mut triple = [0; 3];
        for (t, i) in triple.iter_mut().zip(sample(rng, m_us, 3)) {
            *t = i as u32;
        }
        triple.sort();
        if !triples.contains(&triple) {
            triples.push(triple);
        }
    }

    // Applying X on the shifted bits before and after O_f turns it into O_g.
    let shift_gates: Vec<Gate> = (0..n).filter(|&i| shift[i as usize]).map(X).collect();
    let mut gates = Vec::new();

    gates.extend((0..n).map(H));

    // O_g
    gates.extend(&shift_gates);
    gates.extend((0..m).map(|i| Cz(i, m + i)));
    for [a, b, c] in &triples {
        ccz(&mut gates, m + a, m + b, m + c);
    }
    gates.extend(&shift_gates);

    gates.extend((0..n).map(H));

    // O_f~
    gates.extend((0..m).map(|i| Cz(i, m + i)));
    for [a, b, c] in &triples {
        ccz(&mut gates, *a, *b, *c);
    }

    gates.extend((0..n).map(H));

    (
        BitString::from(shift),
        Circuit::new(n, gates).expect("generated circuit is valid"),
    )
}

/// Sized to finish within a few seconds on both CPU and GPU.
fn setup_hidden_shift() -> (BitString, Circuit) {
    init();

    let rng = &mut Xoshiro128PlusPlus::seed_from_u64(123);
    hidden_shift(17, 64, rng)
}

/// The amplitude of the hidden shift is exactly 1.
/// The tolerance accommodates the single precision used on the GPU.
fn check_hidden_shift(amplitude: Complex<f64>) {
    assert!(
        (amplitude - 1.0).norm() < 1e-3,
        "hidden shift amplitude should be 1, got {amplitude}"
    );
}

mod cpu {
    use super::*;
    use qlit::simulate_circuit;

    pub fn cpu_small(c: &mut Criterion) {
        let (w, circuit) = setup(8, 64, 5);
        c.bench_function("cpu_small", |b| {
            b.iter(|| simulate_circuit(black_box(&w), black_box(&circuit)))
        });
    }

    pub fn cpu_large(c: &mut Criterion) {
        let (w, circuit) = setup(32, 512, 15);
        c.bench_function("cpu_large", |b| {
            b.iter(|| simulate_circuit(black_box(&w), black_box(&circuit)))
        });
    }

    pub fn cpu_hidden_shift(c: &mut Criterion) {
        let (w, circuit) = setup_hidden_shift();
        check_hidden_shift(simulate_circuit(&w, &circuit));
        c.bench_function("cpu_hidden_shift", |b| {
            b.iter(|| simulate_circuit(black_box(&w), black_box(&circuit)))
        });
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use qlit::simulate_circuit_gpu;

    pub fn gpu_small(c: &mut Criterion) {
        let (w, circuit) = setup(8, 64, 5);
        c.bench_function("gpu_small", |b| {
            b.iter(|| simulate_circuit_gpu(black_box(&w), black_box(&circuit)))
        });
    }

    pub fn gpu_large(c: &mut Criterion) {
        let (w, circuit) = setup(32, 512, 15);
        c.bench_function("gpu_large", |b| {
            b.iter(|| simulate_circuit_gpu(black_box(&w), black_box(&circuit)))
        });
    }

    pub fn gpu_hidden_shift(c: &mut Criterion) {
        let (w, circuit) = setup_hidden_shift();
        check_hidden_shift(simulate_circuit_gpu(&w, &circuit));
        c.bench_function("gpu_hidden_shift", |b| {
            b.iter(|| simulate_circuit_gpu(black_box(&w), black_box(&circuit)))
        });
    }
}

#[cfg(feature = "gpu")]
mod hybrid {
    use super::*;
    use qlit::simulate_circuit_hybrid;

    pub fn hybrid_small(c: &mut Criterion) {
        let (w, circuit) = setup(8, 64, 5);
        c.bench_function("hybrid_small", |b| {
            b.iter(|| simulate_circuit_hybrid(black_box(&w), black_box(&circuit)))
        });
    }

    pub fn hybrid_large(c: &mut Criterion) {
        let (w, circuit) = setup(32, 512, 15);
        c.bench_function("hybrid_large", |b| {
            b.iter(|| simulate_circuit_hybrid(black_box(&w), black_box(&circuit)))
        });
    }

    pub fn hybrid_hidden_shift(c: &mut Criterion) {
        let (w, circuit) = setup_hidden_shift();
        check_hidden_shift(simulate_circuit_hybrid(&w, &circuit));
        c.bench_function("hybrid_hidden_shift", |b| {
            b.iter(|| simulate_circuit_hybrid(black_box(&w), black_box(&circuit)))
        });
    }
}

fn main() {
    let mut c = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(60))
        .configure_from_args();

    cpu::cpu_small(&mut c);
    cpu::cpu_large(&mut c);
    cpu::cpu_hidden_shift(&mut c);
    #[cfg(feature = "gpu")]
    {
        gpu::gpu_small(&mut c);
        gpu::gpu_large(&mut c);
        gpu::gpu_hidden_shift(&mut c);
        hybrid::hybrid_small(&mut c);
        hybrid::hybrid_large(&mut c);
        hybrid::hybrid_hidden_shift(&mut c);
    }

    c.final_summary();
}
