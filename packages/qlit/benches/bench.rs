use criterion::Criterion;
use qlit::{CliffordTCircuit, initialize_global};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::{hint::black_box, time::Duration};

fn setup(qubits: u32, gates: usize, t_gates: usize) -> (Vec<bool>, CliffordTCircuit) {
    let _ = rayon::ThreadPoolBuilder::new().build_global();
    initialize_global();

    let seed = 123;
    let rng = SmallRng::seed_from_u64(seed);
    let w = rng
        .random_iter()
        .take(usize::try_from(qubits).unwrap())
        .collect();
    let circuit = CliffordTCircuit::random(qubits, gates, t_gates, seed);
    (w, circuit)
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
}

fn main() {
    let mut c = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(10))
        .configure_from_args();

    cpu::cpu_small(&mut c);
    cpu::cpu_large(&mut c);
    #[cfg(feature = "gpu")]
    {
        gpu::gpu_small(&mut c);
        gpu::gpu_large(&mut c);
        hybrid::hybrid_small(&mut c);
        hybrid::hybrid_large(&mut c);
    }

    c.final_summary();
}
