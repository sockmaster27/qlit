use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use qlit::{CliffordTCircuit, initialize_global};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::hint::black_box;

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

    fn cpu_small(c: &mut Criterion) {
        let (w, circuit) = setup(8, 64, 5);
        c.bench_function("cpu_small", |b| {
            b.iter(|| simulate_circuit(black_box(&w), black_box(&circuit)))
        });
    }

    fn cpu_large(c: &mut Criterion) {
        let (w, circuit) = setup(32, 512, 17);
        c.bench_function("cpu_large", |b| {
            b.iter(|| simulate_circuit(black_box(&w), black_box(&circuit)))
        });
    }

    criterion_group!(benches, cpu_small, cpu_large);
}

mod gpu {
    use super::*;
    use qlit::simulate_circuit_gpu;

    fn gpu_small(c: &mut Criterion) {
        let (w, circuit) = setup(8, 64, 5);
        c.bench_function("gpu_small", |b| {
            b.iter(|| simulate_circuit_gpu(black_box(&w), black_box(&circuit)))
        });
    }

    fn gpu_large(c: &mut Criterion) {
        let (w, circuit) = setup(32, 512, 17);
        c.bench_function("gpu_large", |b| {
            b.iter(|| simulate_circuit_gpu(black_box(&w), black_box(&circuit)))
        });
    }

    criterion_group!(benches, gpu_small, gpu_large);
}

mod hybrid {
    use super::*;
    use qlit::simulate_circuit_hybrid;

    fn hybrid_small(c: &mut Criterion) {
        let (w, circuit) = setup(8, 64, 5);
        c.bench_function("hybrid_small", |b| {
            b.iter(|| simulate_circuit_hybrid(black_box(&w), black_box(&circuit)))
        });
    }

    fn hybrid_large(c: &mut Criterion) {
        let (w, circuit) = setup(32, 512, 17);
        c.bench_function("hybrid_large", |b| {
            b.iter(|| simulate_circuit_hybrid(black_box(&w), black_box(&circuit)))
        });
    }

    criterion_group!(benches, hybrid_small, hybrid_large);
}

criterion_main!(cpu::benches, gpu::benches, hybrid::benches);
