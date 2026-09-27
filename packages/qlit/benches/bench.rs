use criterion::Criterion;
use qlit::{CliffordTCircuit, initialize_global};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
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

fn main() {
    let mut c = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(30))
        .configure_from_args();

    // CPU
    use qlit::simulate_circuit;
    {
        let (w_small, circuit_small) = setup(8, 64, 5);
        c.bench_function("cpu_small", |b| {
            b.iter(|| simulate_circuit(black_box(&w_small), black_box(&circuit_small)))
        });
    }
    {
        let (w_large, circuit_large) = setup(32, 512, 15);
        c.bench_function("cpu_large", |b| {
            b.iter(|| simulate_circuit(black_box(&w_large), black_box(&circuit_large)))
        });
    }

    #[cfg(feature = "gpu")]
    {
        // GPU
        use qlit::simulate_circuit_gpu;
        {
            let (w_small, circuit_small) = setup(8, 64, 5);
            c.bench_function("gpu_small", |b| {
                b.iter(|| simulate_circuit_gpu(black_box(&w_small), black_box(&circuit_small)))
            });
        }
        {
            let (w_large, circuit_large) = setup(32, 512, 15);
            c.bench_function("gpu_large", |b| {
                b.iter(|| simulate_circuit_gpu(black_box(&w_large), black_box(&circuit_large)))
            });
        }

        // Hybrid
        use qlit::simulate_circuit_hybrid;
        {
            let (w_small, circuit_small) = setup(8, 64, 5);
            c.bench_function("hybrid_small", |b| {
                b.iter(|| simulate_circuit_hybrid(black_box(&w_small), black_box(&circuit_small)))
            });
        }
        {
            let (w_large, circuit_large) = setup(32, 512, 15);
            c.bench_function("hybrid_large", |b| {
                b.iter(|| simulate_circuit_hybrid(black_box(&w_large), black_box(&circuit_large)))
            });
        }
    }

    c.final_summary();
}
