use divan::{Bencher, black_box};
use qlit::{CliffordTCircuit, initialize_global};
use rand::{Rng, SeedableRng, rngs::SmallRng};

fn main() {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();
    initialize_global();
    divan::main();
}

fn setup(qubits: u32, gates: usize, t_gates: usize) -> (Vec<bool>, CliffordTCircuit) {
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

    #[divan::bench]
    fn cpu_small(bencher: Bencher) {
        let (w, circuit) = setup(8, 64, 5);
        bencher.bench_local(move || simulate_circuit(black_box(&w), black_box(&circuit)));
    }

    #[divan::bench(ignore = std::env::var("CI").is_ok())]
    fn cpu_large(bencher: Bencher) {
        let (w, circuit) = setup(32, 512, 17);
        bencher.bench_local(move || simulate_circuit(black_box(&w), black_box(&circuit)));
    }
}

mod gpu {
    use super::*;
    use qlit::simulate_circuit_gpu;

    #[divan::bench]
    fn gpu_small(bencher: Bencher) {
        let (w, circuit) = setup(8, 64, 5);
        bencher.bench_local(move || simulate_circuit_gpu(black_box(&w), black_box(&circuit)));
    }

    #[divan::bench(ignore = std::env::var("CI").is_ok())]
    fn gpu_large(bencher: Bencher) {
        let (w, circuit) = setup(32, 512, 17);
        bencher.bench_local(move || simulate_circuit_gpu(black_box(&w), black_box(&circuit)));
    }
}

mod hybrid {
    use super::*;
    use qlit::simulate_circuit_hybrid;

    #[divan::bench]
    fn hybrid_small(bencher: Bencher) {
        let (w, circuit) = setup(8, 64, 5);
        bencher.bench_local(move || simulate_circuit_hybrid(black_box(&w), black_box(&circuit)));
    }

    #[divan::bench(ignore = std::env::var("CI").is_ok())]
    fn hybrid_large(bencher: Bencher) {
        let (w, circuit) = setup(32, 512, 17);
        bencher.bench_local(move || simulate_circuit_hybrid(black_box(&w), black_box(&circuit)));
    }
}
