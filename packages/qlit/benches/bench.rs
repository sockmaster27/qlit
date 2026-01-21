use divan::{Bencher, black_box};
use qlit::{CliffordTCircuit, initialize_global, simulate_circuit_gpu};
use rand::{Rng, SeedableRng, rngs::SmallRng};

const DEFAULT_QUBITS: usize = 100;
const DEFAULT_GATES: usize = 10;
const DEFAULT_T_GATES: usize = 5;

fn main() {
    initialize_global();
    divan::main();
}

fn setup(qubits: usize, gates: usize, t_gates: usize) -> (Vec<bool>, CliffordTCircuit) {
    let seed = 123;
    let rng = SmallRng::seed_from_u64(seed);
    let w = rng.random_iter().take(qubits).collect();
    let circuit = CliffordTCircuit::random(qubits, gates, t_gates, seed);
    (w, circuit)
}

#[divan::bench(args = [5, 10, 15])]
fn qubits(bencher: Bencher, qubits: usize) {
    let (w, circuit) = setup(qubits, DEFAULT_GATES, DEFAULT_T_GATES);
    bencher.bench_local(move || simulate_circuit_gpu(black_box(&w), black_box(&circuit)));
}

#[divan::bench(args = [5, 10, 15])]
fn gates(bencher: Bencher, gates: usize) {
    let (w, circuit) = setup(DEFAULT_QUBITS, gates, DEFAULT_T_GATES);
    bencher.bench_local(move || simulate_circuit_gpu(black_box(&w), black_box(&circuit)));
}

#[divan::bench(args = [0, 5, 6])]
fn t_gates(bencher: Bencher, t_gates: usize) {
    let (w, circuit) = setup(DEFAULT_QUBITS, DEFAULT_GATES, t_gates);
    bencher.bench_local(move || simulate_circuit_gpu(black_box(&w), black_box(&circuit)));
}
