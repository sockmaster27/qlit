use divan::{Bencher, black_box};
use qlit::{circuit::CliffordTCircuit, simulate_circuit_parallel2};
use rand::{Rng, SeedableRng, rngs::SmallRng};

fn main() {
    divan::main();
}

fn setup(qubits: usize, gates: usize, t_gates: usize) -> (Vec<bool>, CliffordTCircuit) {
    let seed = 123;
    let rng = SmallRng::seed_from_u64(seed);
    let w = rng.random_iter().take(qubits).collect();
    let circuit = CliffordTCircuit::random(qubits, gates, t_gates, seed);
    (w, circuit)
}

#[divan::bench(args = [10, 100, 1000])]
fn qubits(bencher: Bencher, qubits: usize) {
    let gates = 1000;
    let t_gates = 5;
    let (w, circuit) = setup(qubits, gates, t_gates);
    bencher.bench_local(move || {
        simulate_circuit_parallel2(black_box(&w), black_box(&circuit));
    });
}

#[divan::bench(args = [100, 1000, 10000])]
fn gates(bencher: Bencher, gates: usize) {
    let qubits = 100;
    let t_gates = 5;
    let (w, circuit) = setup(qubits, gates, t_gates);
    bencher.bench_local(move || {
        simulate_circuit_parallel2(black_box(&w), black_box(&circuit));
    });
}

#[divan::bench(args = [0, 5, 10])]
fn t_gates(bencher: Bencher, t_gates: usize) {
    let qubits = 100;
    let gates = 1000;
    let (w, circuit) = setup(qubits, gates, t_gates);
    bencher.bench_local(move || {
        simulate_circuit_parallel2(black_box(&w), black_box(&circuit));
    });
}
