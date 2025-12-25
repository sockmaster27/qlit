use qlit::{
    circuit::{CliffordTCircuit, CliffordTGate},
    simulate_circuit, simulate_circuit_parallel, simulate_circuit_parallel1,
    simulate_circuit_parallel2,
};

fn main() {
    divan::main();
}

// Helper function to create a sample circuit
fn create_sample_circuit(qubits: usize, depth: usize) -> CliffordTCircuit {
    let mut gates = Vec::new();
    
    // Add layers of gates
    for layer in 0..depth {
        // Add Hadamard gates
        for i in 0..qubits {
            gates.push(CliffordTGate::H(i));
        }
        
        // Add CNOT gates
        for i in 0..(qubits - 1) {
            gates.push(CliffordTGate::Cnot(i, i + 1));
        }
        
        // Add some T gates (fewer to keep simulation tractable)
        if layer % 2 == 0 {
            for i in 0..qubits.min(2) {
                gates.push(CliffordTGate::T(i));
            }
        }
    }
    
    CliffordTCircuit::new(qubits, gates)
}

#[divan::bench]
fn simulate_small_circuit(bencher: divan::Bencher) {
    let circuit = create_sample_circuit(4, 3);
    let w = vec![false; 4];
    
    bencher.bench(|| {
        simulate_circuit(&w, &circuit)
    });
}

#[divan::bench]
fn simulate_medium_circuit(bencher: divan::Bencher) {
    let circuit = create_sample_circuit(6, 4);
    let w = vec![false; 6];
    
    bencher.bench(|| {
        simulate_circuit(&w, &circuit)
    });
}

#[divan::bench]
fn simulate_small_circuit_parallel(bencher: divan::Bencher) {
    let circuit = create_sample_circuit(4, 3);
    let w = vec![false; 4];
    
    bencher.bench(|| {
        simulate_circuit_parallel(&w, &circuit)
    });
}

#[divan::bench]
fn simulate_medium_circuit_parallel(bencher: divan::Bencher) {
    let circuit = create_sample_circuit(6, 4);
    let w = vec![false; 6];
    
    bencher.bench(|| {
        simulate_circuit_parallel(&w, &circuit)
    });
}

#[divan::bench]
fn simulate_small_circuit_parallel1(bencher: divan::Bencher) {
    let circuit = create_sample_circuit(4, 3);
    let w = vec![false; 4];
    
    bencher.bench(|| {
        simulate_circuit_parallel1(&w, &circuit)
    });
}

#[divan::bench]
fn simulate_small_circuit_parallel2(bencher: divan::Bencher) {
    let circuit = create_sample_circuit(4, 3);
    let w = vec![false; 4];
    
    bencher.bench(|| {
        simulate_circuit_parallel2(&w, &circuit)
    });
}
