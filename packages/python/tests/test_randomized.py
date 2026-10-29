import json
import random
from pathlib import Path

import pytest
from qlit import (
    Circuit,
    simulate_circuit,
    simulate_circuit_gpu,
    simulate_circuit_hybrid,
)

FIXTURES_PATH = Path(__file__).parent / "randomized_fixtures.json"
DELTA = 1e-6
ABS_DELTA = 1e-6


class TestQlitRandomized:
    def setup_test_against_statevector(self):
        with open(FIXTURES_PATH, "r") as f:
            fixtures = json.load(f)
        for fixture in fixtures:
            circuit = Circuit.random(
                fixture["qubits"], 
                fixture["gates"], 
                fixture["non_clifford"], 
                fixture["seed"],
            )
            for w_data in fixture["w_data"]:
                w = w_data["w"]
                w_coeff_qiskit = complex(w_data["w_coeff_re"], w_data["w_coeff_im"])
                yield w, circuit, w_coeff_qiskit
    
    def test_against_statevector(self):
        for w, circuit, w_coeff_qiskit in self.setup_test_against_statevector():
            assert simulate_circuit(w, circuit) == pytest.approx(w_coeff_qiskit, rel=DELTA, abs=ABS_DELTA), f"\nCPU: {w=} \n{circuit.gates=}"
            
    @pytest.mark.gpu
    def test_against_statevector_gpu(self):
        for w, circuit, w_coeff_qiskit in self.setup_test_against_statevector():
            assert simulate_circuit_gpu(w, circuit) == pytest.approx(w_coeff_qiskit, rel=DELTA, abs=ABS_DELTA), f"\nGPU: {w=} \n{circuit.gates=}"

    @pytest.mark.gpu
    def test_against_statevector_hybrid(self):
        for w, circuit, w_coeff_qiskit in self.setup_test_against_statevector():
            assert simulate_circuit_hybrid(w, circuit) == pytest.approx(w_coeff_qiskit, rel=DELTA, abs=ABS_DELTA), f"\nHybrid: {w=} \n{circuit.gates=}"

    @pytest.mark.gpu
    def test_implementations_against_eachother(self):
        iterations = 5
        w_samples = 5
        for i in range(iterations):
            qubits = 130
            non_clifford = 3
            gates = 1000
            seed = 1234 + i
            circuit = Circuit.random(qubits, gates, non_clifford, seed)

            for _ in range(w_samples):
                w = format(random.getrandbits(qubits), f"00{qubits}b")
                simulate_circuit_res = simulate_circuit(w, circuit)
                simulate_circuit_gpu_res = simulate_circuit_gpu(w, circuit)
                simulate_circuit_hybrid_res = simulate_circuit_hybrid(w, circuit)

                assert simulate_circuit_res == pytest.approx(simulate_circuit_gpu_res, rel=DELTA, abs=ABS_DELTA), f"\nCPU/GPU: {w=} \n{circuit.gates=}"
                assert simulate_circuit_gpu_res == pytest.approx(simulate_circuit_hybrid_res, rel=DELTA, abs=ABS_DELTA), f"\nGPU/Hybrid: {w=} \n{circuit.gates=}"
